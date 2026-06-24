cmake_minimum_required(VERSION 3.24)
project(LUNA VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##################
## User options ##
##################

option(BUILD_STATIC_LUNA "Build static LUNA binary" OFF)
option(RUN_UNIT_TEST "Build and register unit tests" ON)
option(RUN_REGRESS_TEST "Build and register regression tests" OFF)
option(RUN_SYSTEM_TEST "Build and register system tests" OFF)
option(RUN_PYTHON_TEST "Enable Python-based regression tests" OFF)
# Default ON so standalone builds keep producing the `luna` CLI; embedding
# projects flip this OFF to consume only the LUNACore library / luna::core
# alias without paying for the executable build or its install side-effects.
option(LUNA_BUILD_EXECUTABLE "Build the luna CLI executable" ON)

###################
## Git variables ##
###################

execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

#########################
## Basic configuration ##
#########################

set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(RESOURCES_DIR "${PROJECT_SOURCE_DIR}/resources")
set(COMMON_DIR "${SRC_DIR}/common")
set(ENGINE_DIR "${SRC_DIR}/engine")
set(INPUT_PARSERS_DIR "${SRC_DIR}/input_parsers")

include(FetchContent)
include(cmake/versions.cmake)

##############
## Protobuf ##
##############

# Protobuf must be declared before libtorch to prevent target conflicts.
# Prefer a system / conda Protobuf when one is available so embedders don't
# end up with two protobuf static archives (Luna's bundled 3.19.2 plus the
# host project's) in the final link line. FetchContent is the fallback for
# standalone builds in environments without a usable Protobuf install.
set(PROTOBUF_VERSION ${LUNA_PROTOBUF_VERSION})

find_package(Protobuf CONFIG QUIET)

if(Protobuf_FOUND AND TARGET protobuf::libprotobuf AND TARGET protobuf::protoc)
    message(STATUS "Found system Protobuf ${Protobuf_VERSION} via CONFIG; skipping FetchContent")
    set(LUNA_PROTOBUF_TARGET protobuf::libprotobuf)
    set(LUNA_PROTOC_TARGET protobuf::protoc)
else()
    message(STATUS "System Protobuf not found; fetching ${PROTOBUF_VERSION} via FetchContent")

    set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)

    # protobuf 3.19.x has cmake_minimum_required(VERSION 2.8.12)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

    FetchContent_Declare(protobuf
        URL https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-cpp-${PROTOBUF_VERSION}.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR cmake
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(protobuf)

    unset(CMAKE_POLICY_VERSION_MINIMUM)

    set(LUNA_PROTOBUF_TARGET libprotobuf)
    set(LUNA_PROTOC_TARGET protoc)
endif()

list(APPEND LIBS ${LUNA_PROTOBUF_TARGET})

##########
## ONNX ##
##########

set(ONNX_VERSION ${LUNA_ONNX_VERSION})
set(ONNX_PROTO_DIR "${CMAKE_BINARY_DIR}/onnx-${ONNX_VERSION}")
file(MAKE_DIRECTORY ${ONNX_PROTO_DIR})

if(NOT EXISTS "${ONNX_PROTO_DIR}/onnx.proto3")
    message(STATUS "Downloading ONNX proto file v${ONNX_VERSION}...")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/onnx/onnx/v${ONNX_VERSION}/onnx/onnx.proto3"
        "${ONNX_PROTO_DIR}/onnx.proto3"
        STATUS ONNX_DOWNLOAD_STATUS
    )
    list(GET ONNX_DOWNLOAD_STATUS 0 ONNX_DOWNLOAD_RESULT)
    if(NOT ONNX_DOWNLOAD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to download ONNX proto file")
    endif()
endif()

add_custom_command(
    OUTPUT "${ONNX_PROTO_DIR}/onnx.proto3.pb.cc" "${ONNX_PROTO_DIR}/onnx.proto3.pb.h"
    COMMAND ${LUNA_PROTOC_TARGET} --cpp_out=${ONNX_PROTO_DIR} -I${ONNX_PROTO_DIR} onnx.proto3
    WORKING_DIRECTORY ${ONNX_PROTO_DIR}
    DEPENDS "${ONNX_PROTO_DIR}/onnx.proto3" ${LUNA_PROTOC_TARGET}
    COMMENT "Compiling ONNX protobuf"
)

set(DEPS_ONNX "${ONNX_PROTO_DIR}/onnx.proto3.pb.cc")

#############
## PyTorch ##
#############

set(PYTORCH_VERSION ${LUNA_PYTORCH_VERSION})
option(LUNA_GPU "Enable CUDA libtorch" OFF)
option(LUNA_AUTO_DOWNLOAD "Download dependencies via FetchContent when missing" OFF)

# Try to find PyTorch from the active Python environment first
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PYTHON_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE TORCH_PYTHON_RESULT
)

if(TORCH_PYTHON_RESULT EQUAL 0 AND TORCH_PYTHON_PREFIX)
    message(STATUS "Found PyTorch from Python environment: ${TORCH_PYTHON_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_PYTHON_PREFIX})
else()
    # Embedders (e.g. Marabou3) forward their own CMAKE_PREFIX_PATH that
    # already points at a usable libtorch; honour that before forcing a
    # download. find_package below issues REQUIRED, so a real miss still
    # FATALs -- this just unlocks the embedder path.
    find_package(Torch QUIET CONFIG)
endif()

if(NOT Torch_FOUND AND NOT TORCH_PYTHON_PREFIX)
    if(LUNA_AUTO_DOWNLOAD)
        if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${PYTORCH_VERSION}.zip")
        elseif(APPLE)
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-${PYTORCH_VERSION}.zip")
        elseif(LUNA_GPU)
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcu121.zip")
        else()
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
        endif()

        message(STATUS "Downloading libtorch from ${LIBTORCH_URL}")
        FetchContent_Declare(libtorch
            URL ${LIBTORCH_URL}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(libtorch)
        list(APPEND CMAKE_PREFIX_PATH ${libtorch_SOURCE_DIR})
    else()
        message(FATAL_ERROR
            "PyTorch not found via Python, embedder CMAKE_PREFIX_PATH, or LUNA_AUTO_DOWNLOAD.\n"
            "Either install PyTorch in your Python environment (pip install torch),\n"
            "pass -DCMAKE_PREFIX_PATH=/path/to/libtorch, or re-run with -DLUNA_AUTO_DOWNLOAD=ON."
        )
    endif()
endif()

# Re-prepend conda prefix after libtorch discovery to prefer full shared libraries
if(DEFINED ENV{CONDA_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
list(APPEND LIBS ${TORCH_LIBRARIES})

##############
## pybind11 ##
##############

option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" ON)

if(BUILD_PYTHON_BINDINGS)
  set(PYBIND11_VERSION ${LUNA_PYBIND11_VERSION})

  FetchContent_Declare(pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v${PYBIND11_VERSION}
      GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(pybind11)
  message(STATUS "Building Python bindings with pybind11")
endif()

###########
## Build ##
###########

set(LUNA_LIB LUNACore)
set(BIN_DIR "${CMAKE_BINARY_DIR}/bin")

# Compiler flags
if (NOT MSVC)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(COMPILE_FLAGS -Wall -Wextra -MMD -Qunused-arguments -Wno-deprecated-declarations -Wno-unused-but-set-variable)
    elseif (CMAKE_BUILD_TYPE MATCHES "Release")
        set(COMPILE_FLAGS -Wall)
    else()
        set(COMPILE_FLAGS -Wall -Wextra -MMD)
    endif()
    set(RELEASE_FLAGS ${COMPILE_FLAGS} -O3)
endif()

# pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
list(APPEND LIBS Threads::Threads)

# Create core library with ONNX dependencies
add_library(${LUNA_LIB} ${DEPS_ONNX})
# Namespaced alias so embedders link `luna::core` and get a clear error if
# the target ever moves, instead of silently linking against a same-named
# library that happens to be on the link line.
add_library(luna::core ALIAS ${LUNA_LIB})

target_compile_definitions(${LUNA_LIB} PUBLIC
    LUNA_VERSION="${PROJECT_VERSION}"
    GIT_BRANCH="${GIT_BRANCH}"
    GIT_COMMIT_HASH="${GIT_COMMIT_HASH}"
    RESOURCES_DIR="${RESOURCES_DIR}"
)

target_include_directories(${LUNA_LIB} PUBLIC
    ${SRC_DIR}
)
target_include_directories(${LUNA_LIB} SYSTEM PUBLIC
    ${ONNX_PROTO_DIR}
)

# Link libraries
if (BUILD_STATIC_LUNA)
  target_link_libraries(${LUNA_LIB} PUBLIC ${LIBS} -static)
else()
  target_link_libraries(${LUNA_LIB} PUBLIC ${LIBS})
endif()

target_compile_options(${LUNA_LIB} PRIVATE ${RELEASE_FLAGS})

# Add subdirectories - each will populate LUNACore using target_sources()
add_subdirectory(src/common)
add_subdirectory(src/engine)
add_subdirectory(src/input_parsers)
add_subdirectory(src/configuration)

if(LUNA_BUILD_EXECUTABLE)
    add_executable(luna src/main.cpp src/engine/LunaMain.cpp)
    target_link_libraries(luna ${LUNA_LIB} ${LIBS})
    target_include_directories(luna PRIVATE
        ${SRC_DIR}
    )
    target_compile_options(luna PRIVATE ${RELEASE_FLAGS})

    add_custom_command(TARGET luna POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BIN_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:luna> ${BIN_DIR}/
    )

    # Mirror the CLI into ~/.local/bin so a standalone Luna build "just works"
    # on PATH. Embedders set LUNA_BUILD_EXECUTABLE=OFF to skip this entirely;
    # users who don't want the home-dir side-effect can also flip it off and
    # rely on the BIN_DIR copy above plus a normal `cmake --install`.
    set(USER_LOCAL_BIN "$ENV{HOME}/.local/bin")
    install(TARGETS luna
        RUNTIME DESTINATION ${USER_LOCAL_BIN}
        COMPONENT Runtime
    )

    add_custom_command(TARGET luna POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${USER_LOCAL_BIN}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:luna> ${USER_LOCAL_BIN}/
        COMMENT "Installing luna to ${USER_LOCAL_BIN} (add to PATH if needed)"
    )
endif()

###################
## Testing Setup ##
###################

option(BUILD_TESTS "Build the test suite" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)

if(BUILD_TESTS)
    if(NOT RUN_UNIT_TEST AND NOT RUN_REGRESS_TEST AND NOT RUN_SYSTEM_TEST AND NOT RUN_PYTHON_TEST)
        set(RUN_UNIT_TEST ON)
    endif()
else()
    set(RUN_UNIT_TEST OFF CACHE BOOL "" FORCE)
    set(RUN_REGRESS_TEST OFF CACHE BOOL "" FORCE)
    set(RUN_SYSTEM_TEST OFF CACHE BOOL "" FORCE)
    set(RUN_PYTHON_TEST OFF CACHE BOOL "" FORCE)
endif()

if(BUILD_BENCHMARKS)
    FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v${LUNA_BENCHMARK_VERSION})
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

if(BUILD_TESTS)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v${LUNA_GTEST_VERSION}
        GIT_SHALLOW TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    add_subdirectory(tests)
endif()

#######################
## Python Bindings   ##
#######################

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(lunapy)
endif()

message(STATUS "LUNA configured successfully")
message(STATUS "  - LUNA Version: ${PROJECT_VERSION}")
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - GPU mode: ${LUNA_GPU}")
message(STATUS "  - PyTorch version: ${PYTORCH_VERSION}")
message(STATUS "  - Build tests: ${BUILD_TESTS}")
message(STATUS "  - Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  - Build Python bindings: ${BUILD_PYTHON_BINDINGS}")
