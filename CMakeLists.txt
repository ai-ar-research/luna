cmake_minimum_required(VERSION 3.16)
project(LUNA)

set(LUNA_VERSION 1.0.0)
add_definitions("-DLUNA_VERSION=\"${LUNA_VERSION}\"")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

##################
## User options ##
##################

option(BUILD_STATIC_LUNA "Build static LUNA binary" OFF)
option(ENABLE_OPENBLAS "Do symbolic bound tightening using blas" ON)
option(RUN_UNIT_TEST "Build and register unit tests" ON)
option(RUN_REGRESS_TEST "Build and register regression tests" OFF)
option(RUN_SYSTEM_TEST "Build and register system tests" OFF)
option(RUN_PYTHON_TEST "Enable Python-based regression tests" OFF)

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
add_definitions("-DGIT_BRANCH=\"${GIT_BRANCH}\"")

execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
add_definitions("-DGIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")

#########################
## Basic configuration ##
#########################

set(TOOLS_DIR "${PROJECT_SOURCE_DIR}/tools")
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(RESOURCES_DIR "${PROJECT_SOURCE_DIR}/resources")
set(COMMON_DIR "${SRC_DIR}/common")
set(ENGINE_DIR "${SRC_DIR}/engine")
set(INPUT_PARSERS_DIR "${SRC_DIR}/input_parsers")

if (MSVC)
  set(SCRIPT_EXTENSION bat)
else()
  set(SCRIPT_EXTENSION sh)
endif()

###########
## Boost ##
###########

add_definitions(-DBOOST_NO_CXX98_FUNCTION_BASE)
set(BOOST_VERSION 1.84.0)
set(BOOST_DIR "${TOOLS_DIR}/boost-${BOOST_VERSION}")

if (MSVC)
  set(BOOST_ROOT "${BOOST_DIR}/win_installed")
  set(Boost_NAMESPACE libboost)
elseif (${CMAKE_SIZEOF_VOID_P} EQUAL 4 AND NOT MSVC)
  set(BOOST_ROOT "${BOOST_DIR}/installed32")
else()
  set(BOOST_ROOT "${BOOST_DIR}/installed")
endif()

set(Boost_USE_DEBUG_RUNTIME FALSE)
find_package(Boost ${BOOST_VERSION} COMPONENTS program_options timer chrono thread regex)

if (NOT ${Boost_FOUND})
  execute_process(COMMAND ${TOOLS_DIR}/download_boost.${SCRIPT_EXTENSION} ${BOOST_VERSION})
  find_package(Boost ${BOOST_VERSION} REQUIRED COMPONENTS program_options timer chrono thread regex)
endif()

set(LIBS_INCLUDES ${Boost_INCLUDE_DIRS})
list(APPEND LIBS ${Boost_LIBRARIES})

##############
## Protobuf ##
##############

set(PROTOBUF_VERSION 3.19.2)
set(PROTOBUF_DEFAULT_DIR "${TOOLS_DIR}/protobuf-${PROTOBUF_VERSION}")

if (NOT PROTOBUF_DIR)
    set(PROTOBUF_DIR ${PROTOBUF_DEFAULT_DIR})
endif()

if(NOT EXISTS "${PROTOBUF_DIR}/installed/lib/libprotobuf.a")
    message("Can't find protobuf, installing...")
    if (${PROTOBUF_DIR} STREQUAL ${PROTOBUF_DEFAULT_DIR})
        execute_process(COMMAND ${TOOLS_DIR}/download_protobuf.sh ${PROTOBUF_VERSION})
    else()
        message(FATAL_ERROR "Can't find protobuf in the supplied directory")
    endif()
endif()

set(PROTOBUF_LIB protobuf)
add_library(${PROTOBUF_LIB} SHARED IMPORTED)
set_property(TARGET ${PROTOBUF_LIB} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROTOBUF_LIB} PROPERTIES IMPORTED_LOCATION ${PROTOBUF_DIR}/installed/lib/libprotobuf.a)
target_include_directories(${PROTOBUF_LIB} INTERFACE ${PROTOBUF_DIR}/installed/include)
list(APPEND LIBS ${PROTOBUF_LIB})

##########
## ONNX ##
##########

set(ONNX_VERSION 1.15.0)
set(ONNX_DIR "${TOOLS_DIR}/onnx-${ONNX_VERSION}")

if(NOT EXISTS "${ONNX_DIR}/onnx.proto3.pb.h")
    message("Generating ONNX protobuf file...")
    execute_process(COMMAND ${TOOLS_DIR}/download_onnx.sh ${ONNX_VERSION} ${PROTOBUF_VERSION})
endif()

file(GLOB DEPS_ONNX "${ONNX_DIR}/*.cc")
include_directories(SYSTEM ${ONNX_DIR})

#############
## PyTorch ##
#############

set(PYTORCH_VERSION 2.2.1)

# First, try to find PyTorch from the active Python environment (handles CUDA correctly)
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
    # Fallback to local libtorch
    set(PYTORCH_DIR "${TOOLS_DIR}/libtorch-${PYTORCH_VERSION}")
    list(APPEND CMAKE_PREFIX_PATH ${PYTORCH_DIR})
    if(NOT EXISTS "${PYTORCH_DIR}")
        execute_process(COMMAND ${TOOLS_DIR}/download_libtorch.sh ${PYTORCH_VERSION})
    endif()
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
list(APPEND LIBS ${TORCH_LIBRARIES})

##############
## pybind11 ##
##############

option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" ON)

if(BUILD_PYTHON_BINDINGS)
  set(PYBIND11_VERSION 2.11.1)
  set(PYBIND11_DIR "${TOOLS_DIR}/pybind11-${PYBIND11_VERSION}")
  
  if(NOT EXISTS "${PYBIND11_DIR}/include/pybind11/pybind11.h")
    message("Can't find pybind11, downloading...")
    execute_process(COMMAND ${TOOLS_DIR}/download_pybind11.sh ${PYBIND11_VERSION})
  endif()
  
  # Add pybind11 subdirectory
  add_subdirectory(${PYBIND11_DIR} ${CMAKE_BINARY_DIR}/pybind11)
  message(STATUS "Building Python bindings with pybind11")
endif()

##############
## OpenBLAS ##
##############

if (NOT MSVC AND ${ENABLE_OPENBLAS})
  set(OPENBLAS_VERSION 0.3.19)
  set(OPENBLAS_LIB openblas)
  set(OPENBLAS_DEFAULT_DIR "${TOOLS_DIR}/OpenBLAS-${OPENBLAS_VERSION}")

  if (NOT OPENBLAS_DIR)
      set(OPENBLAS_DIR ${OPENBLAS_DEFAULT_DIR})
  endif()

  message(STATUS "Using OpenBLAS for matrix multiplication")
  add_compile_definitions(ENABLE_OPENBLAS)

  if(NOT EXISTS "${OPENBLAS_DIR}/installed/lib/libopenblas.a")
    message("Can't find OpenBLAS, installing...")
    if (${OPENBLAS_DIR} STREQUAL ${OPENBLAS_DEFAULT_DIR})
      execute_process(COMMAND ${TOOLS_DIR}/download_openBLAS.sh ${OPENBLAS_VERSION})
    else()
      message(FATAL_ERROR "Can't find OpenBLAS in the supplied directory")
    endif()
  endif()

  add_library(${OPENBLAS_LIB} SHARED IMPORTED)
  set_target_properties(${OPENBLAS_LIB} PROPERTIES IMPORTED_LOCATION ${OPENBLAS_DIR}/installed/lib/libopenblas.a)
  list(APPEND LIBS ${OPENBLAS_LIB})
  target_include_directories(${OPENBLAS_LIB} INTERFACE ${OPENBLAS_DIR}/installed/include)
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

add_definitions(-DRESOURCES_DIR="${RESOURCES_DIR}")

# pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
list(APPEND LIBS Threads::Threads)

# Create core library with ONNX dependencies
# Subdirectories will add their sources via target_sources()
add_library(${LUNA_LIB} ${DEPS_ONNX})

# Set base include directories
target_include_directories(${LUNA_LIB} PUBLIC
    ${SRC_DIR}
    ${LIBS_INCLUDES}
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

# Build main luna executable
add_executable(luna src/main.cpp src/engine/LunaMain.cpp)
target_link_libraries(luna ${LUNA_LIB} ${LIBS})
target_include_directories(luna PRIVATE
    ${SRC_DIR}
    ${LIBS_INCLUDES}
)
target_compile_options(luna PRIVATE ${RELEASE_FLAGS})

# Copy luna executable to bin directory
add_custom_command(TARGET luna POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BIN_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:luna> ${BIN_DIR}/
)

# Install luna executable to user-local bin directory (~/.local/bin)
# This directory is typically already in PATH on macOS/Linux
set(USER_LOCAL_BIN "$ENV{HOME}/.local/bin")
install(TARGETS luna
    RUNTIME DESTINATION ${USER_LOCAL_BIN}
    COMPONENT Runtime
)

# Automatically install to ~/.local/bin after build (so user can run 'luna' from anywhere)
add_custom_command(TARGET luna POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${USER_LOCAL_BIN}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:luna> ${USER_LOCAL_BIN}/
    COMMENT "Installing luna to ${USER_LOCAL_BIN} (add to PATH if needed)"
)

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
    include(FetchContent)
    FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

if(BUILD_TESTS)
    list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
    list(APPEND CMAKE_PREFIX_PATH "${TOOLS_DIR}/cxxtest")
    find_package(CxxTest)
    if(NOT CXXTEST_FOUND)
        message(FATAL_ERROR "CxxTest not found. Expected at ${TOOLS_DIR}/cxxtest")
    endif()
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
message(STATUS "  - LUNA Version: ${LUNA_VERSION}")
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - PyTorch version: ${PYTORCH_VERSION}")
message(STATUS "  - Boost version: ${BOOST_VERSION}")
message(STATUS "  - Build tests: ${BUILD_TESTS}")
message(STATUS "  - Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  - Build Python bindings: ${BUILD_PYTHON_BINDINGS}")