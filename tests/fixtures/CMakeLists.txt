# tests/fixtures/CMakeLists.txt
# Build shared test fixture library

set(FIXTURES_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Test fixture library sources
set(FIXTURES_SOURCES
    ${FIXTURES_DIR}/model_builders.cpp
    ${FIXTURES_DIR}/test_utils.cpp
    ${FIXTURES_DIR}/tensor_comparators.cpp
)

# Create static library for test fixtures
add_library(TestFixtures STATIC ${FIXTURES_SOURCES})

target_include_directories(TestFixtures PUBLIC
    ${PROJECT_SOURCE_DIR}  # Project root for "src/..." includes
    ${FIXTURES_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/engine
    ${PROJECT_SOURCE_DIR}/src/engine/nodes
    ${PROJECT_SOURCE_DIR}/src/input_parsers
    ${PROJECT_SOURCE_DIR}/src/configuration
)

target_compile_options(TestFixtures PRIVATE ${TEST_COMPILE_FLAGS})

# Link libraries (combine into one call)
target_link_libraries(TestFixtures PUBLIC
    ${LUNA_ONNX_PARSER_LIB}
    ${LUNA_LIB}
    GTest::gtest
    ${LIBS}
)
