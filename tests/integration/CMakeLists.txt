# tests/integration/CMakeLists.txt
# Integration test configuration (system tests and helper executables)

set(INTEGRATION_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Keep existing integration test executables (original CMakeLists.txt content)
function(add_integration_executable TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${INTEGRATION_TEST_DIR}/${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} ${LUNA_LIB})
    target_include_directories(${TEST_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/common
        ${PROJECT_SOURCE_DIR}/src/engine
        ${PROJECT_SOURCE_DIR}/src/engine/nodes
        ${PROJECT_SOURCE_DIR}/src/engine/conv
        ${PROJECT_SOURCE_DIR}/src/input_parsers
        ${PROJECT_SOURCE_DIR}/src/configuration
        ${LIBS_INCLUDES}
    )
    target_compile_options(${TEST_NAME} PRIVATE ${RELEASE_FLAGS})
endfunction()

# Runner executables
add_integration_executable(RunCROWN RunCROWN.cpp)
add_integration_executable(RunAlphaCROWN RunAlphaCROWN.cpp)
add_integration_executable(RunFull RunFull.cpp)
add_integration_executable(RunSpec RunSpec.cpp)
add_integration_executable(RunFullOptimized RunFullOptimized.cpp)

# Test executables
add_integration_executable(TestFlatten TestFlatten.cpp)
add_integration_executable(TestConvDiagnostics TestConvDiagnostics.cpp)
add_integration_executable(TestBatchNorm TestBatchNorm.cpp)
add_integration_executable(TestAdd TestAdd.cpp)
add_integration_executable(TestResidualCrownSoundness TestResidualCrownSoundness.cpp)

# System test executables
luna_add_test(test_crown_pipeline test_crown_pipeline.cpp "integration" "system")
luna_add_test(test_ibp_pipeline test_ibp_pipeline.cpp "integration" "system")
luna_add_test(test_alpha_crown_pipeline test_alpha_crown_pipeline.cpp "integration" "system")
luna_add_test(test_onnx_vnnlib_pipeline test_onnx_vnnlib_pipeline.cpp "integration" "system")
