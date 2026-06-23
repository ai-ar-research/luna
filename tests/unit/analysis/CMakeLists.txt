# tests/unit/analysis/CMakeLists.txt
# Analysis unit tests

function(add_analysis_test TEST_NAME)
    luna_add_test(${TEST_NAME} ${TEST_NAME}.cpp "analysis" "unit")
endfunction()

add_analysis_test(test_crown_analysis)
add_analysis_test(test_alpha_crown_analysis)
add_analysis_test(test_gpu_device)
add_analysis_test(test_bab_state)