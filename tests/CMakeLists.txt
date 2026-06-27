set(TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TESTS_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(TEST_DIR "${CMAKE_BINARY_DIR}/tests")

set(TEST_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/engine
    ${PROJECT_SOURCE_DIR}/src/engine/nodes
    ${PROJECT_SOURCE_DIR}/src/engine/conv
    ${PROJECT_SOURCE_DIR}/src/input_parsers
    ${PROJECT_SOURCE_DIR}/src/configuration
)

if (NOT MSVC)
    set(TEST_COMPILE_FLAGS ${COMPILE_FLAGS})
    string(REPLACE "-O3" "-O2" TEST_COMPILE_FLAGS "${TEST_COMPILE_FLAGS}")
else()
    set(TEST_COMPILE_FLAGS "")
endif()

function(luna_add_test TEST_NAME TEST_SOURCE TEST_LABEL TEST_TIER)
    set(TEST_SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SOURCE})

    add_executable(${TEST_NAME} ${TEST_SOURCE_PATH})

    target_include_directories(${TEST_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tests
        ${TEST_INCLUDE_DIRS}
    )

    target_link_libraries(${TEST_NAME}
        TestFixtures
        ${LUNA_ONNX_PARSER_LIB}
        ${LUNA_LIB}
        GTest::gtest
        GTest::gtest_main
        ${LIBS}
    )

    target_compile_options(${TEST_NAME} PRIVATE ${TEST_COMPILE_FLAGS})

    gtest_discover_tests(${TEST_NAME}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        PROPERTIES LABELS "${TEST_LABEL};${TEST_TIER}"
    )

    add_custom_command(TARGET ${TEST_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_DIR}/${TEST_TIER}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TEST_NAME}> ${TEST_DIR}/${TEST_TIER}/
    )

    set_property(GLOBAL APPEND PROPERTY LUNA_TEST_TARGETS ${TEST_NAME})
endfunction()

include(GoogleTest)

if(RUN_UNIT_TEST OR RUN_SYSTEM_TEST OR RUN_REGRESS_TEST)
    add_subdirectory(fixtures)
endif()

if(RUN_UNIT_TEST)
    add_subdirectory(unit)
    add_subdirectory(property)
endif()

if(RUN_SYSTEM_TEST)
    add_subdirectory(integration)
    add_subdirectory(soundness)
endif()

if(RUN_REGRESS_TEST)
    add_subdirectory(regression)
    add_subdirectory(reference)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

include(ProcessorCount)
ProcessorCount(CTEST_NTHREADS)
if(NOT CTEST_NTHREADS OR CTEST_NTHREADS EQUAL 0)
    set(CTEST_NTHREADS 1)
endif()

get_property(LUNA_TEST_TARGETS GLOBAL PROPERTY LUNA_TEST_TARGETS)
if(LUNA_TEST_TARGETS)
    add_custom_target(build-tests DEPENDS ${LUNA_TEST_TARGETS})
    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j${CTEST_NTHREADS} $$ARGS
        DEPENDS build-tests
    )
    add_custom_target(systemtests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j${CTEST_NTHREADS} -L system $$ARGS
        DEPENDS build-tests
    )
    add_custom_target(regress
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j${CTEST_NTHREADS} -L regress $$ARGS
        DEPENDS build-tests
    )
endif()

if(RUN_PYTHON_TEST)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_target(pytests
        COMMAND ${Python3_EXECUTABLE} ${TESTS_DIR}/regress/run_regression.py $$ARGS
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
endif()
