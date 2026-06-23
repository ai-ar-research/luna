target_sources(${LUNA_LIB} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/OnnxToTorch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OutputConstraint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VnnLibInputParser.cpp
)

target_include_directories(${LUNA_LIB} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
