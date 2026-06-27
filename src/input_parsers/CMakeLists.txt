target_sources(${LUNA_LIB} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/OutputConstraint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VnnLibInputParser.cpp
)

target_include_directories(${LUNA_LIB} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# OnnxToTorch lives in LUNAOnnxParser so the generated ONNX protobuf
# symbols never enter LUNACore. Embedders that don't want the onnx
# protobuf in their binary link only luna::core.
target_sources(${LUNA_ONNX_PARSER_LIB} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/OnnxToTorch.cpp
)
target_include_directories(${LUNA_ONNX_PARSER_LIB} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
