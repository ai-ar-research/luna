target_sources(${LUNA_LIB} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ConstSimpleData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Error.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/File.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HeapData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/IFile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MString.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TimeUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/real/Errno.cpp
)

target_include_directories(${LUNA_LIB} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
