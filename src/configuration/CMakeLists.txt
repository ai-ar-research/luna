# src/configuration/CMakeLists.txt
# Configuration module sources

# Add configuration source files
target_sources(${LUNA_LIB} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/LunaConfiguration.cpp
)

# Set include directories for configuration module
target_include_directories(${LUNA_LIB} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
