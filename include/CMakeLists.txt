file(
    GLOB_RECURSE
    FKL_SOURCES
    CONFIGURE_DEPENDS    
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)

add_library(headers INTERFACE ${FKL_SOURCES} )
# Set virtual folders for MSVC
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${FKL_SOURCES})
#target_include_directories(include INTERFACE .)

#set_target_properties(${cuda_target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)



