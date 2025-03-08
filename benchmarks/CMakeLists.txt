
function (discover_benchmark DIR)
    file(
        GLOB_RECURSE
        CUDA_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.cpp"
        "${DIR}/*.cu"
    )
    
    foreach(cuda_source ${CUDA_SOURCES})
    get_filename_component(cuda_target ${cuda_source} NAME_WE)    
    add_executable(${cuda_target} ${cuda_source} ${LAUNCH_SOURCES}) 
    target_link_libraries(${cuda_target} PRIVATE CUDA::nppc CUDA::nppial CUDA::nppidei CUDA::nppig headers ${PROJECT_NAME}::${PROJECT_NAME})
        
    endforeach()
endfunction()

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})

MACRO(SUBDIRLIST result curdir)
FILE(GLOB children ${curdir}/*) # 
SET(dirlist "")
FOREACH(child ${children})
    IF(IS_DIRECTORY ${child}) 
        LIST(APPEND dirlist ${child})
    ENDIF()
ENDFOREACH()
SET(${result} ${dirlist} )
ENDMACRO()

set (LIST_OF_DIRS "")
SUBDIRLIST(LIST_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
 
foreach(DIR ${LIST_DIRS})
    discover_benchmark(${DIR})    
endforeach()