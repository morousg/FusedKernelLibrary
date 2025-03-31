
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
        add_test(NAME  ${cuda_target} COMMAND ${cuda_target})
        target_link_libraries(${cuda_target} PRIVATE CUDA::nppc CUDA::nppial CUDA::nppidei CUDA::nppig headers FKL::FKL)        
        set_property(TARGET ${cuda_target} PROPERTY FOLDER benchmarks/${DIR_NAME})
        set_target_properties(${cuda_target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)  
        target_include_directories(${cuda_target} PRIVATE "${CMAKE_SOURCE_DIR}")
        set_target_cuda_arch_flags(${cuda_target})        
        add_cuda_to_target(${cuda_target} "")
	
        if(${ENABLE_DEBUG})
            add_cuda_debug_support_to_target(${cuda_target})
        endif()

        if(${ENABLE_NVTX})
            add_nvtx_support_to_target(${cuda_target})
        endif()

        if(${ENABLE_BENCHMARK})
            target_compile_definitions(${cuda_target} PRIVATE ENABLE_BENCHMARK)
        endif()

        
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