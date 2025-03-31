set (LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/tests/utils/main.cpp;${CMAKE_SOURCE_DIR}/tests/utils/main.h")
 
function (discover_tests DIR)    
    file(
        GLOB_RECURSE
        CUDA_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.cpp"
        "${DIR}/*.cu"
        "${DIR}/*.cuh"
        "${DIR}/*.h"
        "${DIR}/*.hpp"
    )
     
    foreach(cuda_source ${CUDA_SOURCES})
        string(FIND ${cuda_source} "utils" is_utils)
        if (${is_utils} EQUAL -1)

            get_filename_component(cuda_target ${cuda_source} NAME_WE)           
            add_executable(${cuda_target} ${cuda_source} )
            target_sources(${cuda_target} PRIVATE ${LAUNCH_SOURCES})            
            
            if(${ENABLE_BENCHMARK})
                target_compile_definitions(${cuda_target} PRIVATE ENABLE_BENCHMARK)
            endif()
            
            cmake_path(SET path2 "${DIR}")
            cmake_path(GET path2 FILENAME DIR_NAME)       
            set_property(TARGET ${cuda_target} PROPERTY FOLDER tests/${DIR_NAME})
            add_cuda_to_target(${cuda_target} "")
                
            if(${ENABLE_DEBUG})
                add_cuda_debug_support_to_target(${cuda_target})
            endif()

            if(${ENABLE_NVTX})
                add_nvtx_support_to_target(${cuda_target})
            endif()

            set_target_properties(${cuda_target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)            
            target_include_directories(${cuda_target} PRIVATE "${CMAKE_SOURCE_DIR}")        
            target_link_libraries(${cuda_target} PRIVATE FKL::FKL)
            
            set_target_cuda_arch_flags(${cuda_target})
            add_test(NAME  ${cuda_target} COMMAND ${cuda_target})
            target_link_libraries(${cuda_target} PRIVATE CUDA::nppc CUDA::nppial CUDA::nppidei CUDA::nppig)
        endif()    
    endforeach()
endfunction()