

if (MSVC)
        #intellisense hack for vs2022
    set (LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/include/intellisense/main.cpp;${CMAKE_SOURCE_DIR}/include/intellisense/main.h")
endif()   

function (discover_tests DIR)    
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

        # Hack to get intellisense working for CUDA includes
        target_link_libraries(${cuda_target} PRIVATE ${PROJECT_NAME})        
        set_target_cuda_arch_flags(${cuda_target})
        add_test(NAME  ${cuda_target} COMMAND ${cuda_target})
    endforeach()
endfunction()