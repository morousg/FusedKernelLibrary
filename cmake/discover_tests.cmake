set (LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/tests/main.cpp;${CMAKE_SOURCE_DIR}/tests/main.h")
if (WIN32)
    list(APPEND LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/manifest.xml") #for utf8 codepage
endif() 

function(add_cuda_to_test TARGET_NAME)
    add_cuda_to_target(${TARGET_NAME} "")
    set_target_cuda_arch_flags(${TARGET_NAME})
        
    if(${ENABLE_DEBUG})
        add_cuda_debug_support_to_target(${TARGET_NAME})
    endif()
    if(${ENABLE_NVTX})
        add_nvtx_support_to_target(${TARGET_NAME})
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE CUDA::nppc CUDA::nppial CUDA::nppidei CUDA::nppig) 
endfunction()

function (add_generated_test TARGET_NAME EXTENSION)
                
        set(TEST_GENERATED_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.${EXTENSION}") #use the same name as the target	)			
        configure_file(${test_source} ${TEST_GENERATED_SOURCE} @ONLY) #replace variables in the test source file                
        add_executable(${TARGET_NAME} ${TEST_GENERATED_SOURCE} )
        target_sources(${TARGET_NAME} PRIVATE ${LAUNCH_SOURCES})            
        
        if(${ENABLE_BENCHMARK})
            target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_BENCHMARK)
        endif()
        
        cmake_path(SET path2 "${DIR}")
        cmake_path(GET path2 FILENAME DIR_NAME)       
      
        #todo: add hip support
        set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)            
        target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_SOURCE_DIR}")        
        target_link_libraries(${TARGET_NAME} PRIVATE FKL::FKL)
        if (MSVC)
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/diagnostics:caret>)
        endif()
        
        add_optimization_flags(${TARGET_NAME})
        
        add_test(NAME  ${TARGET_NAME} COMMAND ${TARGET_NAME})                 
endfunction()

function (discover_tests DIR)    
    file(
        GLOB_RECURSE
        TEST_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.in"        
    )
     
    foreach(test_source ${TEST_SOURCES})
        get_filename_component(TARGET_NAME ${test_source} NAME_WE)   
        add_generated_test("${TARGET_NAME}_cpu" "cpp")
        set_property(TARGET "${TARGET_NAME}_cpu" PROPERTY FOLDER "tests/cpu/${DIR_NAME}")
        if (CMAKE_CUDA_COMPILER)
            add_generated_test("${TARGET_NAME}_cuda" "cu")
            add_cuda_to_test("${TARGET_NAME}_cuda")
            set_property(TARGET "${TARGET_NAME}_cuda" PROPERTY FOLDER "tests/cuda/${DIR_NAME}")
        endif()
    endforeach()
endfunction()
 