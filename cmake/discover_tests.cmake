set (LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/tests/main.cpp;${CMAKE_SOURCE_DIR}/tests/main.h")
if (WIN32)
    list(APPEND LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/utf8cp.manifest") #for utf8 codepage
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

function (add_generated_test TARGET_NAME TEST_SOURCE EXTENSION DIR)
                       
        set(TEST_GENERATED_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_${EXTENSION}/launcher.${EXTENSION}") #use the same name as the target	)			
       
        configure_file(${CMAKE_SOURCE_DIR}/tests/launcher.in ${TEST_GENERATED_SOURCE} @ONLY) #replace variables in the test source file                 
        set (TARGET_NAME_EXT "${TARGET_NAME}_${EXTENSION}")
        #message(STATUS "Adding test: ${TARGET_NAME_EXT} from ${TEST_GENERATED_SOURCE} and ${TEST_SOURCE}")
        add_executable(${TARGET_NAME_EXT} "${TEST_GENERATED_SOURCE};${TEST_SOURCE}" )
        target_sources(${TARGET_NAME_EXT} PRIVATE ${LAUNCH_SOURCES})            
        
        if(${ENABLE_BENCHMARK})
            target_compile_definitions(${TARGET_NAME_EXT} PRIVATE ENABLE_BENCHMARK)
        endif()
         
        set_target_properties(${TARGET_NAME_EXT} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)            
        target_include_directories(${TARGET_NAME_EXT} PUBLIC "${CMAKE_SOURCE_DIR}")        
        target_include_directories(${TARGET_NAME_EXT} PUBLIC "${DIR}")      
        target_link_libraries(${TARGET_NAME_EXT} PUBLIC FKL::FKL)
        if (MSVC)
            target_compile_options(${TARGET_NAME_EXT} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/diagnostics:caret>)
        #    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
             #target_link_libraries(${TARGET_NAME_EXT} -manifest:embed -manifestinput:"${PROJECT_SOURCE_DIR}/myapp.manifest" 
        endif()
        
        add_optimization_flags(${TARGET_NAME_EXT})
        
        add_test(NAME  ${TARGET_NAME_EXT} COMMAND ${TARGET_NAME_EXT})    
        cmake_path(SET path2 "${DIR}")
		cmake_path(GET path2 FILENAME DIR_NAME)       
		set_property(TARGET ${cuda_target} PROPERTY FOLDER tests/${DIR_NAME})
        set_property(TARGET "${TARGET_NAME_EXT}" PROPERTY FOLDER "tests/${EXTENSION}/${DIR_NAME}")    
        
endfunction()

function (discover_tests DIR)    
    file(
        GLOB_RECURSE
        TEST_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.h"        
    )
     
    foreach(test_source ${TEST_SOURCES})
         
        get_filename_component(TARGET_NAME ${test_source} NAME_WE)   
        cmake_path(GET test_source  PARENT_PATH  DIR_NAME) #get the directory name of the test source file
        file (READ ${test_source} TEST_SOURCE_CONTENTS ) #read the contents of the test source file
       
        string(FIND "${TEST_SOURCE_CONTENTS}" "ONLYCU"  POS)
        string(FIND "${TEST_SOURCE_CONTENTS}" "gtest/gtest.h"  GTEST_HEADER_POS)
        
        if (${POS} EQUAL -1) #if the source file does not contain "__ONLY_CU__"    
            if (${ENABLE_CPU})                    
                add_generated_test("${TARGET_NAME}" "${test_source}" "cpp" "${DIR_NAME}")
                if (${GTEST_HEADER_POS} GREATER -1 AND ${GTest_FOUND}) #if the source file does not contain "__ONLY_CU__"    
                    message(STATUS "Adding googletest to target ${TARGET_NAME}_cpp")                    
                    add_googletest_to_target("${TARGET_NAME}_cpp")                 
                endif()
            endif()
        endif()
        if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
            add_generated_test("${TARGET_NAME}"  "${test_source}" "cu"  "${DIR_NAME}")
            add_cuda_to_test("${TARGET_NAME}_cu")   
          
             if (${GTEST_HEADER_POS} GREATER -1 AND ${GTest_FOUND}) #if the source file does not contain "__ONLY_CU__"
                    message(STATUS "Adding googletest to target ${TARGET_NAME}_cu")    
                    add_googletest_to_target("${TARGET_NAME}_cu")                 
            endif()
        endif()
         
      
    endforeach()
endfunction()
 