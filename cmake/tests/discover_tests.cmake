include (${CMAKE_SOURCE_DIR}/cmake/tests/add_generated_test.cmake)

function (discover_tests DIR)    
    file(
        GLOB_RECURSE
        TEST_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*"        
    )
     
    foreach(TEST_SOURCE ${TEST_SOURCES})         
        get_filename_component(TARGET_NAME ${TEST_SOURCE} NAME_WE)           
        file (READ ${TEST_SOURCE} TEST_SOURCE_CONTENTS ) #read the contents of the test source file
       
        string(FIND "${TEST_SOURCE_CONTENTS}" "ONLY_CU"  POS_ONLY_CU)
        string(FIND "${TEST_SOURCE_CONTENTS}" "ONLY_CPU"  POS_ONLY_CPU)        
        cmake_path(GET TEST_SOURCE RELATIVE_PART DIR_RELATIVE_PATH)     
        string(REPLACE "${PROJECT_NAME}/" " " DIR_RELATIVE_PATH "${DIR_RELATIVE_PATH}") #remove the project name from the relative path
         
        message(STATUS "Adding test:  ${test_source}")

        string(FIND "${TEST_SOURCE}" ".in"  IS_CMAKE_GENERATED_SOURCE)
        if (${IS_CMAKE_GENERATED_SOURCE} GREATER -1)            
            set (FUNDAMENTAL_TYPES_COUNT 12) 
            foreach(FUNDALMENTAL_TYPE_OFFSET RANGE 0 ${FUNDAMENTAL_TYPES_COUNT})                
                set(TEST_GENERATED_SOURCE_N "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}/${TARGET_NAME}${FUNDALMENTAL_TYPE_OFFSET}.h") #use the same name as the target	)
                message(STATUS "The test source file ${TEST_SOURCE} --->${TEST_GENERATED_SOURCE_N} ")
                configure_file(${TEST_SOURCE} ${TEST_GENERATED_SOURCE_N} @ONLY) #replace variables in the test source file                                                                         
           
               add_generated_test("${TARGET_NAME}${FUNDALMENTAL_TYPE_OFFSET}" "${TEST_GENERATED_SOURCE_N}" "cpp" "${DIR_RELATIVE_PATH}")   
               if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
               
                   add_generated_test("${TARGET_NAME}${FUNDALMENTAL_TYPE_OFFSET}"  "${TEST_GENERATED_SOURCE_N}" "cu"  "${DIR_RELATIVE_PATH}")
                   add_cuda_to_test("${TARGET_NAME}${FUNDALMENTAL_TYPE_OFFSET}_cu")                       
               endif()
            endforeach()
        else()
            if (${POS_ONLY_CU} EQUAL -1) #if the source file does not contain "__ONLY_CU__"    
                if (${ENABLE_CPU})                    
                    add_generated_test("${TARGET_NAME}" "${TEST_SOURCE}" "cpp" "${DIR_RELATIVE_PATH}")                
                endif()
            endif()

            if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
                if (${POS_ONLY_CPU} EQUAL -1) #if the source file does not contain "__ONLY_CPU__"
                    add_generated_test("${TARGET_NAME}"  "${TEST_SOURCE}" "cu"  "${DIR_RELATIVE_PATH}")
                    add_cuda_to_test("${TARGET_NAME}_cu")                           
                endif()
            endif()
        
        endif()
        
         
      
    endforeach()
endfunction()
 