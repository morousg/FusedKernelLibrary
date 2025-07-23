include (${CMAKE_SOURCE_DIR}/cmake/tests/add_generated_test.cmake)

function (discover_tests DIR)    
    file(
        GLOB_RECURSE
        TEST_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.h"        
    )
     
    foreach(test_source ${TEST_SOURCES})         
        get_filename_component(TARGET_NAME ${test_source} NAME_WE)           
        file (READ ${test_source} TEST_SOURCE_CONTENTS ) #read the contents of the test source file
       
        string(FIND "${TEST_SOURCE_CONTENTS}" "ONLY_CU"  POS_ONLY_CU)
        string(FIND "${TEST_SOURCE_CONTENTS}" "ONLY_CPU"  POS_ONLY_CPU)
        string(FIND "${TEST_SOURCE_CONTENTS}" "NVRTC"  POS_NVRTC)
        cmake_path(GET test_source RELATIVE_PART DIR_RELATIVE_PATH)     
        string(REPLACE "${PROJECT_NAME}/" " " DIR_RELATIVE_PATH "${DIR_RELATIVE_PATH}") #remove the project name from the relative path
        
        if (${POS_ONLY_CU} EQUAL -1) #if the source file does not contain "__ONLY_CU__"    
            if (${ENABLE_CPU})
                # Check if test needs NVRTC and if NVRTC is enabled
                if (POS_NVRTC GREATER_EQUAL 0 AND NOT NVRTC_ENABLE)
                    message(STATUS "Skipping NVRTC test ${TARGET_NAME} because NVRTC is disabled")
                else()
                    add_generated_test("${TARGET_NAME}" "${test_source}" "cpp" "${DIR_RELATIVE_PATH}" ${POS_NVRTC})
                      target_include_directories("${TARGET_NAME}_cpp" PRIVATE ${CLANG_INCLUDE_DIRS})
                    #  llvm_map_components_to_libnames(llvm_libs support core interpreter x86codegen  analysis)
                     #   target_link_libraries("${TARGET_NAME}_cpp" PRIVATE ${llvm_libs})
                     set(CLANG_LIBRARIES Interpreter Frontend CodeGen Sema Analysis AST Parse Lex Basic)
                     list(TRANSFORM CLANG_LIBRARIES PREPEND  clang OUTPUT_VARIABLE  CLANG_LIBRARIES_WITH_PREFIX)
                    
                    
                     target_link_libraries("${TARGET_NAME}_cpp" PRIVATE ${CLANG_LIBRARIES_WITH_PREFIX})


                endif()
            endif()
        endif()

        if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
            if (${POS_ONLY_CPU} EQUAL -1) #if the source file does not contain "__ONLY_CPU__"
                add_generated_test("${TARGET_NAME}"  "${test_source}" "cu"  "${DIR_RELATIVE_PATH}" ${POS_NVRTC})
                add_cuda_to_test("${TARGET_NAME}_cu")    
                
                 if (POS_NVRTC GREATER_EQUAL 0 AND NOT NVRTC_ENABLED)
                    message(STATUS "Skipping NVRTC test ${TARGET_NAME} because NVRTC is disabled")
                else()                   
                    target_link_libraries("${TARGET_NAME}_cu" PUBLIC clangInterpreter)
                endif()

            endif()
        endif()
         
      
    endforeach()

    get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
	list(REMOVE_ITEM importTargetsAfter ${importTargets})
endfunction()
 