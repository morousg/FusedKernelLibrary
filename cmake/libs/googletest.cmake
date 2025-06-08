set(GTEST_VERSION_MAJOR 1)
set(GTEST_VERSION_MINOR 17)
set(GTEST_VERSION_RELEASE 0)
set(GTEST_VERSION_PATCH "release")
set(GTEST_VERSION "${GTEST_VERSION_MAJOR}.${GTEST_VERSION_MINOR}.${GTEST_VERSION_RELEASE}")

#list(APPEND CMAKE_PREFIX_PATH "${APIS_PATH}/googletest-${GTEST_VERSION}/lib/cmake")
list(APPEND CMAKE_PREFIX_PATH "D:/googletest-${GTEST_VERSION}/lib/cmake")
find_package(GTest ${GTEST_VERSION}  CONFIG)

function(add_googletest_to_target TARGET_NAME)
    set(EXPORTED_TARGETS GTest::gtest_main GTest::gmock_main)
    target_link_libraries(${TARGET_NAME} PRIVATE ${EXPORTED_TARGETS})
    deploy_exported_target_dependencies(${TARGET_NAME} ${EXPORTED_TARGETS} GTest::gtest GTest::gmock)    
    target_compile_definitions(${TARGET_NAME} PRIVATE "GTEST=1")
endfunction()

function(deploy_googletest)
    if(WIN32)
        set(GTEST_DEPLOYED_STAMP ${OUT_DIR}/gtest-deployed.stamp)
        set(EXPORTED_TARGETS GTest::gtest_main GTest::gmock_main GTest::gtest GTest::gmock)
        # Add extra target for deployment of GTest libs, depending on ALL target
        add_custom_target(DEPLOY_GTEST ALL DEPENDS ${GTEST_DEPLOYED_STAMP})
        set_property(TARGET DEPLOY_GTEST PROPERTY FOLDER "CMake")
        foreach(EXPORTED_TARGET ${EXPORTED_TARGETS})
            # Add commands to be exectued for this target, copying the files
            add_custom_command(TARGET DEPLOY_GTEST POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                                                      $<TARGET_FILE:${EXPORTED_TARGET}> ${OUT_DIR})
        endforeach()
        # Set required properties on the QT_DEPLOY_STEP_TARGET
        set_target_properties(${DEPLOY_GTEST} PROPERTIES FOLDER "CMake") # Move the target to CMake subfolder in VS

        # Finally, create a dummy file to mark the deployment done
        add_custom_command(OUTPUT ${GTEST_DEPLOYED_STAMP} POST_BUILD COMMAND ${CMAKE_COMMAND} -E touch
                                                                             ${GTEST_DEPLOYED_STAMP})
    endif()
endfunction()
