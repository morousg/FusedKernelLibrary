set (LIB_NAME FKL)

# Source files for the library
set(LIB_SOURCES "")
if(ENABLE_LLVM_JIT)
    list(APPEND LIB_SOURCES cpu_jit_details.cpp)
endif()

# Create library - INTERFACE if no sources, otherwise STATIC
if(LIB_SOURCES)
    add_library(${LIB_NAME} STATIC ${LIB_SOURCES})
    # Add include directories to the static library
    target_include_directories(
        ${LIB_NAME}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
               $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    add_llvm_support(${LIB_NAME})
else()
    add_library(${LIB_NAME} INTERFACE)
    # Add include directories to the interface library
    target_include_directories(
        ${LIB_NAME}
        INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
                  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
endif()

# add alias so the project can be uses with add_subdirectory
add_library(${LIB_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})


# locations are provided by GNUInstallDirs
install(
    TARGETS ${LIB_NAME}
    EXPORT ${LIB_NAME}_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#always update exported version file when the version changes
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/FKLConfigVersion.cmake.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/export/FKLConfigVersion.cmake"
    @ONLY)