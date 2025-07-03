set (LIB_NAME FKL)

# Check for LLVM for JIT compilation support
find_package(LLVM QUIET)
if(LLVM_FOUND)
    message(STATUS "LLVM found: ${LLVM_PACKAGE_VERSION}")
    add_library(${LIB_NAME} STATIC cpu_jit_details.cpp)
    
    # Add LLVM include directories and libraries
    target_include_directories(${LIB_NAME} PRIVATE ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(${LIB_NAME} PRIVATE ${LLVM_DEFINITIONS})
    
    # Link LLVM libraries for ORCv2 JIT
    llvm_map_components_to_libnames(llvm_libs core orcjit native)
    target_link_libraries(${LIB_NAME} PRIVATE ${llvm_libs})
    
    target_compile_definitions(${LIB_NAME} PUBLIC FK_LLVM_AVAILABLE)
else()
    message(STATUS "LLVM not found, JIT compilation will use placeholder implementation")
    add_library(${LIB_NAME} STATIC cpu_jit_details.cpp)
endif()

# Add the include directory for the library
target_include_directories(${LIB_NAME} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# add alias so the project can be uses with add_subdirectory
add_library(${LIB_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})

 target_include_directories(
    ${LIB_NAME}
    INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
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