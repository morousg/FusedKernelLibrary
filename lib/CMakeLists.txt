set (LIB_NAME FKL)

# Create the main interface library
add_library(${LIB_NAME} INTERFACE)
# add alias so the project can be uses with add_subdirectory
add_library(${LIB_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})

target_include_directories(
    ${LIB_NAME}
    INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Always build CPU JIT library with basic functionality
add_library(FKL_CPU_JIT STATIC cpu_jit_details.cpp)

target_include_directories(FKL_CPU_JIT PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)

# Add CPU JIT library features if enabled
if (${ENABLE_CPU_JIT})
    target_include_directories(FKL_CPU_JIT PUBLIC ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(FKL_CPU_JIT PUBLIC ${LLVM_DEFINITIONS})
    target_link_libraries(FKL_CPU_JIT ${llvm_libs})
endif()

# Link the JIT library to the main interface library
target_link_libraries(${LIB_NAME} INTERFACE FKL_CPU_JIT)
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