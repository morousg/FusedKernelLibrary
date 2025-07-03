# LLVM JIT Integration for FusedKernelLibrary
# This file sets up LLVM ORCv2 for runtime compilation

option(ENABLE_LLVM_JIT "Enable LLVM JIT compilation for CPU operations" ON)

if(ENABLE_LLVM_JIT)
    # Find required dependencies first
    find_package(ZLIB REQUIRED)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBFFI REQUIRED libffi)
    
    # Find LLVM installation
    find_package(LLVM REQUIRED CONFIG)
    
    if(LLVM_FOUND)
        message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
        message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
        
        # Set LLVM configuration
        include_directories(${LLVM_INCLUDE_DIRS})
        separate_arguments(LLVM_DEFINITIONS_LIST UNIX_COMMAND "${LLVM_DEFINITIONS}")
        add_definitions(${LLVM_DEFINITIONS_LIST})
        
        # Find the required LLVM components
        llvm_map_components_to_libnames(LLVM_LIBS 
            support 
            core 
            irreader 
            executionengine 
            orcjit 
            runtimedyld 
            native
            x86asmparser
            x86codegen
            x86desc
            x86disassembler
            x86info
        )
        
        # Enable LLVM JIT compilation flag
        add_definitions(-DENABLE_LLVM_JIT)
        
        message(STATUS "LLVM JIT support enabled")
    else()
        message(WARNING "LLVM not found. JIT compilation will be disabled.")
        set(ENABLE_LLVM_JIT OFF)
    endif()
else()
    message(STATUS "LLVM JIT support disabled")
endif()

# Function to add LLVM support to a target
function(add_llvm_support TARGET_NAME)
    if(ENABLE_LLVM_JIT)
        target_link_libraries(${TARGET_NAME} PRIVATE ${LLVM_LIBS})
        target_link_libraries(${TARGET_NAME} PRIVATE ZLIB::ZLIB)
        target_link_libraries(${TARGET_NAME} PRIVATE ${LIBFFI_LIBRARIES})
        target_include_directories(${TARGET_NAME} PRIVATE ${LLVM_INCLUDE_DIRS})
        target_include_directories(${TARGET_NAME} PRIVATE ${LIBFFI_INCLUDE_DIRS})
        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_LLVM_JIT)
        
        # Set required C++ standard for LLVM
        set_target_properties(${TARGET_NAME} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )
    endif()
endfunction()