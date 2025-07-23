
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

set (PROJECT_VERSION_MAJOR 0)
set (PROJECT_VERSION_MINOR 1)
set (PROJECT_VERSION_REV 9)
set (PROJECT_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_REV})
include (cmake/cmake_init.cmake)
include (cmake/doxygen.cmake)
include (cmake/targets/virtualfolders.cmake)
include (cmake/deploy/deploy_dependencies.cmake)

include (cmake/archflags.cmake)
include (cmake/generators/version_header.cmake)
project(FusedKernelLibrary VERSION ${PROJECT_VERSION} LANGUAGES CXX  
DESCRIPTION "Implementation of a methodology that allows all sorts of user defined GPU kernel fusion, for non CUDA programmers." 
HOMEPAGE_URL "https://github.com/morousg/FusedKernelLibrary" )

#cuda is optional, but if it is found, it will be used
option(ENABLE_CPU "Enable CPU support" ON)
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    option (ENABLE_CUDA "Enable CUDA support" ON)
    if (${ENABLE_CUDA})
        include(cmake/cuda_init.cmake)
    endif()
else()
    message(STATUS "CUDA compiler not found, CUDA support will be disabled.")     
endif()



option (BUILD_UTEST "build standard unit tests" ON)
option (ENABLE_BENCHMARK "build benchmarking unit tests" OFF)
option(NVRTC_ENABLE "Enable NVRTC for runtime compilation" OFF)

if (NVRTC_ENABLE)
    if (CUDAToolkit_FOUND)
        message(STATUS "NVRTC support is enabled.")
        option(NVRTC_STATIC_LINK "Enable static linking for NVRTC" OFF)
        if(NVRTC_STATIC_LINK)
            # --- Static Linking (Cross-Platform) ---
            # The CUDA::nvrtc_static imported target handles finding the correct static library
            # (nvrtc_static.lib on MSVC, libnvrtc_static.a on Linux) and its dependencies
            # on all platforms automatically.
            message(STATUS "Configuring for STATIC NVRTC linkage.")
            set(NVRTC_LIBRARIES CUDA::nvrtc_static)
        else()
            # --- Dynamic Linking (Cross-Platform) ---
            # The CUDA::nvrtc imported target is the standard way to link the shared/dynamic
            # NVRTC library on all platforms.
            message(STATUS "Configuring for DYNAMIC NVRTC linkage.")
            set(NVRTC_LIBRARIES CUDA::nvrtc)
        endif()
        
        # LLVM/Clang JIT compilation support for NVRTC
        message(STATUS "nvConfiguring LLVM/Clang JIT compilation support for NVRTC...")
        get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
         message(STATUS LLVM_DIR "${LLVM_DIR}")	

        find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config-18 llvm-config.exe DOC "llvm-config tool" PATHS "${LLVM_DIR}/bin/" "/usr/bin/")
        if(LLVM_CONFIG_EXECUTABLE)
			execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE}  --cmakedir OUTPUT_VARIABLE CMAKE_CONFIG_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
			message(STATUS "llvm-config  ${CMAKE_CONFIG_DIR}")            
		else()
			message(FATAL_ERROR "llvm-config executable not found. Please install LLVM/Clang or set the path to llvm-config in CMAKE_PREFIX_PATH.")
		endif()
		find_package(LLVM  REQUIRED CONFIG HINTS "${CMAKE_CONFIG_DIR}" NO_DEFAULT_PATH)        
        message(STATUS "looking for  clang config at " "${CMAKE_CONFIG_DIR}/../clang/")
        find_package(Clang REQUIRED CONFIG PATHS "${CMAKE_CONFIG_DIR}/../clang/")                 
        message(STATUS "Found llvm ${LLVM_PACKAGE_VERSION}")
        message(STATUS "Found clang ${CLANG_PACKAGE_VERSION}")
        
    else()
        set(NVRTC_ENABLE OFF CACHE BOOL "NVRTC support was disabled because the CUDA Toolkit could not be found." FORCE)
    endif()
endif()


    
add_subdirectory(include)
add_subdirectory(lib)

if (${BUILD_UTEST})    
    include (cmake/tests/discover_tests.cmake)
    enable_testing()  
    add_subdirectory(tests)
    add_subdirectory(utests)         
endif()

if (${ENABLE_BENCHMARK})
    include (cmake/tests/discover_tests.cmake)
    enable_testing()
    add_subdirectory(benchmarks)
endif()
