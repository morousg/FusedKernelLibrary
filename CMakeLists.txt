
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
option(ENABLE_CPU_JIT "Enable CPU JIT runtime compilation support" ON)

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

# LLVM/Clang for CPU JIT compilation
if (${ENABLE_CPU_JIT})
    find_package(LLVM REQUIRED CONFIG)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    
    # Find the libraries that correspond to the LLVM components we need
    llvm_map_components_to_libnames(llvm_libs support core irreader orcjit native)
    
    add_definitions(${LLVM_DEFINITIONS})
    add_definitions(-DENABLE_CPU_JIT)
endif()

option (BUILD_UTEST "build standard unit tests" ON)
option (ENABLE_BENCHMARK "build benchmarking unit tests" OFF)

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
