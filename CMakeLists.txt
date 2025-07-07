
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
option(LLVM_JIT_ENABLE "Enable LLVM ORC JIT for CPU runtime compilation" OFF)

if (NVRTC_ENABLE)
	if (CUDAToolkit_FOUND)
		message(STATUS "NVRTC support is enabled.")
		option(NVRTC_STATIC_LINK "Enable static linking for NVRTC" ON)
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
	else()
		set(NVRTC_ENABLE OFF CACHE BOOL "NVRTC support was disabled because the CUDA Toolkit could not be found." FORCE)
	endif()
endif()

# LLVM ORC JIT support configuration
if (LLVM_JIT_ENABLE)
	find_package(LLVM REQUIRED CONFIG)
	if (LLVM_FOUND)
		message(STATUS "LLVM support is enabled.")
		message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
		message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
		
		# Find the libraries that correspond to the LLVM components
		# that we wish to use
		llvm_map_components_to_libnames(llvm_libs
			Core
			OrcJIT
			native
			RISCV
			Support
			ExecutionEngine
			MCJIT
			Target
			RuntimeDyld
			Object
			InstCombine
			AggressiveInstCombine
			TransformUtils
			Analysis
			ipo
			MC
			Coverage
			ProfileData
		)
		
		# Add clang libraries for C++ compilation (using shared libraries for now)
		set(clang_libs
			/usr/lib/x86_64-linux-gnu/libclang-cpp.so.18
		)
		
		message(STATUS "LLVM libraries: ${llvm_libs}")
		message(STATUS "Clang libraries: ${clang_libs}")
		set(LLVM_JIT_LIBRARIES ${llvm_libs} ${clang_libs})
		
		# Add preprocessor definitions for LLVM JIT
		add_definitions(-DLLVM_JIT_ENABLE)
		
		# Set up include directories
		include_directories(${LLVM_INCLUDE_DIRS})
		include_directories(/usr/lib/llvm-18/include)
		separate_arguments(LLVM_DEFINITIONS_LIST UNIX_COMMAND "${LLVM_DEFINITIONS}")
		add_definitions(${LLVM_DEFINITIONS_LIST})
	else()
		message(WARNING "LLVM not found, disabling LLVM JIT support")
		set(LLVM_JIT_ENABLE OFF CACHE BOOL "LLVM JIT support was disabled because LLVM could not be found." FORCE)
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
