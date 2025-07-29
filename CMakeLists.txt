
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
		
		# LLVM/Clang JIT compilation support for NVRTC
		message(STATUS "Configuring LLVM/Clang JIT compilation support for NVRTC...")
		find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config-18 llvm-config DOC "llvm-config tool")
		
		if(LLVM_CONFIG_EXECUTABLE)
			execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --version OUTPUT_VARIABLE LLVM_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
			execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --prefix OUTPUT_VARIABLE LLVM_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
			execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --includedir OUTPUT_VARIABLE LLVM_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
			execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --libdir OUTPUT_VARIABLE LLVM_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
			execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --cxxflags OUTPUT_VARIABLE LLVM_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
			execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --ldflags OUTPUT_VARIABLE LLVM_LD_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
			execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --libs all OUTPUT_VARIABLE LLVM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
			execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --system-libs OUTPUT_VARIABLE LLVM_SYSTEM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
			
			# Find Clang C++ library for interpreter support
			find_library(CLANG_CPP_LIB 
				NAMES clang-cpp.18.1 clang-cpp.18 clang-cpp
				PATHS ${LLVM_LIB_DIR} /usr/lib/x86_64-linux-gnu /usr/lib
				NO_DEFAULT_PATH)
			if(CLANG_CPP_LIB)
				set(CLANG_LIBRARIES ${CLANG_CPP_LIB})
				message(STATUS "Found Clang C++ library: ${CLANG_CPP_LIB}")
			else()
				# Try direct path to the library
				if(EXISTS "/usr/lib/x86_64-linux-gnu/libclang-cpp.so.18.1")
					set(CLANG_LIBRARIES "/usr/lib/x86_64-linux-gnu/libclang-cpp.so.18.1")
					message(STATUS "Using direct path to Clang C++ library: ${CLANG_LIBRARIES}")
				else()
					message(WARNING "libclang-cpp not found, trying individual Clang libraries")
					set(CLANG_LIBRARIES -lclangInterpreter -lclangFrontend -lclangCodeGen -lclangSema -lclangAnalysis -lclangAST -lclangParse -lclangLex -lclangBasic)
				endif()
			endif()
			
			# Add Clang include directory for interpreter headers
			set(CLANG_INCLUDE_DIR "${LLVM_PREFIX}/include")
			if(NOT EXISTS "${CLANG_INCLUDE_DIR}/clang/Interpreter/Interpreter.h")
				# Try alternative location for Clang headers
				set(CLANG_INCLUDE_DIR "/usr/lib/llvm-18/include")
			endif()
			
			# Verify that the Clang interpreter header exists
			if(EXISTS "${CLANG_INCLUDE_DIR}/clang/Interpreter/Interpreter.h")
				message(STATUS "LLVM/Clang JIT compilation support is enabled for NVRTC.")
				message(STATUS "LLVM version: ${LLVM_VERSION}")
				message(STATUS "LLVM prefix: ${LLVM_PREFIX}")
				message(STATUS "LLVM include dir: ${LLVM_INCLUDE_DIR}")
				message(STATUS "LLVM lib dir: ${LLVM_LIB_DIR}")
				message(STATUS "LLVM LD flags: ${LLVM_LD_FLAGS}")
				message(STATUS "LLVM libs: ${LLVM_LIBS}")
				message(STATUS "LLVM system libs: ${LLVM_SYSTEM_LIBS}")
				message(STATUS "Clang libraries: ${CLANG_LIBRARIES}")
				
				set(LLVM_JIT_LIBRARIES ${LLVM_LD_FLAGS} ${LLVM_LIBS} ${CLANG_LIBRARIES} ${LLVM_SYSTEM_LIBS})
				set(LLVM_JIT_INCLUDE_DIRS ${LLVM_INCLUDE_DIR} ${CLANG_INCLUDE_DIR})
				set(LLVM_JIT_CXX_FLAGS ${LLVM_CXX_FLAGS})
				set(NVRTC_JIT_ENABLED TRUE)
			else()
				message(WARNING "Clang interpreter headers not found at ${CLANG_INCLUDE_DIR}")
				set(NVRTC_JIT_ENABLED FALSE)
			endif()
		else()
			message(STATUS "llvm-config not found, attempting to install LLVM/Clang 18...")
			set(NVRTC_JIT_ENABLED FALSE)
			
			# Attempt automatic installation on Linux
			if(UNIX AND NOT APPLE)
				set(INSTALL_SCRIPT_PATH "${CMAKE_CURRENT_BINARY_DIR}/install_llvm.sh")
				file(WRITE ${INSTALL_SCRIPT_PATH} 
					"#!/bin/bash\n"
					"set -e\n"
					"echo 'Installing LLVM/Clang 18...'\n"
					"wget -O llvm.sh https://apt.llvm.org/llvm.sh\n"
					"chmod +x llvm.sh\n"
					"sudo ./llvm.sh 18\n"
					"sudo apt update\n"
					"sudo apt install -y clang-18 libclang-18-dev llvm-18-dev lld-18 libc++-18-dev libc++abi-18-dev\n"
					"echo 'LLVM/Clang 18 installation completed.'\n"
				)
				
				execute_process(
					COMMAND chmod +x ${INSTALL_SCRIPT_PATH}
					RESULT_VARIABLE CHMOD_RESULT
				)
				
				message(STATUS "Attempting to install LLVM/Clang 18 automatically...")
				message(STATUS "Running installation script: ${INSTALL_SCRIPT_PATH}")
				
				execute_process(
					COMMAND ${INSTALL_SCRIPT_PATH}
					RESULT_VARIABLE INSTALL_RESULT
					OUTPUT_VARIABLE INSTALL_OUTPUT
					ERROR_VARIABLE INSTALL_ERROR
					TIMEOUT 300
				)
				
				if(INSTALL_RESULT EQUAL 0)
					message(STATUS "LLVM/Clang 18 installation successful. Re-running configuration...")
					# Re-try finding llvm-config after installation
					find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config-18 llvm-config DOC "llvm-config tool")
					if(LLVM_CONFIG_EXECUTABLE)
						message(STATUS "Found llvm-config after installation: ${LLVM_CONFIG_EXECUTABLE}")
						# Set NVRTC_JIT_ENABLED to TRUE so configuration continues
						set(NVRTC_JIT_ENABLED TRUE)
					endif()
				else()
					message(WARNING "LLVM/Clang 18 installation failed. JIT compilation will be disabled.")
					message(STATUS "Install output: ${INSTALL_OUTPUT}")
					message(STATUS "Install error: ${INSTALL_ERROR}")
				endif()
			elseif(WIN32)
				set(INSTALL_SCRIPT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_llvm_windows.ps1")
				
				message(STATUS "Attempting to install LLVM/Clang 18 automatically on Windows...")
				message(STATUS "Running installation script: ${INSTALL_SCRIPT_PATH}")
				
				execute_process(
					COMMAND powershell.exe -ExecutionPolicy Bypass -File ${INSTALL_SCRIPT_PATH}
					RESULT_VARIABLE INSTALL_RESULT
					OUTPUT_VARIABLE INSTALL_OUTPUT
					ERROR_VARIABLE INSTALL_ERROR
					TIMEOUT 600
				)
				
				if(INSTALL_RESULT EQUAL 0)
					message(STATUS "LLVM/Clang 18 installation successful. Re-running configuration...")
					# Re-try finding llvm-config after installation
					find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config.exe llvm-config DOC "llvm-config tool")
					if(LLVM_CONFIG_EXECUTABLE)
						message(STATUS "Found llvm-config after installation: ${LLVM_CONFIG_EXECUTABLE}")
						# Set NVRTC_JIT_ENABLED to TRUE so configuration continues
						set(NVRTC_JIT_ENABLED TRUE)
					endif()
				else()
					message(WARNING "LLVM/Clang 18 installation failed. JIT compilation will be disabled.")
					message(STATUS "Install output: ${INSTALL_OUTPUT}")
					message(STATUS "Install error: ${INSTALL_ERROR}")
					message(STATUS "Please install LLVM/Clang 18 manually:")
					message(STATUS "1. Download LLVM 18 from https://github.com/llvm/llvm-project/releases")
					message(STATUS "2. Install and add LLVM/bin to your PATH")
					message(STATUS "3. Re-run CMake configuration")
				endif()
			else()
				message(STATUS "Unsupported platform for automatic LLVM/Clang installation.")
				message(STATUS "Please install LLVM/Clang 18 manually and re-run CMake configuration.")
			endif()
		endif()
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
