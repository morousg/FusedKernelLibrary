set (LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/tests/main.cpp;${CMAKE_SOURCE_DIR}/tests/main.h")
if (WIN32)
	list(APPEND LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/manifest.xml") #for utf8 codepage
endif() 

function (discover_tests DIR)    
    file(
        GLOB_RECURSE
        TEST_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.cpp"
        "${DIR}/*.cu"
        "${DIR}/*.cuh"
        "${DIR}/*.h"
        "${DIR}/*.hpp"
    )
     
    foreach(test_source ${TEST_SOURCES})
		get_filename_component(TARGET_NAME ${test_source} NAME_WE)           
		add_executable(${TARGET_NAME} ${test_source} )
		target_sources(${TARGET_NAME} PRIVATE ${LAUNCH_SOURCES})            
		
		if(${ENABLE_BENCHMARK})
			target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_BENCHMARK)
		endif()
		
		cmake_path(SET path2 "${DIR}")
		cmake_path(GET path2 FILENAME DIR_NAME)       
		set_property(TARGET ${TARGET_NAME} PROPERTY FOLDER tests/${DIR_NAME})
		if (${CMAKE_CUDA_COMPILER})
			add_cuda_to_target(${TARGET_NAME} "")
			set_target_cuda_arch_flags(${TARGET_NAME})
			
			if(${ENABLE_DEBUG})
				add_cuda_debug_support_to_target(${TARGET_NAME})
			endif()

			if(${ENABLE_NVTX})
				add_nvtx_support_to_target(${TARGET_NAME})
			endif()
		target_link_libraries(${TARGET_NAME} PRIVATE CUDA::nppc CUDA::nppial CUDA::nppidei CUDA::nppig) 
		endif()
		#todo: add hip support
		set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)            
		target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_SOURCE_DIR}")        
		target_link_libraries(${TARGET_NAME} PRIVATE FKL::FKL)
        if (MSVC)
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/diagnostics:caret>)
        endif()
		
		add_optimization_flags(${TARGET_NAME})
		add_test(NAME  ${TARGET_NAME} COMMAND ${TARGET_NAME})
		
    endforeach()
endfunction()

function (discover_cpu_tests DIR)
	file(
        GLOB_RECURSE
        CPU_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.cpp"
    )
	foreach(cpu_source ${CPU_SOURCES})
		get_filename_component(cpu_target ${cpu_source} NAME_WE)           
		add_executable(${cpu_target} ${cpu_source} )          
		
		cmake_path(SET path2 "${DIR}")
		cmake_path(GET path2 FILENAME DIR_NAME)       
		set_property(TARGET ${cpu_target} PROPERTY FOLDER cpu_tests/${cpu_target})

		set_target_properties(${cpu_target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)            
		target_include_directories(${cpu_target} PRIVATE "${CMAKE_SOURCE_DIR}")        
		target_link_libraries(${cpu_target} PRIVATE FKL::FKL)
		
		add_test(NAME  ${cpu_target} COMMAND ${cpu_target})
    endforeach()
endfunction()