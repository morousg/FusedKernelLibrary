function(add_optimization_flags TARGET_NAME)    
    # Add architecture-specific optimization flags for the target
    if (MSVC)        
    
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64")
            SET(ARCH_FLAGS "AVX2" CACHE STRING "instrucion set to use")
            SET_PROPERTY(CACHE ARCH_FLAGS PROPERTY STRINGS AVX2 AVX512 AVX10.1) 
            option(ARCH_FLAGS "CPU arch" "AVX2")
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:${ARCH_FLAGS}>)        
        elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "ARM64")    
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:armv8.2>)         
        endif()
    endif()
    if (UNIX)             
        #default> build for native host architecture  for maximum performance (might not work on all systems)   
        #works both for x86_64 and aarch64      
        #we currently don't test avx10.1 with gcc11 on ubuntu 22.04
        SET_PROPERTY(CACHE ARCH_FLAGS PROPERTY STRINGS native haswell skylake-avx512) 
        option(ARCH_FLAGS "CPU arch" "native")          
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=${ARCH_FLAGS}>)
    endif()
endfunction()