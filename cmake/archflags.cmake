function(add_optimization_flags TARGET_NAME)    
    # Add architecture-specific optimization flags for the target
    if (MSVC)        
    
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64")
            SET(ARCH_FLAGS "AVX2" CACHE STRING "instrucion set to use")
            SET_PROPERTY(CACHE ARCH_FLAGS PROPERTY STRINGS AVX2 AVX512 AVX10.1) 
            option(ARCH_FLAGS "CPU arch" "AVX2")
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:${ARCH_FLAGS}>)        
        endif()
        #we don't have  windows arm64 hw to test so no flags for arm64 for now
        
    endif()
    if (UNIX)             
        #default> build for native host architecture  for maximum performance (might not work on all systems)   
        #works both for x86_64 and aarch64      
        #we currently don't test avx10.1 with gcc11 on ubuntu 22.04
        #armv9-a (GH100)  not available in gcc11
        #for arm64 we test on jetson orin and grace-hopper
        #haswell: minimum avx2 arch 
        #skylake: minium avx512 arch
        #AVX10.1 only gcc 14 (and >15 for diamondrapids)
         if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64")
            SET_PROPERTY(CACHE ARCH_FLAGS PROPERTY STRINGS native haswell skylake-avx512 diamondrapids) 
            option(ARCH_FLAGS "CPU arch" "native")          
         elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "ARM64")         
            SET_PROPERTY(CACHE ARCH_FLAGS PROPERTY STRINGS native armv8.2-a armv9-a)  
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=${ARCH_FLAGS}>)     
        endif()
    endif()
    
endfunction()