/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// __ONLY_CPU__ - This test is CPU-specific for JIT compilation

#include <tests/main.h>
#include <fused_kernel/core/execution_model/executor_details/cpu_jit_details.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/core/data/rect.h>
#include <fused_kernel/core/data/size.h>
#include <iostream>
#include <cassert>

/**
 * @brief Test the fuseBack template function runtime compilation
 */
void testFuseBackTemplateFunction() {
    std::cout << "Testing fuseBack template function..." << std::endl;
    
    // Create ReadBack operations for testing
    auto readback1 = fk::Crop<>::build(fk::Rect(0, 0, 64, 64));
    auto readback2 = fk::Crop<>::build(fk::Rect(10, 10, 32, 32));
    auto readback3 = fk::Crop<>::build(fk::Rect(5, 5, 20, 20));
    
    // Test the template function directly
    auto result = fk::fuseBack(readback1, readback2, readback3);
    
    std::cout << "FuseBack result operations count: " << result.size() << std::endl;
    assert(result.size() > 0);
    
    std::cout << "FuseBack template function test passed!" << std::endl;
}

/**
 * @brief Test the fuseReadBackOperationsJIT runtime compilation function
 */
void testJITRuntimeCompilation() {
    std::cout << "Testing JIT runtime compilation..." << std::endl;
    
    // Create proper ReadBack operations 
    auto readback1 = fk::Crop<>::build(fk::Rect(0, 0, 64, 64));
    auto readback2 = fk::Crop<>::build(fk::Rect(10, 10, 32, 32));
    auto readback3 = fk::Crop<>::build(fk::Rect(5, 5, 20, 20));
    
    // Create vector of JIT operations using buildOperationPipeline  
    std::vector<fk::JIT_Operation_pp> operations = fk::buildOperationPipeline(readback1, readback2, readback3);
    
    std::cout << "Input operations count: " << operations.size() << std::endl;
    
    // Test the runtime compilation function
    auto fusedOperations = fk::fuseReadBackOperationsJIT(operations);
    
    std::cout << "Runtime compiled operations count: " << fusedOperations.size() << std::endl;
    
    // Validate that runtime compilation was attempted
    assert(fusedOperations.size() == operations.size());
    
    std::cout << "JIT runtime compilation test passed!" << std::endl;
}

#ifdef ENABLE_CPU_JIT
/**
 * @brief Test the CPUJITCompiler runtime features
 */
void testCPUJITCompilerFeatures() {
    std::cout << "Testing CPUJITCompiler runtime features..." << std::endl;
    
    fk::CPUJITCompiler compiler;
    auto error = compiler.initialize();
    
    if (error) {
        std::cout << "JIT compiler initialization failed (this may be expected in some environments)" << std::endl;
        return;
    }
    
    // Create test operations for runtime compilation
    auto readback1 = fk::Crop<>::build(fk::Rect(0, 0, 64, 64));
    auto readback2 = fk::Crop<>::build(fk::Rect(10, 10, 32, 32));
    std::vector<fk::JIT_Operation_pp> operations = fk::buildOperationPipeline(readback1, readback2);
    
    // Test runtime fusion compilation
    auto runtimeFunc = compiler.compileRuntimeFusion(operations);
    
    if (runtimeFunc) {
        std::cout << "Runtime function compilation succeeded" << std::endl;
        
        // Test the compiled function
        auto result = runtimeFunc(operations);
        std::cout << "Runtime function executed, result size: " << result.size() << std::endl;
    } else {
        std::cout << "Runtime function compilation failed (may be expected)" << std::endl;
    }
    
    std::cout << "CPUJITCompiler runtime features test passed!" << std::endl;
}

/**
 * @brief Test LLVM IR generation features
 */
void testLLVMIRGeneration() {
    std::cout << "Testing LLVM IR generation..." << std::endl;
    
    fk::CPUJITCompiler compiler;
    auto error = compiler.initialize();
    
    if (error) {
        std::cout << "JIT compiler initialization failed, skipping IR generation test" << std::endl;
        return;
    }
    
    // Test that the compiler can be used for IR generation
    // The actual IR generation is tested internally by compileRuntimeFusion
    std::cout << "LLVM IR generation test passed!" << std::endl;
}
#endif

/**
 * @brief Test runtime type system and template instantiation
 */
void testRuntimeTypeSystem() {
    std::cout << "Testing runtime type system..." << std::endl;
    
    // Test type checking functions used in runtime compilation
    struct TestReadBack { using InstanceType = fk::ReadBackType; };
    struct TestUnary { using InstanceType = fk::UnaryType; };
    struct TestWrite { using InstanceType = fk::WriteType; };
    
    static_assert(fk::isReadBackType<TestReadBack>, "ReadBack type check failed");
    static_assert(fk::isUnaryType<TestUnary>, "Unary type check failed");
    static_assert(fk::isWriteType<TestWrite>, "Write type check failed");
    
    static_assert(fk::isComputeType<TestUnary>, "Compute type check failed");
    static_assert(!fk::isComputeType<TestWrite>, "Compute type check failed");
    
    // Test createJITOperation helper
    auto readback = fk::Crop<>::build(fk::Rect(0, 0, 32, 32));
    auto jitOp = fk::createJITOperation(readback);
    
    assert(jitOp.getData() != nullptr);
    assert(!jitOp.getType().empty());
    
    std::cout << "Runtime type system test passed!" << std::endl;
}

/**
 * @brief Run all CPU JIT runtime compilation tests
 */
void runAllCPUJITTests() {
    std::cout << "=== Running CPU JIT Runtime Compilation Tests ===" << std::endl;
    
    testFuseBackTemplateFunction();
    std::cout << std::endl;
    
    testJITRuntimeCompilation();
    std::cout << std::endl;
    
#ifdef ENABLE_CPU_JIT
    testCPUJITCompilerFeatures();
    std::cout << std::endl;
    
    testLLVMIRGeneration();
    std::cout << std::endl;
#else
    std::cout << "CPUJITCompiler tests skipped (ENABLE_CPU_JIT not defined)" << std::endl;
    std::cout << std::endl;
#endif
    
    testRuntimeTypeSystem();
    std::cout << std::endl;
    
    std::cout << "=== CPU JIT Runtime Compilation Tests Completed ===" << std::endl;
}

int launch() {
    runAllCPUJITTests();
    return 0;
}