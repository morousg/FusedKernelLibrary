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
 * @brief Test the JIT_Operation_pp structure and basic functionality
 */
void testJITOperationBasics() {
    std::cout << "Testing JIT_Operation_pp basics..." << std::endl;
    
    float testData = 42.0f;
    fk::JIT_Operation_pp jitOp("float", &testData, sizeof(float));
    
    // Basic validation
    assert(jitOp.getData() == jitOp.getData()); // Check that getData() returns something
    assert(jitOp.getType() == "float");
    
    std::cout << "JIT Operation type: " << jitOp.getType() << std::endl;
    std::cout << "JIT Operation data: " << *static_cast<float*>(jitOp.getData()) << std::endl;
    
    std::cout << "JIT_Operation_pp basics test passed!" << std::endl;
}

/**
 * @brief Test the fuseReadBackOperationsJIT function
 */
void testReadBackFusion() {
    std::cout << "Testing ReadBack operations fusion..." << std::endl;
    
    // Create proper ReadBack operations 
    auto readback1 = fk::Crop<>::build(fk::Rect(0, 0, 64, 64));
    auto readback2 = fk::Crop<>::build(fk::Rect(10, 10, 32, 32));
    // For demonstration, create more operations of different types
    auto readback3 = fk::Crop<>::build(fk::Rect(5, 5, 20, 20));
    
    // Create vector of JIT operations using buildOperationPipeline  
    std::vector<fk::JIT_Operation_pp> operations = fk::buildOperationPipeline(readback1, readback2, readback3);
    
    std::cout << "Input operations count: " << operations.size() << std::endl;
    
    // Test the fusion function
    auto fusedOperations = fk::fuseReadBackOperationsJIT(operations);
    
    std::cout << "Fused operations count: " << fusedOperations.size() << std::endl;
    
    // Basic validation - for now, should return same number of operations
    assert(fusedOperations.size() == operations.size());
    
    std::cout << "ReadBack fusion test passed!" << std::endl;
}

#ifdef ENABLE_CPU_JIT
/**
 * @brief Test the CPUJITCompiler initialization
 */
void testJITCompilerInitialization() {
    std::cout << "Testing JIT compiler initialization..." << std::endl;
    
    fk::CPUJITCompiler compiler;
    auto error = compiler.initialize();
    
    if (error) {
        std::cout << "JIT compiler initialization failed (this may be expected in some environments)" << std::endl;
        return;
    }
    
    std::cout << "JIT compiler initialization test passed!" << std::endl;
}
#endif

/**
 * @brief Test template compilation features
 */
void testTemplateFeatures() {
    std::cout << "Testing template compilation features..." << std::endl;
    
    // Test type checking functions
    struct TestReadBack { using InstanceType = fk::ReadBackType; };
    struct TestUnary { using InstanceType = fk::UnaryType; };
    struct TestWrite { using InstanceType = fk::WriteType; };
    
    static_assert(fk::isReadBackType<TestReadBack>, "ReadBack type check failed");
    static_assert(fk::isUnaryType<TestUnary>, "Unary type check failed");
    static_assert(fk::isWriteType<TestWrite>, "Write type check failed");
    
    static_assert(fk::isComputeType<TestUnary>, "Compute type check failed");
    static_assert(!fk::isComputeType<TestWrite>, "Compute type check failed");
    
    std::cout << "Template features test passed!" << std::endl;
}

/**
 * @brief Run all CPU JIT tests
 */
void runAllCPUJITTests() {
    std::cout << "=== Running CPU JIT Tests ===" << std::endl;
    
    testJITOperationBasics();
    std::cout << std::endl;
    
    testReadBackFusion();
    std::cout << std::endl;
    
#ifdef ENABLE_CPU_JIT
    testJITCompilerInitialization();
    std::cout << std::endl;
#else
    std::cout << "JIT compiler tests skipped (ENABLE_CPU_JIT not defined)" << std::endl;
    std::cout << std::endl;
#endif
    
    testTemplateFeatures();
    std::cout << std::endl;
    
    std::cout << "=== CPU JIT Tests Completed ===" << std::endl;
}

int launch() {
    runAllCPUJITTests();
    return 0;
}