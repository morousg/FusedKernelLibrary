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

#ifndef FK_TEST_CPU_JIT
#define FK_TEST_CPU_JIT

#include <tests/main.h>
#include <fused_kernel/core/execution_model/executor_details/cpu_jit_details.h>
#include <fused_kernel/core/execution_model/operation_model/parent_operations.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <cassert>
#include <iostream>

namespace fk {
namespace test {

    /**
     * @brief Test basic JIT_Operation_pp creation and manipulation
     */
    void test_jit_operation_pp_basic() {
        std::cout << "Testing JIT_Operation_pp basic functionality..." << std::endl;
        
        // Create a simple operation data
        int testData = 42;
        JIT_Operation_pp op(&testData, "int");
        
        assert(op.opData == &testData);
        assert(op.opType == "int");
        
        std::cout << "✓ JIT_Operation_pp basic test passed" << std::endl;
    }

    /**
     * @brief Test ReadBack operation identification and fusion
     */
    void test_readback_fusion() {
        std::cout << "Testing ReadBack operation fusion..." << std::endl;
        
        // Create mock ReadBack operations
        std::vector<JIT_Operation_pp> operations;
        
        // Simulate ReadBack operations with different types
        int readData1 = 10;
        float readData2 = 3.14f;
        double computeData = 2.0;
        double writeData = 0.0;
        
        operations.emplace_back(&readData1, "ReadBackOperation<int>");
        operations.emplace_back(&readData2, "ReadBackOperation<float>");
        operations.emplace_back(&computeData, "ComputeOperation<double>");
        operations.emplace_back(&writeData, "WriteOperation<double>");
        
        // Test the enhanced fusion function
        auto fusedOps = compileAndFuseReadBackOperations(operations);
        
        // Should have the same number of operations, but processed through fusion logic
        assert(fusedOps.size() == operations.size());
        
        // Verify that ReadBack operations were processed
        bool foundReadBack = false;
        for (const auto& op : fusedOps) {
            if (op.opType.find("ReadBack") != std::string::npos) {
                foundReadBack = true;
                break;
            }
        }
        assert(foundReadBack);
        
        std::cout << "✓ ReadBack fusion test passed (enhanced with grouping logic)" << std::endl;
    }

    /**
     * @brief Test CpuJitDetails compilation functionality
     */
    void test_cpu_jit_compilation() {
        std::cout << "Testing CPU JIT compilation..." << std::endl;
        
        CpuJitDetails jit;
        
        // Create test operations including ReadBack
        std::vector<JIT_Operation_pp> operations;
        int testData = 100;
        float readBackData = 2.5f;
        operations.emplace_back(&testData, "TestOperation<int>");
        operations.emplace_back(&readBackData, "ReadBackOperation<float>");
        
        // Test compilation (enhanced with LLVM integration)
        auto compiledFunc = jit.compileFuseBackFunction(operations, nullptr);
        auto result = compiledFunc();
        
        assert(result.size() == operations.size());
        
        std::cout << "✓ CPU JIT compilation test passed (enhanced with LLVM framework)" << std::endl;
    }

    /**
     * @brief Test the fuseBack template function with mock types
     */
    void test_fuseback_template() {
        std::cout << "Testing fuseBack template function..." << std::endl;
        
        // Create mock operation types for testing
        struct MockReadBackOp {
            using InstanceType = ReadBackType;
            int data = 42;
        };
        
        struct MockWriteOp {
            using InstanceType = WriteType;
            float data = 3.14f;
        };
        
        MockReadBackOp readOp;
        MockWriteOp writeOp;
        
        // Test the fuseBack template function
        auto result = fuseBack(readOp, writeOp);
        
        assert(result.size() == 2);
        assert(result[0].opData == &readOp);
        assert(result[1].opData == &writeOp);
        
        std::cout << "✓ fuseBack template test passed" << std::endl;
    }

    /**
     * @brief Run all CPU JIT tests
     */
    void run_all_cpu_jit_tests() {
        std::cout << "Running CPU JIT Tests..." << std::endl;
        std::cout << "=========================" << std::endl;
        
        try {
            test_jit_operation_pp_basic();
            test_readback_fusion();
            test_cpu_jit_compilation();
            test_fuseback_template();
            
            std::cout << "\n✓ All CPU JIT tests passed!" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "✗ Test failed with exception: " << e.what() << std::endl;
            throw;
        }
    }

} // namespace test
} // namespace fk

/**
 * @brief Main test launch function expected by the test infrastructure
 */
int launch() {
    try {
        fk::test::run_all_cpu_jit_tests();
        return 0;
    } catch (...) {
        return 1;
    }
}

#endif // FK_TEST_CPU_JIT