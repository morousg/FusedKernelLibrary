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

#include <fused_kernel/core/execution_model/executor_details/cpu_jit_details.h>
#include <fused_kernel/core/execution_model/executor_details/jit_executor_details.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/core/utils/type_to_string.h>
#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/border_reader.h>

#include <iostream>
#include <string>
#include <vector>
#include <cassert>

namespace fk {
namespace test {

    // Test helper function to create a mock JIT_Operation_pp
    template<typename T>
    JIT_Operation_pp createMockOperation(const T& op) {
        return JIT_Operation_pp(typeToString<T>(), &op, sizeof(T));
    }
    
    // Test basic CPU JIT functionality
    bool testBasicCPUJIT() {
        std::cout << "Testing basic CPU JIT functionality..." << std::endl;
        
        try {
            // Create a simple pipeline with basic operations
            const auto mul_op = fk::Mul<float>::build(2.0f);
            const auto add_op = fk::Add<float>::build(5.0f);
            
            std::vector<JIT_Operation_pp> pipeline;
            pipeline.push_back(createMockOperation(mul_op));
            pipeline.push_back(createMockOperation(add_op));
            
            // Test the fuseBackCPU function
            auto result = cpu_jit::fuseBackCPU(pipeline);
            
            // This test should always return the same, because the functions used are not ReadBack
            // Since Mul and Add are BinaryType operations, not ReadBackType, no fusion should occur
            if (result.size() != pipeline.size()) {
                std::cerr << "Error: Expected same size for non-ReadBack operations, got " 
                         << result.size() << " instead of " << pipeline.size() << std::endl;
                return false;
            }
            
            std::cout << "Basic CPU JIT test passed. Input size: " << pipeline.size() 
                     << ", Output size: " << result.size() << std::endl;
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception in testBasicCPUJIT: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Test CPU JIT with ReadBack operations
    bool testCPUJITWithReadBack() {
        std::cout << "Testing CPU JIT with ReadBack operations..." << std::endl;
        
        try {
            // Create a pipeline that includes actual ReadBack operations
            // Use BorderReader which creates ReadBack operations
            constexpr auto readIOp = fk::PerThreadRead<fk::_2D, uchar3>::build(
                fk::RawPtr<fk::_2D, uchar3>{ nullptr, { 128, 128, 128 * sizeof(uchar3) }});
            
            constexpr auto borderIOp = fk::BorderReader<fk::BorderType::CONSTANT>::build(readIOp, fk::make_set<uchar3>(0));
            
            // Verify this is indeed a ReadBack operation
            static_assert(fk::isReadBackType<decltype(borderIOp)>, "borderIOp should be ReadBackType");
            
            // Create additional operations for the pipeline
            const auto mul_op = fk::Mul<float>::build(3.0f);
            
            std::vector<JIT_Operation_pp> pipeline;
            
            // Add the ReadBack operation
            pipeline.push_back(createMockOperation(borderIOp));
            // Add compute operation after ReadBack
            pipeline.push_back(createMockOperation(mul_op));
            
            // Test the fuseBackCPU function with ReadBack operations
            auto result = cpu_jit::fuseBackCPU(pipeline);
            
            // The function should process the pipeline
            std::cout << "CPU JIT ReadBack test completed. Input size: " << pipeline.size() 
                     << ", Output size: " << result.size() << std::endl;
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception in testCPUJITWithReadBack: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Test CPU JIT fusion analysis
    bool testFusionAnalysis() {
        std::cout << "Testing CPU JIT fusion analysis..." << std::endl;
        
        try {
            auto& compiler = cpu_jit::CPUJITCompiler::getInstance();
            
            // Test case 1: No ReadBack operations - should not require fusion
            const auto mul_op = fk::Mul<float>::build(2.0f);
            const auto add_op = fk::Add<float>::build(5.0f);
            
            std::vector<JIT_Operation_pp> pipeline1;
            pipeline1.push_back(createMockOperation(mul_op));
            pipeline1.push_back(createMockOperation(add_op));
            
            if (compiler.requiresFusion(pipeline1)) {
                std::cerr << "Error: Pipeline without ReadBack should not require fusion" << std::endl;
                return false;
            }
            
            // Test case 2: ReadBack with compute operations - should require fusion
            constexpr auto readIOp = fk::PerThreadRead<fk::_2D, uchar3>::build(
                fk::RawPtr<fk::_2D, uchar3>{ nullptr, { 128, 128, 128 * sizeof(uchar3) }});
            constexpr auto borderIOp = fk::BorderReader<fk::BorderType::CONSTANT>::build(readIOp, fk::make_set<uchar3>(0));
            
            std::vector<JIT_Operation_pp> pipeline2;
            pipeline2.push_back(createMockOperation(borderIOp));
            pipeline2.push_back(createMockOperation(add_op));
            
            if (!compiler.requiresFusion(pipeline2)) {
                std::cerr << "Error: Pipeline with ReadBack operations should require fusion" << std::endl;
                return false;
            }
            
            std::cout << "Fusion analysis test passed." << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception in testFusionAnalysis: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Test empty pipeline handling
    bool testEmptyPipeline() {
        std::cout << "Testing empty pipeline handling..." << std::endl;
        
        try {
            std::vector<JIT_Operation_pp> emptyPipeline;
            auto result = cpu_jit::fuseBackCPU(emptyPipeline);
            
            if (!result.empty()) {
                std::cerr << "Error: Empty pipeline should return empty result" << std::endl;
                return false;
            }
            
            std::cout << "Empty pipeline test passed." << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception in testEmptyPipeline: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Test single operation pipeline
    bool testSingleOperation() {
        std::cout << "Testing single operation pipeline..." << std::endl;
        
        try {
            const auto mul_op = fk::Mul<float>::build(4.0f);
            std::vector<JIT_Operation_pp> pipeline;
            pipeline.push_back(createMockOperation(mul_op));
            
            auto result = cpu_jit::fuseBackCPU(pipeline);
            
            // Single operation should return as-is
            if (result.size() != 1) {
                std::cerr << "Error: Single operation pipeline should return one operation" << std::endl;
                return false;
            }
            
            std::cout << "Single operation test passed." << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception in testSingleOperation: " << e.what() << std::endl;
            return false;
        }
    }

#if defined(LLVM_JIT_ENABLED)
    // Test LLVM JIT compiler initialization (only when LLVM is enabled)
    bool testLLVMInitialization() {
        std::cout << "Testing LLVM JIT compiler initialization..." << std::endl;
        
        try {
            // Get the singleton instance to test initialization
            auto& compiler = cpu_jit::CPUJITCompiler::getInstance();
            
            // If we get here without exception, initialization was successful
            std::cout << "LLVM JIT compiler initialization test passed." << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception in testLLVMInitialization: " << e.what() << std::endl;
            return false;
        }
    }
#endif

} // namespace test
} // namespace fk

int launch() {
    std::cout << "Running CPU JIT tests..." << std::endl;
    
    bool allTestsPassed = true;
    
    // Run all tests
    allTestsPassed &= fk::test::testBasicCPUJIT();
    allTestsPassed &= fk::test::testCPUJITWithReadBack();
    allTestsPassed &= fk::test::testFusionAnalysis();
    allTestsPassed &= fk::test::testEmptyPipeline();
    allTestsPassed &= fk::test::testSingleOperation();
    
#if defined(LLVM_JIT_ENABLED)
    std::cout << "LLVM JIT is enabled - running LLVM-specific tests..." << std::endl;
    allTestsPassed &= fk::test::testLLVMInitialization();
#else
    std::cout << "LLVM JIT is not enabled - skipping LLVM-specific tests." << std::endl;
#endif
    
    if (allTestsPassed) {
        std::cout << "All CPU JIT tests passed!" << std::endl;
        return 0;
    } else {
        std::cerr << "Some CPU JIT tests failed!" << std::endl;
        return 1;
    }
}

#endif // FK_TEST_CPU_JIT