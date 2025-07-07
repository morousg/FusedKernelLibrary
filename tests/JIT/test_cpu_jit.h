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
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/algorithms/image_processing/warping.h>

#include <iostream>
#include <string>
#include <vector>
#include <cassert>

namespace fk {
namespace test {

    // Utility function to print pipeline details in the requested format
    void printPipelineDetails(const std::vector<JIT_Operation_pp>& pipeline, const std::string& label) {
        std::cout << label << ": ";
        for (size_t i = 0; i < pipeline.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << pipeline[i].getType();
        }
        std::cout << std::endl;
    }
    
    // Test basic CPU JIT functionality
    bool testBasicCPUJIT() {
        std::cout << "Testing basic CPU JIT functionality..." << std::endl;
        
        try {
            // Create a simple pipeline with basic operations
            const auto mul_op = fk::Mul<float>::build(2.0f);
            const auto add_op = fk::Add<float>::build(5.0f);
            
            // Convert to JIT_Operation_pp using the fk namespace function
            std::vector<fk::JIT_Operation_pp> pipeline = fk::buildOperationPipeline(mul_op, add_op);
            
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
            // Use real operations: readIOp, borderIOp and mul_op
            constexpr auto readIOp = fk::PerThreadRead<fk::_2D, uchar3>::build(
                fk::RawPtr<fk::_2D, uchar3>{ nullptr, { 128, 128, 128 * sizeof(uchar3) }});
            
            constexpr auto borderIOp = fk::BorderReader<fk::BorderType::CONSTANT>::build(readIOp, fk::make_set<uchar3>(0));
            
            // Verify this is indeed a ReadBack operation
            static_assert(fk::isReadBackType<decltype(borderIOp)>, "borderIOp should be ReadBackType");
            
            // Create additional operations for the pipeline
            const auto mul_op = fk::Mul<float>::build(3.0f);
            
            // Convert them into JIT_Operation_pp with the function available in fk namespace
            std::vector<fk::JIT_Operation_pp> pipeline = fk::buildOperationPipeline(readIOp, borderIOp, mul_op);
            
            // Print original pipeline details
            printPipelineDetails(pipeline, "Original pipeline");
            
            // Test the fuseBackCPU function with ReadBack operations
            auto result = cpu_jit::fuseBackCPU(pipeline);
            
            // Print fused result details
            printPipelineDetails(result, "Fused result");
            
            // Test if requiresFusion works correctly for this pipeline
            auto& compiler = cpu_jit::CPUJITCompiler::getInstance();
            bool shouldFuse = compiler.requiresFusion(pipeline);
            std::cout << "RequiresFusion result: " << (shouldFuse ? "true" : "false") << std::endl;
            
            // Verify the fusion worked as expected
            if (pipeline.size() != 3) {
                std::cerr << "Error: Expected pipeline size 3, got " << pipeline.size() << std::endl;
                return false;
            }
            
            if (result.size() != 2) {
                std::cerr << "Error: Expected result size 2, got " << result.size() << std::endl;
                return false;
            }
            
            std::cout << "CPU JIT ReadBack test completed. Input size: " << pipeline.size() 
                     << ", Output size: " << result.size() << std::endl;
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception in testCPUJITWithReadBack: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Test CPU JIT with Crop operations
    bool testCPUJITWithCrop() {
        std::cout << "Testing CPU JIT with Crop operations..." << std::endl;
        
        try {
            // Create a pipeline with Crop ReadBack operation
            constexpr auto readIOp = fk::PerThreadRead<fk::_2D, uchar3>::build(
                fk::RawPtr<fk::_2D, uchar3>{ nullptr, { 256, 256, 256 * sizeof(uchar3) }});
            
            constexpr auto cropIOp = readIOp.then(fk::Crop<>::build(fk::Rect{50, 50, 128, 128}));
            
            // Verify this is indeed a ReadBack operation
            static_assert(fk::isReadBackType<decltype(cropIOp)>, "cropIOp should be ReadBackType");
            
            // Create additional operations for the pipeline
            const auto mul_op = fk::Mul<float>::build(2.5f);
            
            // Convert them into JIT_Operation_pp - put ReadBack operation in middle position
            std::vector<fk::JIT_Operation_pp> pipeline = fk::buildOperationPipeline(readIOp, cropIOp, mul_op);
            
            // Print original pipeline details  
            printPipelineDetails(pipeline, "Original pipeline");
            
            // Test the fuseBackCPU function with Crop operations
            auto result = cpu_jit::fuseBackCPU(pipeline);
            
            // Print fused result details
            printPipelineDetails(result, "Fused result");
            
            // Verify the fusion worked as expected
            if (pipeline.size() != 3) {
                std::cerr << "Error: Expected pipeline size 3, got " << pipeline.size() << std::endl;
                return false;
            }
            
            if (result.size() != 2) {
                std::cerr << "Error: Expected result size 2, got " << result.size() << std::endl;
                return false;
            }
            
            std::cout << "CPU JIT Crop test completed. Input size: " << pipeline.size() 
                     << ", Output size: " << result.size() << std::endl;
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception in testCPUJITWithCrop: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Test CPU JIT with Resize operations  
    bool testCPUJITWithResize() {
        std::cout << "Testing CPU JIT with Resize operations..." << std::endl;
        
        try {
            // Create a pipeline with Resize ReadBack operation
            constexpr auto readIOp = fk::PerThreadRead<fk::_2D, float3>::build(
                fk::RawPtr<fk::_2D, float3>{ nullptr, { 512, 512, 512 * sizeof(float3) }});
            
            constexpr auto resizeIOp = readIOp.then(fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::Size{256, 256}));
            
            // Verify this is indeed a ReadBack operation
            static_assert(fk::isReadBackType<decltype(resizeIOp)>, "resizeIOp should be ReadBackType");
            
            // Create additional operations for the pipeline
            const auto add_op = fk::Add<float>::build(1.5f);
            
            // Convert them into JIT_Operation_pp - put ReadBack operation in middle position
            std::vector<fk::JIT_Operation_pp> pipeline = fk::buildOperationPipeline(readIOp, resizeIOp, add_op);
            
            // Print original pipeline details
            printPipelineDetails(pipeline, "Original pipeline");
            
            // Test the fuseBackCPU function with Resize operations
            auto result = cpu_jit::fuseBackCPU(pipeline);
            
            // Print fused result details
            printPipelineDetails(result, "Fused result");
            
            // Verify the fusion worked as expected
            if (pipeline.size() != 3) {
                std::cerr << "Error: Expected pipeline size 3, got " << pipeline.size() << std::endl;
                return false;
            }
            
            if (result.size() != 2) {
                std::cerr << "Error: Expected result size 2, got " << result.size() << std::endl;
                return false;
            }
            
            std::cout << "CPU JIT Resize test completed. Input size: " << pipeline.size() 
                     << ", Output size: " << result.size() << std::endl;
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception in testCPUJITWithResize: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Test CPU JIT with Warp operations
    bool testCPUJITWithWarp() {
        std::cout << "Testing CPU JIT with Warp operations..." << std::endl;
        
        try {
            // Create a pipeline with Warp ReadBack operation
            constexpr auto readIOp = fk::PerThreadRead<fk::_2D, float4>::build(
                fk::RawPtr<fk::_2D, float4>{ nullptr, { 320, 240, 320 * sizeof(float4) }});
            
            constexpr auto warpIOp = readIOp.then(fk::Warping<fk::WarpType::Perspective>::build(
                fk::WarpingParameters<fk::WarpType::Perspective>{}));
            
            // Verify this is indeed a ReadBack operation
            static_assert(fk::isReadBackType<decltype(warpIOp)>, "warpIOp should be ReadBackType");
            
            // Create additional operations for the pipeline
            const auto div_op = fk::Div<float>::build(2.0f);
            
            // Convert them into JIT_Operation_pp - put ReadBack operation in middle position
            std::vector<fk::JIT_Operation_pp> pipeline = fk::buildOperationPipeline(readIOp, warpIOp, div_op);
            
            // Print original pipeline details
            printPipelineDetails(pipeline, "Original pipeline");
            
            // Test the fuseBackCPU function with Warp operations
            auto result = cpu_jit::fuseBackCPU(pipeline);
            
            // Print fused result details
            printPipelineDetails(result, "Fused result");
            
            // Verify the fusion worked as expected
            if (pipeline.size() != 3) {
                std::cerr << "Error: Expected pipeline size 3, got " << pipeline.size() << std::endl;
                return false;
            }
            
            if (result.size() != 2) {
                std::cerr << "Error: Expected result size 2, got " << result.size() << std::endl;
                return false;
            }
            
            std::cout << "CPU JIT Warp test completed. Input size: " << pipeline.size() 
                     << ", Output size: " << result.size() << std::endl;
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception in testCPUJITWithWarp: " << e.what() << std::endl;
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
            
            // Convert to JIT_Operation_pp using the fk namespace function
            std::vector<fk::JIT_Operation_pp> pipeline1 = fk::buildOperationPipeline(mul_op, add_op);
            
            if (compiler.requiresFusion(pipeline1)) {
                std::cerr << "Error: Pipeline without ReadBack should not require fusion" << std::endl;
                return false;
            }
            
            // Test case 2: ReadBack with compute operations - should require fusion
            // BorderReader creates ReadBack operations, when not in first position requires fusion
            constexpr auto readIOp = fk::PerThreadRead<fk::_2D, uchar3>::build(
                fk::RawPtr<fk::_2D, uchar3>{ nullptr, { 128, 128, 128 * sizeof(uchar3) }});
            constexpr auto borderIOp = fk::BorderReader<fk::BorderType::CONSTANT>::build(readIOp, fk::make_set<uchar3>(0));
            
            // Convert to JIT_Operation_pp using the fk namespace function
            // This puts borderIOp (ReadBack) in second position, should require fusion
            std::vector<fk::JIT_Operation_pp> pipeline2 = fk::buildOperationPipeline(add_op, borderIOp);
            
            if (!compiler.requiresFusion(pipeline2)) {
                std::cerr << "Error: Pipeline with ReadBack operations NOT in first position should require fusion" << std::endl;
                return false;
            }
            
            // Test case 3: ReadBack in first position - should NOT require fusion
            std::vector<fk::JIT_Operation_pp> pipeline3 = fk::buildOperationPipeline(borderIOp, add_op);
            
            if (compiler.requiresFusion(pipeline3)) {
                std::cerr << "Error: Pipeline with ReadBack in first position should NOT require fusion" << std::endl;
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
            
            // Convert to JIT_Operation_pp using the fk namespace function
            std::vector<fk::JIT_Operation_pp> pipeline = fk::buildOperationPipeline(mul_op);
            
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
    allTestsPassed &= fk::test::testCPUJITWithCrop();
    allTestsPassed &= fk::test::testCPUJITWithResize();
    allTestsPassed &= fk::test::testCPUJITWithWarp();
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