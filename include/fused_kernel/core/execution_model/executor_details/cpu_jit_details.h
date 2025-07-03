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

#ifndef FK_CPU_JIT_DETAILS_H
#define FK_CPU_JIT_DETAILS_H

#if defined(LLVM_JIT_ENABLED)

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Verifier.h>

#include <fused_kernel/core/execution_model/executor_details/jit_executor_details.h>
#include <fused_kernel/core/execution_model/operation_model/operation_types.h>
#include <fused_kernel/core/execution_model/operation_model/iop_fuser.h>

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <sstream>

namespace fk {
namespace cpu_jit {

    // Function pointer type for the runtime compiled fuse function
    using FuseBackFunctionPtr = std::vector<JIT_Operation_pp>(*)(void** opDataPtrs, size_t numOps);

    class CPUJITCompiler {
    private:
        std::unique_ptr<llvm::orc::LLJIT> jit_;
        std::unique_ptr<llvm::LLVMContext> context_;
        std::unordered_map<std::string, FuseBackFunctionPtr> compiledFunctions_;
        
        static bool llvmInitialized_;
        
        void initializeLLVM() {
            if (!llvmInitialized_) {
                llvm::InitializeNativeTarget();
                llvm::InitializeNativeTargetAsmPrinter();
                llvm::InitializeNativeTargetAsmParser();
                llvmInitialized_ = true;
            }
        }
        
        // Generate the signature hash for caching
        std::string generateSignature(const std::vector<std::string>& typeNames) {
            std::stringstream ss;
            for (size_t i = 0; i < typeNames.size(); ++i) {
                ss << typeNames[i];
                if (i < typeNames.size() - 1) {
                    ss << "_";
                }
            }
            return ss.str();
        }
        
        // Generate C++ source code for the runtime function
        std::string generateFuseBackSource(const std::vector<std::string>& typeNames) {
            std::stringstream source;
            
            source << R"(
#include <vector>
#include <fused_kernel/core/execution_model/executor_details/jit_executor_details.h>
#include <fused_kernel/core/execution_model/operation_model/operation_types.h>
#include <fused_kernel/core/execution_model/operation_model/iop_fuser.h>

extern "C" {

std::vector<fk::JIT_Operation_pp>* fuseBackRuntime(void** opDataPtrs, size_t numOps) {
    if (numOps < 2) {
        return nullptr;
    }
    
    // Cast the void pointers to their actual types
)";
            
            // Generate casting code for each operation
            for (size_t i = 0; i < typeNames.size(); ++i) {
                source << "    const " << typeNames[i] << "& op" << i 
                      << " = *reinterpret_cast<const " << typeNames[i] << "*>(opDataPtrs[" << i << "]);\n";
            }
            
            source << "\n    // Call the original fuseBack function\n";
            source << "    auto result = fk::fuseBack(";
            
            for (size_t i = 0; i < typeNames.size(); ++i) {
                source << "op" << i;
                if (i < typeNames.size() - 1) {
                    source << ", ";
                }
            }
            
            source << ");\n";
            source << "    \n    // Return the result as a heap-allocated vector\n";
            source << "    return new std::vector<fk::JIT_Operation_pp>(std::move(result));\n";
            source << "}\n\n}";
            
            return source.str();
        }
        
        // Compile the generated source using LLVM JIT
        FuseBackFunctionPtr compileFunction(const std::string& source) {
            context_ = std::make_unique<llvm::LLVMContext>();
            auto module = std::make_unique<llvm::Module>("cpu_jit_module", *context_);
            
            // For simplicity, we'll use a mock compilation here
            // In a real implementation, you would parse the C++ source using Clang
            // and generate LLVM IR, then compile it with the JIT
            
            // This is a simplified mock implementation that returns nullptr
            // A full implementation would require integrating Clang frontend
            return nullptr;
        }
        
    public:
        CPUJITCompiler() {
            initializeLLVM();
            
            auto jitExpected = llvm::orc::LLJITBuilder().create();
            if (auto err = jitExpected.takeError()) {
                llvm::errs() << "Failed to create LLJIT: " << err << "\n";
                return;
            }
            jit_ = std::move(*jitExpected);
        }
        
        ~CPUJITCompiler() = default;
        
        // Compile and execute fuseBack for the given operation pipeline
        std::vector<JIT_Operation_pp> compileFuseBack(const std::vector<JIT_Operation_pp>& pipeline) {
            if (pipeline.size() < 2) {
                return pipeline; // Nothing to fuse
            }
            
            // Extract type names
            std::vector<std::string> typeNames;
            for (const auto& op : pipeline) {
                typeNames.push_back(op.getType());
            }
            
            // Generate signature for caching
            std::string signature = generateSignature(typeNames);
            
            // Check if already compiled
            auto it = compiledFunctions_.find(signature);
            if (it != compiledFunctions_.end() && it->second != nullptr) {
                // Use cached function
                std::vector<void*> opDataPtrs;
                for (const auto& op : pipeline) {
                    opDataPtrs.push_back(op.getData());
                }
                
                auto result = it->second(opDataPtrs.data(), opDataPtrs.size());
                return result;
            }
            
            // For now, return a fallback implementation
            // In a complete implementation, this would compile and execute the runtime function
            return pipeline;
        }
        
        // Static method to get the singleton instance
        static CPUJITCompiler& getInstance() {
            static CPUJITCompiler instance;
            return instance;
        }
    };
    
    // Static member definition
    bool CPUJITCompiler::llvmInitialized_ = false;
    
    // Main API function to fuse ReadBack operations from a pipeline
    std::vector<JIT_Operation_pp> fuseBackCPU(const std::vector<JIT_Operation_pp>& pipeline) {
        // Check if pipeline contains ReadBack operations that need fusing
        bool hasReadBack = false;
        for (const auto& op : pipeline) {
            // This is a simplified check - in reality you'd need to parse the type string
            // to determine if it's a ReadBack type
            if (op.getType().find("ReadBack") != std::string::npos) {
                hasReadBack = true;
                break;
            }
        }
        
        if (!hasReadBack) {
            return pipeline; // No ReadBack operations to fuse
        }
        
        // Use the JIT compiler to fuse the operations
        return CPUJITCompiler::getInstance().compileFuseBack(pipeline);
    }

} // namespace cpu_jit
} // namespace fk

#else

namespace fk {
namespace cpu_jit {
    
    // Fallback implementation when LLVM JIT is not enabled
    std::vector<JIT_Operation_pp> fuseBackCPU(const std::vector<JIT_Operation_pp>& pipeline) {
        // Just return the original pipeline without fusion
        return pipeline;
    }

} // namespace cpu_jit
} // namespace fk

#endif // LLVM_JIT_ENABLED

#endif // FK_CPU_JIT_DETAILS_H