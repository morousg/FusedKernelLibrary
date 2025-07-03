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

#include <fused_kernel/core/execution_model/executor_details/cpu_jit_details.h>

#ifdef FK_LLVM_AVAILABLE
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Error.h>
#endif

namespace fk {

    CpuJitDetails::CpuJitDetails() : jit_instance(nullptr), context_instance(nullptr) {
#ifdef FK_LLVM_AVAILABLE
        // Initialize LLVM native target for JIT compilation
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
        
        // Create LLVM context
        auto context = std::make_unique<llvm::LLVMContext>();
        context_instance = context.release();
        
        // Create JIT instance
        auto jitExpected = llvm::orc::LLJITBuilder().create();
        if (jitExpected) {
            jit_instance = jitExpected->release();
        } else {
            // Handle error - for now, just set to nullptr
            jit_instance = nullptr;
        }
#endif
    }

    CpuJitDetails::~CpuJitDetails() {
#ifdef FK_LLVM_AVAILABLE
        if (jit_instance) {
            delete static_cast<llvm::orc::LLJIT*>(jit_instance);
        }
        if (context_instance) {
            delete static_cast<llvm::LLVMContext*>(context_instance);
        }
#endif
    }

    std::function<std::vector<JIT_Operation_pp>()> 
    CpuJitDetails::compileFuseBackFunction(const std::vector<JIT_Operation_pp>& operations,
                                         const void* tDPPDetails) {
#ifdef FK_LLVM_AVAILABLE
        if (!jit_instance || !context_instance) {
            // Fallback to placeholder implementation if LLVM initialization failed
            return [operations]() -> std::vector<JIT_Operation_pp> {
                return operations;
            };
        }
        
        // Generate LLVM IR code for the fusion function
        std::string code = generateFusionCode(operations);
        
        // Compile the generated code using LLVM ORCv2
        return compileCode(code, operations);
#else
        // Placeholder implementation when LLVM is not available
        return [operations]() -> std::vector<JIT_Operation_pp> {
            return operations;
        };
#endif
    }

    std::string CpuJitDetails::generateFusionCode(const std::vector<JIT_Operation_pp>& operations) {
#ifdef FK_LLVM_AVAILABLE
        // Generate LLVM IR code for runtime compilation
        std::string code = R"(
; Generated fusion code for ReadBack operations
; Function that casts void* to concrete types and calls fuseBack

define void @fusion_function() {
entry:
)";
        
        // Add operations to the generated code
        for (size_t i = 0; i < operations.size(); ++i) {
            code += "  ; Operation " + std::to_string(i) + ": " + operations[i].opType + "\n";
        }
        
        code += R"(
  ret void
}
)";
        
        return code;
#else
        // Placeholder when LLVM is not available
        std::string code = "// Generated fusion code for " + std::to_string(operations.size()) + " operations\n";
        return code;
#endif
    }

    std::function<std::vector<JIT_Operation_pp>()> 
    CpuJitDetails::compileCode(const std::string& code, const std::vector<JIT_Operation_pp>& operations) {
#ifdef FK_LLVM_AVAILABLE
        if (!jit_instance || !context_instance) {
            // Fallback implementation
            return [operations]() -> std::vector<JIT_Operation_pp> {
                return operations;
            };
        }
        
        // For now, return placeholder - full LLVM IR compilation would be implemented here
        // This would involve:
        // 1. Parsing the LLVM IR string
        // 2. Creating a module from it
        // 3. Adding it to the JIT
        // 4. Looking up the compiled function
        // 5. Returning a wrapper that calls the compiled function
        
        return [operations]() -> std::vector<JIT_Operation_pp> {
            // In the full implementation, this would call the JIT-compiled function
            return operations;
        };
#else
        // Placeholder when LLVM is not available
        return [operations]() -> std::vector<JIT_Operation_pp> {
            return operations;
        };
#endif
    }

    std::vector<JIT_Operation_pp> 
    compileAndFuseReadBackOperations(const std::vector<JIT_Operation_pp>& operations,
                                   const void* tDPPDetails) {
        // Enhanced implementation that identifies ReadBack operations and groups them
        std::vector<JIT_Operation_pp> fusedOps;
        std::vector<JIT_Operation_pp> readBackGroup;
        
        for (const auto& op : operations) {
            // Check if this is a ReadBack operation by examining the type string
            if (op.opType.find("ReadBack") != std::string::npos) {
                readBackGroup.push_back(op);
            } else {
                // If we have accumulated ReadBack operations, fuse them first
                if (!readBackGroup.empty()) {
                    // Create JIT compiler instance for this group
                    CpuJitDetails jit;
                    auto fusionFunc = jit.compileFuseBackFunction(readBackGroup, tDPPDetails);
                    auto fusedGroup = fusionFunc();
                    
                    // Add the fused group to results
                    fusedOps.insert(fusedOps.end(), fusedGroup.begin(), fusedGroup.end());
                    readBackGroup.clear();
                }
                
                // Add the non-ReadBack operation
                fusedOps.push_back(op);
            }
        }
        
        // Handle any remaining ReadBack operations
        if (!readBackGroup.empty()) {
            CpuJitDetails jit;
            auto fusionFunc = jit.compileFuseBackFunction(readBackGroup, tDPPDetails);
            auto fusedGroup = fusionFunc();
            fusedOps.insert(fusedOps.end(), fusedGroup.begin(), fusedGroup.end());
        }
        
        return fusedOps;
    }

} // namespace fk