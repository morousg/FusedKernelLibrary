/* Copyright 2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <sstream>

#include <fused_kernel/core/execution_model/operation_model/operation_types.h>
#include <fused_kernel/core/execution_model/operation_model/iop_fuser.h>

#ifdef ENABLE_LLVM_JIT
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Error.h>
#endif

namespace fk {

/**
 * @brief JIT_Operation_pp: A polymorphic operation wrapper for JIT compilation
 * Contains type information as a string and data as a void pointer for runtime processing
 */
struct JIT_Operation_pp {
    std::string opType;     ///< String representation of the operation type
    void* opData;          ///< Pointer to the actual operation data
    
    JIT_Operation_pp(const std::string& type, void* data) 
        : opType(type), opData(data) {}
    
    template<typename T>
    JIT_Operation_pp(const T& operation) 
        : opType(typeid(T).name()), opData(const_cast<void*>(static_cast<const void*>(&operation))) {}
};

#ifdef ENABLE_LLVM_JIT

/**
 * @brief CPU JIT Compiler for fusing ReadBack operations
 * Uses LLVM ORCv2 for runtime compilation of operation fusion functions
 */
class CPUJITCompiler {
public:
    CPUJITCompiler();
    ~CPUJITCompiler();
    
    /**
     * @brief Compile and execute a fusion function for the given operations
     * @param operations Vector of JIT operations to fuse
     * @return Resulting vector of fused operations
     */
    std::vector<JIT_Operation_pp> fuseOperations(const std::vector<JIT_Operation_pp>& operations);
    
private:
    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::unique_ptr<llvm::LLVMContext> context_;
    
    /**
     * @brief Generate LLVM IR code for the fusion function
     * @param operations Operations to generate code for
     * @return Generated LLVM module
     */
    std::unique_ptr<llvm::Module> generateFusionCode(const std::vector<JIT_Operation_pp>& operations);
    
    /**
     * @brief Build type-specific casting and fusion logic
     * @param operations Operations to analyze
     * @return Generated function code as string
     */
    std::string buildFusionFunctionCode(const std::vector<JIT_Operation_pp>& operations);
    
    /**
     * @brief Analyze operations to determine if ReadBack fusion is needed
     * @param operations Operations to analyze
     * @return True if ReadBack operations are present and need fusion
     */
    bool needsReadBackFusion(const std::vector<JIT_Operation_pp>& operations);
};

#endif // ENABLE_LLVM_JIT

/**
 * @brief Template function for fusing ReadBack operations at compile time
 * This function is included in the JIT-compiled code for runtime execution
 */
template <typename Read, typename Next, typename... IOps>
constexpr inline std::vector<JIT_Operation_pp> fuseBack(const Read& read, const Next& nextOp, const IOps&... iOps) {
    static_assert(!isReadType<Next>, "A Read Operation can not go after another Read Operation, it has to be ReadBack");
    
    std::vector<JIT_Operation_pp> result;
    
    if constexpr (sizeof...(iOps) > 0) {
        constexpr bool nextIsReadBack = isReadBackType<Next>;
        constexpr bool iOpsContainsReadBack = (isReadBackType<IOps> || ...);
        constexpr bool nextIsComputeOrMidWrite = isComputeType<Next> || isMidWriteType<Next>;
        
        if constexpr (nextIsReadBack || (nextIsComputeOrMidWrite && iOpsContainsReadBack)) {
            // Fuse the read and next operation
            auto fusedOp = fuse(read, nextOp);
            result.emplace_back(fusedOp);
            
            // Continue with remaining operations
            auto remaining = fuseBack(iOps...);
            result.insert(result.end(), remaining.begin(), remaining.end());
        } else {
            // Build operation pipeline without fusion
            result.emplace_back(read);
            result.emplace_back(nextOp);
            auto remaining = fuseBack(iOps...);
            result.insert(result.end(), remaining.begin(), remaining.end());
        }
    } else {
        static_assert(isWriteType<Next>, "Last IOp must be WriteType");
        // Fuse read and write operations
        auto fusedOp = fuse(read, nextOp);
        result.emplace_back(fusedOp);
    }
    
    return result;
}

/**
 * @brief Runtime compilation entry point
 * Builds and compiles the fusion function, then executes it
 */
std::vector<JIT_Operation_pp> compileAndFuseOperations(const std::vector<JIT_Operation_pp>& operations);

/**
 * @brief Fallback implementation when LLVM JIT is not available
 * Returns the operations unchanged
 */
std::vector<JIT_Operation_pp> fallbackFuseOperations(const std::vector<JIT_Operation_pp>& operations);

} // namespace fk