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

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <fused_kernel/core/execution_model/operation_model/operation_types.h>
#include <fused_kernel/core/execution_model/operation_model/iop_fuser.h>
#include <fused_kernel/core/execution_model/data_parallel_patterns.h>

#ifdef ENABLE_CPU_JIT
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#endif

namespace fk {

    /**
     * @brief JIT_Operation_pp: Runtime polymorphic wrapper for operations
     * Contains type-erased operation data and type information for runtime compilation
     */
    struct JIT_Operation_pp {
        void* opData;           // Pointer to the actual operation data
        std::string opType;     // String representation of the operation type
        
        // Function pointer for execution (to be set by runtime compilation)
        std::function<void()> exec;
        
        JIT_Operation_pp(void* data, const std::string& type) 
            : opData(data), opType(type) {}
    };

    // Forward declarations for functions that will be called from fuseBack
    template <typename TDPPDetails, typename... IOps>
    void fuseReadsLaunchTransformDPP(const TDPPDetails& tDPPDetails, const IOps&... iOps) {
        // Placeholder implementation - would call the actual DPP execution
        // This would be properly implemented to launch the transform DPP
    }

    template <typename... IOps>
    void buildOperationPipeline(const IOps&... iOps) {
        // Placeholder implementation - would build operation pipeline
        // This would be properly implemented to build the operation pipeline
    }

    /**
     * @brief The template function from the issue description that needs to be compiled at runtime
     */
    template <typename TDPPDetails, typename Read, typename Next, typename... IOps>
    constexpr inline std::vector<JIT_Operation_pp> fuseBack(const TDPPDetails& tDPPDetails, const Read& read, const Next& nextOp, const IOps&... iOps) {
        static_assert(!isReadType<Next>, "A Read Operation can not go after another Read Operation, it has to be ReadBack");
        if constexpr (sizeof...(iOps) > 0) {
            constexpr bool nextIsReadBack = isReadBackType<Next>;
            constexpr bool iOpsContainsReadBack = (isReadBackType<IOps> || ...);
            constexpr bool nextIsComputeOrMidWrite = isComputeType<Next> || isMidWriteType<Next>;
            if constexpr (nextIsReadBack || (nextIsComputeOrMidWrite && iOpsContainsReadBack)) {
                auto fused = Fuser{}.fuse(read, nextOp);
                fuseReadsLaunchTransformDPP(tDPPDetails, fused, iOps...);
            } else {
                buildOperationPipeline(read, nextOp, iOps...);
            }
        } else {
            static_assert(isWriteType<Next>, "Last IOp must be WriteType");
            buildOperationPipeline(read, nextOp);
        }
        
        // For now, return the input operations as JIT_Operation_pp
        // This will be properly implemented when the runtime compilation is complete
        std::vector<JIT_Operation_pp> result;
        result.emplace_back(const_cast<void*>(static_cast<const void*>(&read)), typeid(Read).name());
        result.emplace_back(const_cast<void*>(static_cast<const void*>(&nextOp)), typeid(Next).name());
        (result.emplace_back(const_cast<void*>(static_cast<const void*>(&iOps)), typeid(IOps).name()), ...);
        return result;
    }

#ifdef ENABLE_CPU_JIT

    /**
     * @brief CPU JIT Runtime Compiler using LLVM ORCv2
     */
    class CPUJITCompiler {
    private:
        std::unique_ptr<llvm::orc::LLJIT> jit;
        llvm::LLVMContext context;
        
    public:
        CPUJITCompiler();
        ~CPUJITCompiler() = default;
        
        /**
         * @brief Initialize the JIT compiler
         * @return Error status
         */
        llvm::Error initialize();
        
        /**
         * @brief Generate and compile the runtime function that casts void* operations to proper types
         * @param operations Vector of JIT operations with type information
         * @return Compiled function pointer
         */
        std::function<std::vector<JIT_Operation_pp>(const std::vector<JIT_Operation_pp>&)> 
            compileRuntimeFusion(const std::vector<JIT_Operation_pp>& operations);
        
    private:
        /**
         * @brief Generate LLVM IR for the runtime casting and fusion function
         * @param operations Vector of operations to generate IR for
         * @return LLVM Module containing the generated function
         */
        std::unique_ptr<llvm::Module> generateRuntimeFusionIR(const std::vector<JIT_Operation_pp>& operations);
        
        /**
         * @brief Generate casting code for a specific operation type
         * @param builder LLVM IR builder
         * @param opDataPtr Pointer to operation data
         * @param typeName Type name for casting
         * @return LLVM Value representing the casted operation
         */
        llvm::Value* generateCastingCode(llvm::IRBuilder<>& builder, llvm::Value* opDataPtr, const std::string& typeName);
    };

#endif // ENABLE_CPU_JIT

    /**
     * @brief Main entry point for CPU JIT fusion of ReadBack operations
     * This function takes a vector of JIT_Operation_pp containing ReadBack operations,
     * builds a runtime function to cast and fuse them, compiles it, and executes it.
     * @param operations Vector of JIT operations to fuse
     * @return Vector of fused JIT operations
     */
    std::vector<JIT_Operation_pp> fuseReadBackOperationsJIT(const std::vector<JIT_Operation_pp>& operations);

    /**
     * @brief Helper function to create JIT_Operation_pp from typed operations
     * @tparam T Type of the operation
     * @param operation Typed operation to wrap
     * @return JIT_Operation_pp wrapper
     */
    template <typename T>
    JIT_Operation_pp createJITOperation(const T& operation) {
        return JIT_Operation_pp(const_cast<void*>(static_cast<const void*>(&operation)), typeid(T).name());
    }

} // namespace fk

#endif // FK_CPU_JIT_DETAILS_H