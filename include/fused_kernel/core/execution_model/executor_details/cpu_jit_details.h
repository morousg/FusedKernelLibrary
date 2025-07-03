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

#ifndef FK_CPU_JIT_DETAILS
#define FK_CPU_JIT_DETAILS

#include <fused_kernel/core/execution_model/operation_model/operation_types.h>
#include <fused_kernel/core/execution_model/operation_model/instantiable_operations.h>
#include <fused_kernel/core/execution_model/data_parallel_patterns.h>
#include <vector>
#include <string>
#include <memory>
#include <functional>

// Forward declarations for LLVM (when available)
#ifdef FK_LLVM_AVAILABLE
namespace llvm {
    class LLVMContext;
    class Module;
    class Function;
    class ExecutionEngine;
}

namespace llvm::orc {
    class LLJIT;
    class ExecutionSession;
    class JITDylib;
}
#endif

namespace fk {

    /**
     * @brief Represents a JIT operation with runtime type information
     */
    struct JIT_Operation_pp {
        void* opData;           ///< Pointer to operation data
        std::string opType;     ///< String representation of the operation type
        
        JIT_Operation_pp(void* data, const std::string& type) 
            : opData(data), opType(type) {}
    };

    /**
     * @brief CPU JIT compilation details for fusing ReadBack operations
     */
    class CpuJitDetails {
    private:
        // Use void pointers to avoid incomplete type issues with forward declarations
        void* jit_instance;
        void* context_instance;
        
    public:
        CpuJitDetails();
        ~CpuJitDetails();
        
        /**
         * @brief Generates and compiles a function that fuses ReadBack operations
         * @param operations Vector of JIT operations to fuse
         * @param tDPPDetails Transform DPP details for execution
         * @return Compiled function pointer that returns fused operations vector
         */
        std::function<std::vector<JIT_Operation_pp>()> 
        compileFuseBackFunction(const std::vector<JIT_Operation_pp>& operations,
                               const void* tDPPDetails);
        
    private:
        /**
         * @brief Generates LLVM IR code for the runtime fusion function
         */
        std::string generateFusionCode(const std::vector<JIT_Operation_pp>& operations);
        
        /**
         * @brief Compiles the generated code using LLVM ORCv2
         */
        std::function<std::vector<JIT_Operation_pp>()> 
        compileCode(const std::string& code, const std::vector<JIT_Operation_pp>& operations);
    };

    /**
     * @brief Template function for fusing ReadBack operations (compile-time template)
     * This function would be included in the generated code and called with proper types
     */
    template <typename Read, typename Next, typename... IOps>
    constexpr inline std::vector<JIT_Operation_pp> fuseBack(const Read& read, const Next& nextOp, const IOps&... iOps) {
        static_assert(!isReadType<Next>, "A Read Operation can not go after another Read Operation, it has to be ReadBack");
        
        if constexpr (sizeof...(iOps) > 0) {
            constexpr bool nextIsReadBack = isReadBackType<Next>;
            constexpr bool iOpsContainsReadBack = (isReadBackType<IOps> || ...);
            constexpr bool nextIsComputeOrMidWrite = isComputeType<Next> || isMidWriteType<Next>;
            
            if constexpr (nextIsReadBack || (nextIsComputeOrMidWrite && iOpsContainsReadBack)) {
                // For now, return a placeholder - this would call the actual fusion logic
                std::vector<JIT_Operation_pp> result;
                result.emplace_back(const_cast<void*>(static_cast<const void*>(&read)), typeid(Read).name());
                result.emplace_back(const_cast<void*>(static_cast<const void*>(&nextOp)), typeid(Next).name());
                return result;
            } else {
                // Build operation pipeline without fusion
                std::vector<JIT_Operation_pp> result;
                result.emplace_back(const_cast<void*>(static_cast<const void*>(&read)), typeid(Read).name());
                result.emplace_back(const_cast<void*>(static_cast<const void*>(&nextOp)), typeid(Next).name());
                return result;
            }
        } else {
            static_assert(isWriteType<Next>, "Last IOp must be WriteType");
            std::vector<JIT_Operation_pp> result;
            result.emplace_back(const_cast<void*>(static_cast<const void*>(&read)), typeid(Read).name());
            result.emplace_back(const_cast<void*>(static_cast<const void*>(&nextOp)), typeid(Next).name());
            return result;
        }
    }

    /**
     * @brief Compile-time function that builds, compiles and executes the runtime fusion
     * @param operations Vector of JIT operations containing ReadBack operations to fuse
     * @param tDPPDetails Transform DPP details for execution context
     * @return Vector of fused JIT operations
     */
    std::vector<JIT_Operation_pp> 
    compileAndFuseReadBackOperations(const std::vector<JIT_Operation_pp>& operations,
                                   const void* tDPPDetails = nullptr);

} // namespace fk

#endif // FK_CPU_JIT_DETAILS