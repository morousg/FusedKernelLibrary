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
#include <iostream>
#include <fused_kernel/core/execution_model/operation_model/operation_types.h>
#include <fused_kernel/core/execution_model/operation_model/iop_fuser.h>
#include <fused_kernel/core/execution_model/data_parallel_patterns.h>

#ifdef ENABLE_CPU_JIT
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#endif

namespace fk {

    /**
     * @brief JIT_Operation_pp: Runtime polymorphic wrapper for operations
     * Contains type-erased operation data and type information for runtime compilation
     */
    struct JIT_Operation_pp {
        void* opData;           // Pointer to the actual operation data
        std::string opType;     // String representation of the operation type
        
        JIT_Operation_pp(void* data, const std::string& type) 
            : opData(data), opType(type) {}
    };

    // Forward declarations for functions that will be called from fuseBack
    template <typename... IOps>
    std::vector<JIT_Operation_pp> fuseReadsLaunchTransformDPP(const IOps&... iOps) {
        // Placeholder implementation - would call the actual DPP execution
        // This would be properly implemented to launch the transform DPP
        return std::vector<JIT_Operation_pp>();
    }

    template <typename... IOps>
    std::vector<JIT_Operation_pp> buildOperationPipeline(const IOps&... iOps) {
        // Placeholder implementation - would build operation pipeline
        // This would be properly implemented to build the operation pipeline
        return std::vector<JIT_Operation_pp>();
    }

    /**
     * @brief The template function from the issue description that needs to be compiled at runtime
     */
    template <typename Read, typename Next, typename... IOps>
    constexpr inline std::vector<JIT_Operation_pp> fuseBack(const Read& read, const Next& nextOp, const IOps&... iOps) {
        static_assert(!isReadType<Next>, "A Read Operation can not go after another Read Operation, it has to be ReadBack");
        if constexpr (sizeof...(iOps) > 0) {
            constexpr bool nextIsReadBack = isReadBackType<Next>;
            constexpr bool iOpsContainsReadBack = (isReadBackType<IOps> || ...);
            constexpr bool nextIsComputeOrMidWrite = isComputeType<Next> || isMidWriteType<Next>;
            if constexpr (nextIsReadBack || (nextIsComputeOrMidWrite && iOpsContainsReadBack)) {
                auto fused = Fuser{}.fuse(read, nextOp);
                return fuseReadsLaunchTransformDPP(fused, iOps...);
            } else {
                return buildOperationPipeline(read, nextOp, iOps...);
            }
        } else {
            static_assert(isWriteType<Next>, "Last IOp must be WriteType");
            return buildOperationPipeline(read, nextOp);
        }
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

    // Implementation of CPUJITCompiler methods
    inline CPUJITCompiler::CPUJITCompiler() = default;

    inline llvm::Error CPUJITCompiler::initialize() {
        // Initialize LLVM native target
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();

        // Create the JIT instance
        auto jitExpected = llvm::orc::LLJITBuilder().create();
        if (!jitExpected) {
            return jitExpected.takeError();
        }
        
        jit = std::move(*jitExpected);
        return llvm::Error::success();
    }

    inline std::function<std::vector<JIT_Operation_pp>(const std::vector<JIT_Operation_pp>&)> 
    CPUJITCompiler::compileRuntimeFusion(const std::vector<JIT_Operation_pp>& operations) {
        
        // Generate the IR module
        auto module = generateRuntimeFusionIR(operations);
        if (!module) {
            return nullptr;
        }

        // Verify the module
        std::string errorMsg;
        llvm::raw_string_ostream errorStream(errorMsg);
        if (llvm::verifyModule(*module, &errorStream)) {
            llvm::errs() << "Module verification failed: " << errorMsg << "\n";
            return nullptr;
        }

        // Add the module to the JIT
        auto tsm = llvm::orc::ThreadSafeModule(std::move(module), 
                                               llvm::orc::ThreadSafeContext(std::make_unique<llvm::LLVMContext>()));
        
        if (auto err = jit->addIRModule(std::move(tsm))) {
            llvm::errs() << "Failed to add module to JIT: " << err << "\n";
            return nullptr;
        }

        // Get the compiled function
        auto symbol = jit->lookup("runtimeFusionFunction");
        if (!symbol) {
            llvm::errs() << "Failed to find compiled function: " << symbol.takeError() << "\n";
            return nullptr;
        }

        // Cast to the correct function type
        using FunctionType = std::vector<JIT_Operation_pp>(*)(const std::vector<JIT_Operation_pp>&);
        auto funcPtr = reinterpret_cast<FunctionType>(symbol->getValue());

        return [funcPtr](const std::vector<JIT_Operation_pp>& ops) -> std::vector<JIT_Operation_pp> {
            return funcPtr(ops);
        };
    }

    inline std::unique_ptr<llvm::Module> CPUJITCompiler::generateRuntimeFusionIR(const std::vector<JIT_Operation_pp>& operations) {
        auto module = std::make_unique<llvm::Module>("RuntimeFusionModule", context);
        
        // Create function type: vector<JIT_Operation_pp> func(const vector<JIT_Operation_pp>&)
        llvm::Type* voidPtrType = llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0);
        llvm::Type* vectorType = voidPtrType; // Simplified for now - represents vector as void*
        
        llvm::FunctionType* funcType = llvm::FunctionType::get(
            vectorType,                    // Return type
            {vectorType},                  // Parameters  
            false                          // Not variadic
        );

        // Create the function
        llvm::Function* func = llvm::Function::Create(
            funcType,
            llvm::Function::ExternalLinkage,
            "runtimeFusionFunction",
            module.get()
        );

        // Create basic block
        llvm::BasicBlock* entry = llvm::BasicBlock::Create(context, "entry", func);
        llvm::IRBuilder<> builder(entry);

        // For now, create a simple implementation that returns the input
        // This is a placeholder - the actual implementation would:
        // 1. Extract operation data from the input vector
        // 2. Cast each void* to the appropriate type based on opType
        // 3. Call the templated fuseBack function
        // 4. Return the result as a new vector

        llvm::Value* inputParam = func->arg_begin();
        builder.CreateRet(inputParam);

        return module;
    }

    inline llvm::Value* CPUJITCompiler::generateCastingCode(llvm::IRBuilder<>& builder, llvm::Value* opDataPtr, const std::string& typeName) {
        // This would generate casting code based on the type name
        // For now, return the pointer as-is
        return opDataPtr;
    }

#endif // ENABLE_CPU_JIT

    inline std::vector<JIT_Operation_pp> fuseReadBackOperationsJIT(const std::vector<JIT_Operation_pp>& operations) {
#ifdef ENABLE_CPU_JIT
        // For now, disable the actual JIT compilation and just return the operations
        // The JIT functionality is a placeholder and needs more development
        std::cout << "JIT fusion called (JIT compilation currently disabled for safety)" << std::endl;
#endif

        // Return input operations (no fusion performed yet)
        return operations;
    }

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