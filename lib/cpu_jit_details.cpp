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

#ifdef ENABLE_CPU_JIT
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#endif

#include <iostream>

namespace fk {

#ifdef ENABLE_CPU_JIT

    CPUJITCompiler::CPUJITCompiler() = default;

    llvm::Error CPUJITCompiler::initialize() {
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

    std::function<std::vector<JIT_Operation_pp>(const std::vector<JIT_Operation_pp>&)> 
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

    std::unique_ptr<llvm::Module> CPUJITCompiler::generateRuntimeFusionIR(const std::vector<JIT_Operation_pp>& operations) {
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

    llvm::Value* CPUJITCompiler::generateCastingCode(llvm::IRBuilder<>& builder, llvm::Value* opDataPtr, const std::string& typeName) {
        // This would generate casting code based on the type name
        // For now, return the pointer as-is
        return opDataPtr;
    }

#endif // ENABLE_CPU_JIT

    std::vector<JIT_Operation_pp> fuseReadBackOperationsJIT(const std::vector<JIT_Operation_pp>& operations) {
#ifdef ENABLE_CPU_JIT
        // For now, disable the actual JIT compilation and just return the operations
        // The JIT functionality is a placeholder and needs more development
        std::cout << "JIT fusion called (JIT compilation currently disabled for safety)" << std::endl;
#endif

        // Return input operations (no fusion performed yet)
        return operations;
    }

} // namespace fk