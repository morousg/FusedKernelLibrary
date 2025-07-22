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

#ifndef FK_TEST_CLANG_RUNTIME_COMPILATION_H
#define FK_TEST_CLANG_RUNTIME_COMPILATION_H

//__ONLY_CPU__
//__LLVM_JIT__

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <iostream>

int launch() {
    // Initialize LLVM for JIT compilation
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    // Create LLVM context and module
    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("test_module", *context);

    // Create the function signature: int test()
    llvm::FunctionType* funcType = llvm::FunctionType::get(
        llvm::Type::getInt32Ty(*context), // return type: int
        false  // not variadic
    );

    // Create the function
    llvm::Function* testFunc = llvm::Function::Create(
        funcType, 
        llvm::Function::ExternalLinkage, 
        "test", 
        module.get()
    );

    // Create a basic block and IRBuilder
    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create(*context, "entry", testFunc);
    llvm::IRBuilder<> builder(entryBlock);

    // Generate the function body: return 23;
    llvm::Value* returnValue = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context), 23);
    builder.CreateRet(returnValue);

    // Verify the module
    std::string error;
    llvm::raw_string_ostream errorStream(error);
    if (llvm::verifyModule(*module, &errorStream)) {
        std::cerr << "Module verification failed: " << error << std::endl;
        return 1;
    }

    // Create execution engine for JIT compilation
    std::string engineError;
    llvm::ExecutionEngine* executionEngine = llvm::EngineBuilder(std::move(module))
        .setErrorStr(&engineError)
        .setEngineKind(llvm::EngineKind::JIT)
        .create();

    if (!executionEngine) {
        std::cerr << "Failed to create execution engine: " << engineError << std::endl;
        return 1;
    }

    // Get pointer to the compiled function
    uint64_t funcAddr = executionEngine->getFunctionAddress("test");
    if (!funcAddr) {
        std::cerr << "Failed to get function address" << std::endl;
        delete executionEngine;
        return 1;
    }

    // Cast the function address to a callable function pointer
    typedef int (*TestFuncPtr)();
    TestFuncPtr testFuncPtr = reinterpret_cast<TestFuncPtr>(funcAddr);

    // Execute the compiled function
    int result = testFuncPtr();

    // Clean up
    delete executionEngine;

    // Verify the result
    if (result == 23) {
        std::cout << "SUCCESS: Function returned expected value 23" << std::endl;
        return 0;
    } else {
        std::cerr << "FAILURE: Function returned " << result << " instead of 23" << std::endl;
        return 1;
    }
}

#endif