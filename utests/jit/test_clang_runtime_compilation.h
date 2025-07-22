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

#include <clang/Frontend/CompilerInstance.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/LangStandard.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Lex/HeaderSearch.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Frontend/FrontendActions.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>
#include <memory>
#include <iostream>
#include <string>

int launch() {
    // Initialize LLVM for JIT compilation
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    // C++ source code to compile at runtime
    std::string code = "int test() { return 23; }";
    std::cout << "Compiling C++ code: " << code << std::endl;
    
    try {
        // Create diagnostic options
        auto diagOpts = llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions>(new clang::DiagnosticOptions());
        clang::TextDiagnosticPrinter diagPrinter(llvm::errs(), diagOpts.get());
        clang::DiagnosticsEngine diags(
            llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(new clang::DiagnosticIDs()),
            diagOpts,
            &diagPrinter,
            false
        );

        std::cout << "Created diagnostic engine" << std::endl;

        // Create compiler instance
        auto compilerInstance = std::make_unique<clang::CompilerInstance>();
        compilerInstance->setDiagnostics(&diags);

        std::cout << "Created compiler instance" << std::endl;

        // Create compiler invocation - this sets up defaults
        auto invocation = std::make_shared<clang::CompilerInvocation>();
        
        // Set the target triple
        invocation->getTargetOpts().Triple = llvm::sys::getDefaultTargetTriple();
        
        // Set language options for C++
        clang::LangOptions& langOpts = invocation->getLangOpts();
        langOpts.CPlusPlus = true;
        langOpts.CPlusPlus11 = true;
        langOpts.Bool = true;
        langOpts.WChar = true;
        
        // Set frontend options
        invocation->getFrontendOpts().ProgramAction = clang::frontend::EmitLLVMOnly;
        
        compilerInstance->setInvocation(invocation);

        std::cout << "Created compiler invocation" << std::endl;

        // Set target info based on invocation
        compilerInstance->setTarget(clang::TargetInfo::CreateTargetInfo(
            compilerInstance->getDiagnostics(), compilerInstance->getInvocation().TargetOpts));

        if (!compilerInstance->hasTarget()) {
            std::cerr << "Failed to create target info" << std::endl;
            return 1;
        }

        std::cout << "Created target info" << std::endl;

        // Set file manager
        compilerInstance->createFileManager();
        
        // Set source manager
        compilerInstance->createSourceManager(compilerInstance->getFileManager());

        std::cout << "Created source manager" << std::endl;

        // Create in-memory file for the C++ source code
        llvm::StringRef sourceCode(code);
        auto memBuffer = llvm::MemoryBuffer::getMemBufferCopy(sourceCode, "input.cpp");
        llvm::MemoryBufferRef memBufferRef = *memBuffer;
        auto fileID = compilerInstance->getSourceManager().createFileID(std::move(memBuffer));
        compilerInstance->getSourceManager().setMainFileID(fileID);

        std::cout << "Created source file" << std::endl;

        // Set up input file in the invocation
        clang::InputKind inputKind(clang::Language::CXX);
        clang::FrontendInputFile inputFile(memBufferRef, inputKind);
        invocation->getFrontendOpts().Inputs.clear();
        invocation->getFrontendOpts().Inputs.push_back(inputFile);

        // Create preprocessor
        compilerInstance->createPreprocessor(clang::TU_Complete);

        std::cout << "Created preprocessor" << std::endl;

        // Set AST context
        compilerInstance->createASTContext();

        std::cout << "Created AST context" << std::endl;

        // Create CodeGen action to compile to LLVM IR
        auto action = std::make_unique<clang::EmitLLVMOnlyAction>();
        
        std::cout << "Starting compilation..." << std::endl;

        // Execute compilation
        if (!action->BeginSourceFile(*compilerInstance, inputFile)) {
            std::cerr << "Failed to begin source file compilation" << std::endl;
            return 1;
        }

        std::cout << "BeginSourceFile succeeded" << std::endl;

        if (!action->Execute()) {
            std::cerr << "Failed to execute compilation" << std::endl;
            return 1;
        }

        std::cout << "Execute succeeded" << std::endl;

        action->EndSourceFile();

        std::cout << "EndSourceFile completed" << std::endl;

        // Get the compiled LLVM module
        std::unique_ptr<llvm::Module> module = action->takeModule();
        if (!module) {
            std::cerr << "Failed to get compiled module" << std::endl;
            return 1;
        }

        std::cout << "Got compiled module" << std::endl;

        // Verify the module
        std::string error;
        llvm::raw_string_ostream errorStream(error);
        if (llvm::verifyModule(*module, &errorStream)) {
            std::cerr << "Module verification failed: " << error << std::endl;
            return 1;
        }

        std::cout << "Module verified successfully" << std::endl;

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

        std::cout << "Created execution engine" << std::endl;

        // Get pointer to the compiled function
        uint64_t funcAddr = executionEngine->getFunctionAddress("test");
        if (!funcAddr) {
            std::cerr << "Failed to get function address" << std::endl;
            delete executionEngine;
            return 1;
        }

        std::cout << "Got function address" << std::endl;

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
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
        return 1;
    }
}

#endif