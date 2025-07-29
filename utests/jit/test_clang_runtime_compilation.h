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
//__NVRTC__

#include "clang/Interpreter/Interpreter.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <iostream>
#include <string>
#include <vector>

int launch() {
    // Initialize LLVM for JIT compilation
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // C++ source code to compile at runtime using clang::Interpreter
    std::string code = "int test() { return 23; }";
    std::cout << "C++ source code string: " << code << std::endl;

    // Prepare compiler arguments
    std::vector<const char *> args;

    std::cout << "Creating compiler instance..." << std::endl;

    // Create compiler instance using IncrementalCompilerBuilder
    clang::IncrementalCompilerBuilder builder;
    builder.SetCompilerArgs(args);
    
    auto compilerInstanceExpected = builder.CreateCpp();
    if (!compilerInstanceExpected) {
        llvm::errs() << "Failed to create compiler instance: " 
                     << llvm::toString(compilerInstanceExpected.takeError()) << "\n";
        return 1;
    }

    std::cout << "Compiler instance created successfully" << std::endl;

    // Create interpreter from compiler instance
    auto interpreterExpected = clang::Interpreter::create(std::move(*compilerInstanceExpected));
    if (!interpreterExpected) {
        llvm::errs() << "Failed to create Clang interpreter: " 
                     << llvm::toString(interpreterExpected.takeError()) << "\n";
        return 1;
    }

    auto& interpreter = *interpreterExpected;
    std::cout << "Clang interpreter created successfully" << std::endl;

    // Compile the C++ function using extern "C" for C linkage
    std::string fullCode = "extern \"C\" " + code;
    std::cout << "Compiling: " << fullCode << std::endl;
    
    if (auto err = interpreter->ParseAndExecute(fullCode)) {
        llvm::errs() << "Failed to compile function: " 
                     << llvm::toString(std::move(err)) << "\n";
        return 1;
    }

    std::cout << "Function compiled successfully" << std::endl;

    // Lookup the symbol
    auto symAddr = interpreter->getSymbolAddress("test");
    if (!symAddr) {
        llvm::errs() << "Failed to find symbol 'test': " 
                     << llvm::toString(symAddr.takeError()) << "\n";
        return 1;
    }

    std::cout << "Symbol 'test' found successfully" << std::endl;

    // Cast to function pointer
    using TestFunc = int (*)();
    auto testFunc = reinterpret_cast<TestFunc>(symAddr->getValue());

    // Call the function
    int result = testFunc();

    std::cout << "Executed JIT compiled function from C++ string" << std::endl;

    // Verify the result
    if (result == 23) {
        std::cout << "SUCCESS: Function returned expected value " << result << std::endl;
        std::cout << "Successfully demonstrated runtime compilation of C++ code: " << code << std::endl;
        return 0;
    } else {
        std::cerr << "FAILURE: Function returned " << result << " instead of 23" << std::endl;
        return 1;
    }
}

#endif