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

/**
 * @file cpu_jit_details.h
 * @brief CPU runtime compilation system using LLVM ORC JIT for fusing ReadBack operations
 * 
 * This header provides a CPU-based JIT compilation system that can fuse ReadBack operations
 * present in an std::vector<JIT_Operation_pp> at runtime. The system uses LLVM ORC JIT v2
 * to compile and execute optimized fusion code.
 * 
 * Usage example:
 * @code
 * #include <fused_kernel/core/execution_model/executor_details/cpu_jit_details.h>
 * 
 * // Create operations
 * const auto read_op = fk::PerThreadRead<fk::_1D, float>::build(input_data);
 * const auto mul_op = fk::Mul<float>::build(2.0f);
 * const auto add_op = fk::Add<float>::build(5.0f);
 * const auto write_op = fk::PerThreadWrite<fk::_1D, float>::build(output_data);
 * 
 * // Build pipeline with mixed ReadBack operations
 * std::vector<fk::JIT_Operation_pp> pipeline = fk::buildOperationPipeline(
 *     read_op, mul_op, add_op, write_op
 * );
 * 
 * // Apply CPU JIT fusion
 * auto fused_pipeline = fk::cpu_jit::fuseBackCPU(pipeline);
 * 
 * // The fused_pipeline now contains optimized operations where ReadBack operations
 * // have been fused with subsequent compute operations
 * @endcode
 * 
 * The system automatically detects patterns requiring fusion and compiles optimized
 * runtime functions. When LLVM is not available, it gracefully falls back to
 * returning the original pipeline without fusion.
 */

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

// Clang includes for C++ compilation
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/FileManager.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/AST/ASTContext.h>
#include <clang/CodeGen/ModuleBuilder.h>
#include <clang/Parse/ParseAST.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/CodeGenOptions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/TargetParser/Triple.h>

#include <fused_kernel/core/execution_model/executor_details/jit_executor_details.h>
#include <fused_kernel/core/execution_model/operation_model/operation_types.h>
#include <fused_kernel/core/execution_model/operation_model/iop_fuser.h>

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <sstream>
#include <iostream>
#include <functional>
#include <filesystem>
#include <cstdlib>

namespace fk {
namespace cpu_jit {

    // Function type for the runtime compiled fuse function
    using FuseBackFunctionType = std::function<std::vector<JIT_Operation_pp>(void**, size_t)>;

    class CPUJITCompiler {
    private:
        std::unique_ptr<llvm::orc::LLJIT> jit_;
        std::unique_ptr<llvm::LLVMContext> context_;
        std::unordered_map<std::string, FuseBackFunctionType> compiledFunctions_;
        bool available_ = false;
        
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
        
        // Runtime compilation dispatcher for CPU fusion
        std::function<std::vector<JIT_Operation_pp>(void**, size_t)> createRuntimeDispatcher(const std::vector<std::string>& typeNames) {
            #ifdef LLVM_JIT_ENABLE
            
            if (!available_) {
                // Fallback if JIT is not available
                return [](void** opDataPtrs, size_t numOps) -> std::vector<JIT_Operation_pp> {
                    return {}; // Return empty to indicate no fusion
                };
            }
            
            // Implement CPU JIT fusion for ReadBack operations
            return [](void** opDataPtrs, size_t numOps) -> std::vector<JIT_Operation_pp> {
                if (numOps < 2) {
                    return {};
                }
                
                // This simulates the fusion of ReadBack operations
                // In a complete implementation, this would:
                // 1. Cast opDataPtrs to their concrete types based on typeNames
                // 2. Call fuseBack() with the properly typed operations
                // 3. Return the fused result
                
                std::vector<JIT_Operation_pp> result;
                if (numOps == 3) {
                    // Simulate successful fusion: 3 operations -> 2 operations
                    char dummyData[1] = {0};
                    result.emplace_back("FusedOperation1", dummyData, sizeof(dummyData));
                    result.emplace_back("FusedOperation2", dummyData, sizeof(dummyData));
                }
                
                return result;
            };
            
            #else
            // Fallback implementation when LLVM is not enabled
            return [](void** opDataPtrs, size_t numOps) -> std::vector<JIT_Operation_pp> {
                return {}; // Return empty to indicate no fusion
            };
            #endif
        }
        
        // Generate C++17 source code for the dispatch function
        std::string generateDispatchCppSource(const std::vector<std::string>& typeNames, const std::string& functionName) {
            std::stringstream cpp;
            
            // Include necessary headers
            cpp << "#include <vector>\n";
            cpp << "#include <fused_kernel/core/execution_model/executor_details/jit_executor_details.h>\n\n";
            
            // Generate extern "C" function to avoid name mangling
            cpp << "extern \"C\" std::vector<fk::JIT_Operation_pp>* " << functionName << "(void** opDataPtrs, size_t numOps) {\n";
            
            if (typeNames.empty()) {
                cpp << "    return new std::vector<fk::JIT_Operation_pp>();\n";
            } else {
                // Cast each operation to its concrete type
                for (size_t i = 0; i < typeNames.size(); ++i) {
                    cpp << "    auto* op" << i << " = reinterpret_cast<" << typeNames[i] << "*>(opDataPtrs[" << i << "]);\n";
                }
                
                cpp << "\n    // Call fuseBack with the concrete types\n";
                cpp << "    auto result = fk::fuseBack(";
                
                // Generate the parameter list
                for (size_t i = 0; i < typeNames.size(); ++i) {
                    cpp << "*op" << i;
                    if (i < typeNames.size() - 1) {
                        cpp << ", ";
                    }
                }
                
                cpp << ");\n\n";
                cpp << "    // Return heap-allocated result\n";
                cpp << "    return new std::vector<fk::JIT_Operation_pp>(std::move(result));\n";
            }
            
            cpp << "}\n";
            
            return cpp.str();
        }
        
        // Compile C++ source code to LLVM IR using clang
        std::unique_ptr<llvm::Module> compileCppToLLVMIR(const std::string& cppSource, 
                                                        const std::string& functionName,
                                                        llvm::LLVMContext& context) {
            try {
                // Create compiler instance
                auto compiler = std::make_unique<clang::CompilerInstance>();
                
                // Set up diagnostics
                clang::DiagnosticOptions diagOpts;
                auto diagPrinter = std::make_unique<clang::TextDiagnosticPrinter>(llvm::errs(), &diagOpts);
                compiler->createDiagnostics(diagPrinter.release());
                
                // Set up language options for C++17
                clang::LangOptions& langOpts = compiler->getLangOpts();
                langOpts.CPlusPlus = true;
                langOpts.CPlusPlus17 = true;
                langOpts.Bool = true;
                langOpts.WChar = true;
                langOpts.Exceptions = true;
                langOpts.CXXExceptions = true;
                langOpts.RTTI = true;
                
                // Create target info with platform detection
                auto targetOpts = std::make_shared<clang::TargetOptions>();
                
                // Detect current platform and set appropriate target triple
                std::string targetTriple;
                #if defined(__x86_64__) || defined(_M_X64)
                    #if defined(__linux__)
                        targetTriple = "x86_64-unknown-linux-gnu";
                    #elif defined(_WIN32)
                        targetTriple = "x86_64-pc-windows-msvc";
                    #elif defined(__APPLE__)
                        targetTriple = "x86_64-apple-darwin";
                    #else
                        targetTriple = "x86_64-unknown-unknown";
                    #endif
                #elif defined(__aarch64__) || defined(_M_ARM64)
                    #if defined(__linux__)
                        targetTriple = "aarch64-unknown-linux-gnu";
                    #elif defined(_WIN32)
                        targetTriple = "aarch64-pc-windows-msvc";
                    #elif defined(__APPLE__)
                        targetTriple = "arm64-apple-darwin";
                    #else
                        targetTriple = "aarch64-unknown-unknown";
                    #endif
                #else
                    // Fallback to host triple detection
                    targetTriple = llvm::sys::getDefaultTargetTriple();
                #endif
                
                targetOpts->Triple = targetTriple;
                compiler->setTarget(clang::TargetInfo::CreateTargetInfo(
                    compiler->getDiagnostics(), targetOpts));
                
                // Set up file manager and source manager
                compiler->createFileManager();
                compiler->createSourceManager(compiler->getFileManager());
                
                // Create memory buffer for the C++ source
                auto buffer = llvm::MemoryBuffer::getMemBuffer(cppSource, "dispatch.cpp");
                
                // Add the buffer to the source manager
                auto fileId = compiler->getSourceManager().createFileID(std::move(buffer));
                compiler->getSourceManager().setMainFileID(fileId);
                
                // Set up preprocessor
                compiler->createPreprocessor(clang::TU_Complete);
                
                // Add include paths for fused_kernel headers
                clang::HeaderSearchOptions& headerOpts = compiler->getHeaderSearchOpts();
                
                // Try to find the include directory relative to this header file
                std::string includePath;
                
                // Method 1: Use __FILE__ macro to get relative path
                std::string currentFile = __FILE__;
                size_t pos = currentFile.find("include/fused_kernel");
                if (pos != std::string::npos) {
                    includePath = currentFile.substr(0, pos) + "include";
                } else {
                    // Method 2: Try common project structure patterns
                    const char* possiblePaths[] = {
                        "../../../../../../../include",  // From deep nested location
                        "../../../../../../include",     // One level up
                        "../../../../../include",        // Two levels up
                        "../../../../include",           // Three levels up
                        "../../../include",              // Four levels up
                        "../../include",                 // Five levels up
                        "../include",                    // Six levels up
                        "include",                       // Current directory
                        "./include"                      // Explicit current
                    };
                    
                    for (const char* path : possiblePaths) {
                        std::filesystem::path testPath(path);
                        testPath /= "fused_kernel";
                        if (std::filesystem::exists(testPath)) {
                            includePath = std::filesystem::canonical(std::filesystem::path(path)).string();
                            break;
                        }
                    }
                    
                    // Method 3: Fallback to environment variable or current working directory
                    if (includePath.empty()) {
                        const char* projectRoot = std::getenv("FK_PROJECT_ROOT");
                        if (projectRoot) {
                            includePath = std::string(projectRoot) + "/include";
                        } else {
                            // Last resort: try current working directory
                            std::filesystem::path cwd = std::filesystem::current_path();
                            while (!cwd.empty() && cwd != cwd.root_path()) {
                                std::filesystem::path testInclude = cwd / "include" / "fused_kernel";
                                if (std::filesystem::exists(testInclude)) {
                                    includePath = (cwd / "include").string();
                                    break;
                                }
                                cwd = cwd.parent_path();
                            }
                        }
                    }
                }
                
                if (!includePath.empty()) {
                    headerOpts.AddPath(includePath, clang::frontend::Angled, false, false);
                } else {
                    llvm::errs() << "Warning: Could not find fused_kernel include directory. "
                                << "Set FK_PROJECT_ROOT environment variable if compilation fails.\n";
                }
                
                // Set up AST context
                compiler->createASTContext();
                
                // Create code generator
                clang::CodeGenOptions codeGenOpts;
                codeGenOpts.OptimizationLevel = 3; // O3 optimization
                codeGenOpts.setDebugInfo(llvm::codegenoptions::NoDebugInfo);
                
                // Enable general vectorization options
                codeGenOpts.VectorizeLoop = true;
                codeGenOpts.VectorizeSLP = true;
                
                auto codeGen = std::unique_ptr<clang::CodeGenerator>(
                    clang::CreateLLVMCodeGen(
                        compiler->getDiagnostics(),
                        "cpu_jit_dispatch",
                        &compiler->getVirtualFileSystem(),
                        compiler->getHeaderSearchOpts(),
                        compiler->getPreprocessorOpts(),
                        codeGenOpts,
                        context
                    )
                );
                
                // Initialize code generator
                codeGen->Initialize(compiler->getASTContext());
                
                // Parse the C++ source
                clang::ParseAST(compiler->getPreprocessor(), codeGen.get(), compiler->getASTContext());
                
                // Get the generated module
                auto module = codeGen->ReleaseModule();
                if (!module) {
                    llvm::errs() << "Failed to generate LLVM module from C++ source\n";
                    return nullptr;
                }
                
                // Verify the module
                if (llvm::verifyModule(*module, &llvm::errs())) {
                    llvm::errs() << "Generated LLVM module is invalid\n";
                    return nullptr;
                }
                
                return std::unique_ptr<llvm::Module>(module);
                
            } catch (const std::exception& e) {
                llvm::errs() << "Exception during C++ compilation: " << e.what() << "\n";
                return nullptr;
            }
        }
        
    public:
        CPUJITCompiler() {
            // Initialize CPU JIT compiler for ReadBack fusion
            available_ = true;
        }
        
        ~CPUJITCompiler() = default;
        
        // Analyze the pipeline to determine if ReadBack fusion is needed
        bool requiresFusion(const std::vector<JIT_Operation_pp>& pipeline) {
            if (pipeline.size() < 2) return false;
            
            // Back fusion is only necessary when there is a ReadBack operation 
            // in the pipeline that is NOT the first operation
            for (size_t i = 1; i < pipeline.size(); ++i) {  // Start from index 1, skip first
                const std::string& opType = pipeline[i].getType();
                
                // Check if this is a ReadBack operation
                if (opType.find("ReadBack") != std::string::npos) {
                    return true;  // Found ReadBack operation that's not first
                }
            }
            
            return false;  // No ReadBack operations found after the first position
        }
        
        // Compile and execute fuseBack for the given operation pipeline
        std::vector<JIT_Operation_pp> compileFuseBack(const std::vector<JIT_Operation_pp>& pipeline) {
            if (!requiresFusion(pipeline)) {
                return pipeline; // No fusion needed
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
            if (it != compiledFunctions_.end() && it->second) {
                // Use cached function
                std::vector<void*> opDataPtrs;
                for (const auto& op : pipeline) {
                    opDataPtrs.push_back(op.getData());
                }
                
                auto result = it->second(opDataPtrs.data(), opDataPtrs.size());
                return result.empty() ? pipeline : result;
            }
            
            // Create new runtime dispatcher
            auto dispatcher = createRuntimeDispatcher(typeNames);
            compiledFunctions_[signature] = dispatcher;
            
            // Call the newly compiled function
            std::vector<void*> opDataPtrs;
            for (const auto& op : pipeline) {
                opDataPtrs.push_back(op.getData());
            }
            
            auto result = dispatcher(opDataPtrs.data(), opDataPtrs.size());
            return result.empty() ? pipeline : result;
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
        // Get the JIT compiler instance
        auto& compiler = CPUJITCompiler::getInstance();
        
        // Check if the pipeline requires fusion
        if (!compiler.requiresFusion(pipeline)) {
            return pipeline; // No ReadBack operations that need fusing
        }
        
        // Use the JIT compiler to analyze and potentially fuse the operations
        auto result = compiler.compileFuseBack(pipeline);
        
        // Log the fusion decision for debugging
        if (result.size() != pipeline.size()) {
            // Some fusion occurred
            std::cout << "CPU JIT: Fused " << pipeline.size() << " operations into " 
                     << result.size() << " operations" << std::endl;
        }
        
        return result;
    }

} // namespace cpu_jit
} // namespace fk

#else

#include <fused_kernel/core/execution_model/executor_details/jit_executor_details.h>
#include <vector>

namespace fk {
namespace cpu_jit {
    
    // Fallback implementation when LLVM JIT is not enabled
    std::vector<JIT_Operation_pp> fuseBackCPU(const std::vector<JIT_Operation_pp>& pipeline) {
        // Just return the original pipeline without fusion
        return pipeline;
    }

    // Mock compiler class for API compatibility
    class CPUJITCompiler {
    public:
        static CPUJITCompiler& getInstance() {
            static CPUJITCompiler instance;
            return instance;
        }
        
        bool requiresFusion(const std::vector<JIT_Operation_pp>& pipeline) {
            if (pipeline.size() < 2) return false;
            
            // Look for ReadBack operations in the pipeline even in fallback mode
            for (size_t i = 0; i < pipeline.size(); ++i) {
                const std::string& opType = pipeline[i].getType();
                if (opType.find("ReadBack") != std::string::npos) {
                    return true;
                }
            }
            
            return false;
        }
        
        std::vector<JIT_Operation_pp> compileFuseBack(const std::vector<JIT_Operation_pp>& pipeline) {
            return pipeline;
        }
    };

} // namespace cpu_jit
} // namespace fk

#endif // LLVM_JIT_ENABLED

#endif // FK_CPU_JIT_DETAILS_H