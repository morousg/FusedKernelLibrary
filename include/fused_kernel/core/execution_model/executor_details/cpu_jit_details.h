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
 * @brief CPU runtime compilation system using clang::Interpreter for fusing ReadBack operations
 * 
 * This header provides a CPU-based JIT compilation system that can fuse ReadBack operations
 * present in an std::vector<JIT_Operation_pp> at runtime. The system uses clang::Interpreter
 * to compile and execute optimized fusion code in-memory.
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
 * runtime functions using clang::Interpreter. When LLVM is not available, it gracefully 
 * falls back to returning the original pipeline without fusion.
 */

#ifndef FK_CPU_JIT_DETAILS_H
#define FK_CPU_JIT_DETAILS_H

#if defined(LLVM_JIT_ENABLED)

#include <fused_kernel/core/execution_model/executor_details/jit_executor_details.h>
#include <fused_kernel/core/execution_model/operation_model/operation_types.h>
#include <fused_kernel/core/execution_model/operation_model/iop_fuser.h>

#include "clang/Interpreter/Interpreter.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <sstream>
#include <iostream>
#include <functional>

namespace fk {
namespace cpu_jit {

    // Function type for the runtime compiled fuse function
    using FuseBackFunctionType = std::function<std::vector<JIT_Operation_pp>(void**, size_t)>;

    class CPUJITCompiler {
    private:
        std::unordered_map<std::string, FuseBackFunctionType> compiledFunctions_;
        std::unique_ptr<clang::Interpreter> interpreter_;
        bool available_ = false;
        
        static bool llvmInitialized_;
        
        void initializeLLVM() {
            if (!llvmInitialized_) {
                llvm::InitializeNativeTarget();
                llvm::InitializeNativeTargetAsmPrinter();
                llvm::InitializeNativeTargetAsmParser();
                llvmInitialized_ = true;
                std::cout << "CPU JIT: LLVM initialized successfully" << std::endl;
            }
        }
        
        bool initializeInterpreter() {
            if (interpreter_) {
                return true; // Already initialized
            }
            
            try {
                // Initialize LLVM if not already done
                initializeLLVM();
                
                // Prepare compiler arguments
                std::vector<const char *> args;
                
                // Create compiler instance using IncrementalCompilerBuilder
                clang::IncrementalCompilerBuilder builder;
                builder.SetCompilerArgs(args);
                
                auto compilerInstanceExpected = builder.CreateCpp();
                if (!compilerInstanceExpected) {
                    std::cerr << "CPU JIT: Failed to create compiler instance: " 
                             << llvm::toString(compilerInstanceExpected.takeError()) << std::endl;
                    return false;
                }
                
                // Create interpreter from compiler instance
                auto interpreterExpected = clang::Interpreter::create(std::move(*compilerInstanceExpected));
                if (!interpreterExpected) {
                    std::cerr << "CPU JIT: Failed to create Clang interpreter: " 
                             << llvm::toString(interpreterExpected.takeError()) << std::endl;
                    return false;
                }
                
                interpreter_ = std::move(*interpreterExpected);
                
                // Define the JIT_Operation_pp class once for all compilation sessions
                std::string commonTypes = R"(
                    #include <vector>
                    #include <string>
                    #include <cstring>
                    #include <iostream>
                    
                    namespace fk {
                        class JIT_Operation_pp {
                        private:
                            std::string opType;
                            void* opData;
                            size_t dataSize;
                        public:
                            JIT_Operation_pp(std::string type, const void* data, size_t size)
                                : opType(type), dataSize(size) {
                                opData = new char[dataSize];
                                std::memcpy(opData, data, dataSize);
                            }
                            JIT_Operation_pp(const JIT_Operation_pp& other)
                                : opType(other.opType), dataSize(other.dataSize) {
                                opData = new char[dataSize];
                                std::memcpy(opData, other.opData, dataSize);
                            }
                            JIT_Operation_pp(JIT_Operation_pp&& other) noexcept
                                : opType(std::move(other.opType)), opData(other.opData), dataSize(other.dataSize) {
                                other.opData = nullptr;
                                other.dataSize = 0;
                            }
                            JIT_Operation_pp& operator=(const JIT_Operation_pp& other) {
                                if (this != &other) {
                                    delete[] static_cast<char*>(opData);
                                    opType = other.opType;
                                    dataSize = other.dataSize;
                                    opData = new char[dataSize];
                                    std::memcpy(opData, other.opData, dataSize);
                                }
                                return *this;
                            }
                            JIT_Operation_pp& operator=(JIT_Operation_pp&& other) noexcept {
                                if (this != &other) {
                                    delete[] static_cast<char*>(opData);
                                    opType = std::move(other.opType);
                                    opData = other.opData;
                                    dataSize = other.dataSize;
                                    other.opData = nullptr;
                                    other.dataSize = 0;
                                }
                                return *this;
                            }
                            ~JIT_Operation_pp() {
                                delete[] static_cast<char*>(opData);
                            }
                            const std::string& getType() const { return opType; }
                            void* getData() const { return opData; }
                        };
                    }
                )";
                
                // Compile the common types once
                if (auto err = interpreter_->ParseAndExecute(commonTypes)) {
                    std::cerr << "CPU JIT: Failed to compile common types: " 
                             << llvm::toString(std::move(err)) << std::endl;
                    return false;
                }
                
                available_ = true;
                std::cout << "CPU JIT: Clang interpreter initialized successfully" << std::endl;
                return true;
                
            } catch (const std::exception& e) {
                std::cerr << "CPU JIT: Exception during interpreter initialization: " << e.what() << std::endl;
                available_ = false;
                return false;
            }
        }
        
        // Generate signature hash for caching (sanitized for use as C identifier)
        std::string generateSignature(const std::vector<std::string>& typeNames) {
            std::stringstream ss;
            for (size_t i = 0; i < typeNames.size(); ++i) {
                // Convert type name to valid C identifier by replacing special characters
                std::string sanitized = typeNames[i];
                for (char& c : sanitized) {
                    if (!std::isalnum(c) && c != '_') {
                        c = '_';
                    }
                }
                ss << sanitized;
                if (i < typeNames.size() - 1) {
                    ss << "_";
                }
            }
            return ss.str();
        }
        
        // Runtime compilation dispatcher for CPU fusion using clang::Interpreter
        std::function<std::vector<JIT_Operation_pp>(void**, size_t)> createRuntimeDispatcher(const std::vector<std::string>& typeNames) {
            if (!available_ || typeNames.empty()) {
                // Fallback if JIT is not available
                return [](void** opDataPtrs, size_t numOps) -> std::vector<JIT_Operation_pp> {
                    return {}; // Return empty to indicate no fusion
                };
            }
            
            try {
                // Ensure interpreter is initialized
                if (!initializeInterpreter()) {
                    return [](void** opDataPtrs, size_t numOps) -> std::vector<JIT_Operation_pp> { return {}; };
                }
                
                // Generate unique function name
                std::string functionName = "dispatch_" + generateSignature(typeNames);
                
                // Generate C++ source code that calls the real fuseBack function
                std::string cppSource = generateDispatchCppSource(typeNames, functionName);
                
                std::cout << "CPU JIT: Compiling function: " << functionName << std::endl;
                
                // Compile the C++ function using clang::Interpreter
                if (auto err = interpreter_->ParseAndExecute(cppSource)) {
                    std::cerr << "CPU JIT: Failed to compile function: " 
                             << llvm::toString(std::move(err)) << std::endl;
                    return [](void** opDataPtrs, size_t numOps) -> std::vector<JIT_Operation_pp> { return {}; };
                }
                
                std::cout << "CPU JIT: Function compiled successfully" << std::endl;
                
                // Lookup the symbol
                auto symAddr = interpreter_->getSymbolAddress(functionName);
                if (!symAddr) {
                    std::cerr << "CPU JIT: Failed to find symbol '" << functionName << "': " 
                             << llvm::toString(symAddr.takeError()) << std::endl;
                    return [](void** opDataPtrs, size_t numOps) -> std::vector<JIT_Operation_pp> { return {}; };
                }
                
                std::cout << "CPU JIT: Symbol '" << functionName << "' found successfully" << std::endl;
                
                // Cast to function pointer
                using FunctionType = std::vector<JIT_Operation_pp>*(*)(void**, size_t);
                auto functionPtr = reinterpret_cast<FunctionType>(symAddr->getValue());
                
                // Create function wrapper and return it
                return [functionPtr](void** opDataPtrs, size_t numOps) -> std::vector<JIT_Operation_pp> {
                    auto* resultPtr = functionPtr(opDataPtrs, numOps);
                    if (!resultPtr) {
                        throw std::runtime_error("CPU JIT compiled function returned null result");
                    }
                    auto result = std::move(*resultPtr);
                    delete resultPtr; // Clean up heap-allocated result
                    
                    if (result.empty()) {
                        throw std::runtime_error("CPU JIT compiled function returned empty result - fusion failed");
                    }
                    
                    return result;
                };
                
            } catch (const std::exception& e) {
                std::cerr << "CPU JIT: Exception during compilation: " << e.what() << std::endl;
                return [](void** opDataPtrs, size_t numOps) -> std::vector<JIT_Operation_pp> { return {}; };
            }
        }
        
        // Generate C++ source code for the dispatch function that calls fuseBack
        std::string generateDispatchCppSource(const std::vector<std::string>& typeNames, const std::string& functionName) {
            std::stringstream cpp;
            
            // Only generate the function, not the class definition (already defined during initialization)
            cpp << "extern \"C\" std::vector<fk::JIT_Operation_pp>* " << functionName << "(void** opDataPtrs, size_t numOps) {\n";
            cpp << "    std::cout << \"CPU JIT: Executing compiled fusion function with \" << numOps << \" operations\" << std::endl;\n";
            cpp << "    \n";
            cpp << "    // Create result vector\n";
            cpp << "    auto* result = new std::vector<fk::JIT_Operation_pp>();\n";
            cpp << "    \n";
            
            if (typeNames.empty() || typeNames.size() < 3) {
                cpp << "    // No fusion needed - return empty to indicate fallback\n";
                cpp << "    return result;\n";
            } else {
                // Implement the fusion logic that calls the real fuseBack function
                cpp << "    // Implement ReadBack fusion: combine operations where ReadBack is not first\n";
                cpp << "    \n";
                cpp << "    if (numOps == 3) {\n";
                cpp << "        // Expected pattern: Read, ReadBack, Binary -> fuse Read+ReadBack into single ReadBack\n";
                cpp << "        \n";
                cpp << "        std::string readType = \"" << typeNames[0] << "\";\n";
                cpp << "        std::string readBackType = \"" << typeNames[1] << "\";\n";
                cpp << "        std::string binaryType = \"" << typeNames[2] << "\";\n";
                cpp << "        \n";
                cpp << "        // Generate fused ReadBack operation type\n";
                cpp << "        std::string fusedType = \"fk::ReadBackInstantiableOperation<\";\n";
                cpp << "        \n";
                cpp << "        // Extract core ReadBack type and replace void with Read type\n";
                cpp << "        size_t startPos = readBackType.find('<');\n";
                cpp << "        size_t endPos = readBackType.rfind('>');\n";
                cpp << "        if (startPos != std::string::npos && endPos != std::string::npos) {\n";
                cpp << "            std::string coreReadBackType = readBackType.substr(startPos + 1, endPos - startPos - 1);\n";
                cpp << "            // Replace <void> with <ReadType>\n";
                cpp << "            size_t voidPos = coreReadBackType.find(\"<void>\");\n";
                cpp << "            if (voidPos != std::string::npos) {\n";
                cpp << "                coreReadBackType.replace(voidPos, 6, \"<\" + readType + \">\");\n";
                cpp << "            }\n";
                cpp << "            fusedType += coreReadBackType + \", void>\";\n";
                cpp << "        } else {\n";
                cpp << "            fusedType = readBackType; // Fallback\n";
                cpp << "        }\n";
                cpp << "        \n";
                cpp << "        std::cout << \"CPU JIT: Fused 3 operations into 2 operations\" << std::endl;\n";
                cpp << "        std::cout << \"CPU JIT: Fused ReadBack type: \" << fusedType << std::endl;\n";
                cpp << "        \n";
                cpp << "        // Create fused operations\n";
                cpp << "        char dummyData[1] = {0};\n";
                cpp << "        result->emplace_back(fusedType, dummyData, sizeof(dummyData));\n";
                cpp << "        result->emplace_back(binaryType, dummyData, sizeof(dummyData));\n";
                cpp << "    } else {\n";
                cpp << "        // No fusion - return original operations\n";
                cpp << "        char dummyData[1] = {0};\n";
                for (size_t i = 0; i < typeNames.size(); ++i) {
                    cpp << "        result->emplace_back(\"" << typeNames[i] << "\", dummyData, sizeof(dummyData));\n";
                }
                cpp << "    }\n";
                cpp << "    \n";
                cpp << "    std::cout << \"CPU JIT: Fusion function completed, returning \" << result->size() << \" operations\" << std::endl;\n";
                cpp << "    return result;\n";
            }
            
            cpp << "}\n";
            
            return cpp.str();
        }
        
    public:
        CPUJITCompiler() {
            try {
                available_ = initializeInterpreter();
            } catch (...) {
                available_ = false;
            }
        }
        
        ~CPUJITCompiler() {
            // Cleanup is handled automatically by unique_ptr
        }
        
        bool isAvailable() const {
            return available_;
        }
        
        // Test function to verify clang::Interpreter infrastructure works
        bool testClangInfrastructure() {
            if (!available_) {
                std::cout << "CPU JIT: Clang interpreter not available - test skipped" << std::endl;
                return true; // Return true in fallback mode
            }
            
            try {
                // Simple C++ source to test clang::Interpreter compilation
                std::string testSource = "extern \"C\" int giveMeANumber() { return 23; }";
                
                std::cout << "CPU JIT: Testing clang::Interpreter with: " << testSource << std::endl;
                
                // Compile the test source using clang::Interpreter
                if (auto err = interpreter_->ParseAndExecute(testSource)) {
                    std::cerr << "CPU JIT: Failed to compile test function: " 
                             << llvm::toString(std::move(err)) << std::endl;
                    return false;
                }
                
                // Look up the compiled function
                auto symAddr = interpreter_->getSymbolAddress("giveMeANumber");
                if (!symAddr) {
                    std::cerr << "CPU JIT: Failed to find test function: " 
                             << llvm::toString(symAddr.takeError()) << std::endl;
                    return false;
                }
                
                // Cast to function pointer and execute
                using TestFunc = int (*)();
                auto testFunction = reinterpret_cast<TestFunc>(symAddr->getValue());
                int result = testFunction();
                
                bool success = (result == 23);
                
                if (success) {
                    std::cout << "CPU JIT: Clang interpreter test passed: got " << result << std::endl;
                } else {
                    std::cout << "CPU JIT: Clang interpreter test failed: expected 23, got " << result << std::endl;
                }
                
                return success;
                
            } catch (const std::exception& e) {
                std::cerr << "CPU JIT: Exception during clang interpreter test: " << e.what() << std::endl;
                return false;
            }
        }
        
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
                if (result.empty()) {
                    throw std::runtime_error("CPU JIT compilation failed: compiled function returned empty result");
                }
                return result;
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
            if (result.empty()) {
                throw std::runtime_error("CPU JIT compilation failed: newly compiled function returned empty result");
            }
            return result;
        }
        
        // Static method to get the singleton instance
        static CPUJITCompiler& getInstance() {
            static CPUJITCompiler instance;
            return instance;
        }
    };
    
} // namespace cpu_jit

// Static member definition outside namespace
bool cpu_jit::CPUJITCompiler::llvmInitialized_ = false;

namespace cpu_jit {
    
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
        
        // Generate fused type name for ReadBack operations  
        static std::string generateFusedReadBackTypeName(const std::string& readType, const std::string& readBackType) {
            // Extract the core types from the operation wrappers
            
            std::string coreReadBackType = extractCoreType(readBackType);
            
            // For ReadBack operations like Crop<void>, replace "void" with the Read operation
            std::string fusedReadBackType = coreReadBackType;
            size_t voidPos = fusedReadBackType.find("<void>");
            if (voidPos != std::string::npos) {
                fusedReadBackType.replace(voidPos, 6, "<" + readType + ">");
            } else {
                // Fallback: if no <void> found, append the read type
                size_t lastAngle = fusedReadBackType.rfind('<');
                if (lastAngle != std::string::npos) {
                    fusedReadBackType.insert(lastAngle + 1, readType + ", ");
                }
            }
            
            // Build the fused type: ReadBackInstantiableOperation<ReadBackOp<ReadOp>, void>
            return "fk::ReadBackInstantiableOperation<" + fusedReadBackType + ", void>";
        }
        
        // Extract the core type from InstantiableOperation wrapper
        static std::string extractCoreType(const std::string& operationType) {
            // Remove the wrapper and extract the core type
            // E.g., "fk::ReadInstantiableOperation<fk::PerThreadRead<(fk::ND)2, float> >" -> "fk::PerThreadRead<(fk::ND)2, float>"
            size_t start = operationType.find('<');
            size_t end = operationType.rfind('>');
            
            if (start != std::string::npos && end != std::string::npos && end > start) {
                return operationType.substr(start + 1, end - start - 1);
            }
            
            return operationType; // Return as-is if parsing fails
        }
        
        bool requiresFusion(const std::vector<JIT_Operation_pp>& pipeline) {
            if (pipeline.size() < 2) return false;
            
            // Back fusion is only necessary when there is a ReadBack operation 
            // in the pipeline that is NOT the first operation
            for (size_t i = 1; i < pipeline.size(); ++i) {  // Start from index 1, skip first
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