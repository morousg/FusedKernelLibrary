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

#include <fused_kernel/core/execution_model/executor_details/cpu_jit_details.h>

namespace fk {

#ifdef ENABLE_LLVM_JIT

CPUJITCompiler::CPUJITCompiler() {
    // Initialize LLVM native target
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    
    // Create LLVM context
    context_ = std::make_unique<llvm::LLVMContext>();
    
    // Create JIT instance
    auto jitBuilder = llvm::orc::LLJITBuilder();
    auto jitResult = jitBuilder.create();
    
    if (!jitResult) {
        throw std::runtime_error("Failed to create LLVM JIT: " + 
                               llvm::toString(jitResult.takeError()));
    }
    
    jit_ = std::move(*jitResult);
}

CPUJITCompiler::~CPUJITCompiler() = default;

std::vector<JIT_Operation_pp> CPUJITCompiler::fuseOperations(const std::vector<JIT_Operation_pp>& operations) {
    // Check if ReadBack fusion is needed
    if (!needsReadBackFusion(operations)) {
        return operations; // No fusion needed
    }
    
    try {
        // Generate LLVM IR for the fusion function
        auto module = generateFusionCode(operations);
        
        // Add the module to the JIT
        auto threadSafeModule = llvm::orc::ThreadSafeModule(std::move(module), 
                                                           llvm::orc::ThreadSafeContext(std::move(context_)));
        
        auto addResult = jit_->addIRModule(std::move(threadSafeModule));
        if (addResult) {
            throw std::runtime_error("Failed to add module to JIT: " + 
                                   llvm::toString(std::move(addResult)));
        }
        
        // Look up the fusion function
        auto fusionSymbol = jit_->lookup("fuse_operations");
        if (!fusionSymbol) {
            throw std::runtime_error("Failed to find fusion function: " + 
                                   llvm::toString(fusionSymbol.takeError()));
        }
        
        // Execute the fusion function
        using FusionFunctionType = std::vector<JIT_Operation_pp>(*)(const std::vector<JIT_Operation_pp>&);
        auto fusionFunction = fusionSymbol->toPtr<FusionFunctionType>();
        
        return fusionFunction(operations);
        
    } catch (const std::exception& e) {
        // Fall back to non-fused operations on error
        return operations;
    }
}

std::unique_ptr<llvm::Module> CPUJITCompiler::generateFusionCode(const std::vector<JIT_Operation_pp>& operations) {
    auto module = std::make_unique<llvm::Module>("fusion_module", *context_);
    llvm::IRBuilder<> builder(*context_);
    
    // Create the fusion function signature
    // std::vector<JIT_Operation_pp> fuse_operations(const std::vector<JIT_Operation_pp>& ops)
    auto voidType = llvm::Type::getVoidTy(*context_);
    auto int8Type = llvm::Type::getInt8Ty(*context_);
    auto int8PtrType = llvm::PointerType::get(int8Type, 0);
    
    // For simplicity, we'll create a stub function that calls the fallback
    // In a real implementation, this would generate the actual casting and fusion logic
    std::vector<llvm::Type*> paramTypes = {int8PtrType};
    auto functionType = llvm::FunctionType::get(voidType, paramTypes, false);
    auto function = llvm::Function::Create(functionType, llvm::Function::ExternalLinkage, 
                                         "fuse_operations", module.get());
    
    // Create basic block
    auto basicBlock = llvm::BasicBlock::Create(*context_, "entry", function);
    builder.SetInsertPoint(basicBlock);
    
    // For now, just return (this is a placeholder for the actual fusion logic)
    builder.CreateRetVoid();
    
    return module;
}

std::string CPUJITCompiler::buildFusionFunctionCode(const std::vector<JIT_Operation_pp>& operations) {
    std::ostringstream code;
    
    code << "#include <fused_kernel/core/execution_model/executor_details/cpu_jit_details.h>\n";
    code << "extern \"C\" std::vector<fk::JIT_Operation_pp> fuse_operations_impl(const std::vector<fk::JIT_Operation_pp>& ops) {\n";
    code << "  std::vector<fk::JIT_Operation_pp> result;\n";
    
    // Generate type-specific casting and fusion logic
    for (size_t i = 0; i < operations.size(); ++i) {
        const auto& op = operations[i];
        code << "  // Process operation " << i << " of type: " << op.opType << "\n";
        
        // Here we would generate the actual casting code based on the type string
        // For now, we'll just copy the operation
        code << "  result.push_back(ops[" << i << "]);\n";
    }
    
    code << "  return result;\n";
    code << "}\n";
    
    return code.str();
}

bool CPUJITCompiler::needsReadBackFusion(const std::vector<JIT_Operation_pp>& operations) {
    // Simple heuristic: check if we have ReadBack operations that can be fused
    // In a real implementation, this would analyze the operation types more thoroughly
    
    bool hasReadBack = false;
    for (const auto& op : operations) {
        // Check if the type name contains "ReadBack" or similar patterns
        if (op.opType.find("ReadBack") != std::string::npos ||
            op.opType.find("Resize") != std::string::npos) {
            hasReadBack = true;
            break;
        }
    }
    
    return hasReadBack && operations.size() > 1;
}

#endif // ENABLE_LLVM_JIT

std::vector<JIT_Operation_pp> compileAndFuseOperations(const std::vector<JIT_Operation_pp>& operations) {
#ifdef ENABLE_LLVM_JIT
    try {
        CPUJITCompiler compiler;
        return compiler.fuseOperations(operations);
    } catch (...) {
        // Fall back to non-fused operations on any error
        return fallbackFuseOperations(operations);
    }
#else
    return fallbackFuseOperations(operations);
#endif
}

std::vector<JIT_Operation_pp> fallbackFuseOperations(const std::vector<JIT_Operation_pp>& operations) {
    // When LLVM JIT is not available or fails, return operations unchanged
    return operations;
}

} // namespace fk