# CPU JIT Compilation for ReadBack Fusion

This implementation provides runtime compilation capabilities for fusing ReadBack operations in the FusedKernelLibrary using LLVM ORCv2.

## Overview

The system allows you to:

1. Collect operations in an `std::vector<JIT_Operation_pp>` where each operation contains:
   - `void* opData`: Pointer to the operation data
   - `std::string opType`: String representation of the operation type

2. Runtime compilation generates code that:
   - Casts `void*` pointers to concrete types based on `opType` strings
   - Calls the `fuseBack` template function with proper types
   - Returns a fused operation pipeline

## Usage Example

```cpp
#include <fused_kernel/core/execution_model/executor_details/cpu_jit_details.h>

// Create operations vector
std::vector<fk::JIT_Operation_pp> operations;

// Add ReadBack operations
SomeReadBackOp readOp1{...};
AnotherReadBackOp readOp2{...};
SomeWriteOp writeOp{...};

operations.emplace_back(&readOp1, "ReadBackOperation<SomeType>");
operations.emplace_back(&readOp2, "ReadBackOperation<AnotherType>");
operations.emplace_back(&writeOp, "WriteOperation<OutputType>");

// Compile and fuse ReadBack operations
auto fusedOps = fk::compileAndFuseReadBackOperations(operations);

// The result contains optimally fused operations
```

## Implementation Details

### Files
- `include/fused_kernel/core/execution_model/executor_details/cpu_jit_details.h`: Main header
- `lib/cpu_jit_details.cpp`: Implementation with LLVM integration
- `tests/JIT/test_cpu_jit.h`: Unit tests

### LLVM Integration
- Uses LLVM ORCv2 for runtime compilation
- Statically links LLVM libraries (~39MB binary size)
- Falls back to placeholder when LLVM is not available
- Supports both FK_LLVM_AVAILABLE and non-LLVM builds

### Key Components
1. **JIT_Operation_pp**: Runtime operation representation
2. **CpuJitDetails**: LLVM ORCv2 compilation engine
3. **fuseBack**: Template function implementing fusion logic
4. **compileAndFuseReadBackOperations**: High-level fusion interface

## Building

The system automatically detects LLVM and enables JIT compilation:

```bash
mkdir build && cd build
cmake ..
make -j4
make test
```

When LLVM is found, the FK_LLVM_AVAILABLE flag is set and full JIT functionality is enabled.