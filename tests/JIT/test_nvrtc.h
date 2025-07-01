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

#ifndef FK_TEST_NVRTC
#define FK_TEST_NVRTC

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/core/utils/type_to_string.h>
#include <fused_kernel/core/execution_model/executors.h>

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <cstring> // For memcpy

#if defined(NVRTC_ENABLED)
// CUDA headers
#include <cuda.h>
#include <nvrtc.h>

// Helper macro for NVRTC error checking
#define NVRTC_CHECK(result) { \
    if (result != NVRTC_SUCCESS) { \
        std::cerr << "NVRTC Error: " << nvrtcGetErrorString(result) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("NVRTC API call failed."); \
    } \
}

int launch() {
    // --- 1. Mock Header Content ---
    const char* main_source_content =
    R"( 
        #include <fused_kernel/core/execution_model/executors.h>
        #include <fused_kernel/algorithms/basic_ops/arithmetic.h>
    )";

    // --- 2. Define the Runtime Pipeline using JIT_Operation_pp ---
    std::cout << "Defining runtime pipeline with C++ JIT_Operation_pp..." << std::endl;
    fk::Stream stream;
    fk::Ptr1D<float> d_data_in(256);
    for (int i = 0; i < 256; ++i) {
        d_data_in.at(fk::Point(i)) = static_cast<float>(i); // Initialize input data
    }
    d_data_in.upload(stream);
    fk::Ptr1D<float> d_data_out(256);
    const auto read_op = fk::PerThreadRead<fk::_1D, float>::build(d_data_in);
    const auto mul_op = fk::Mul<float>::build(2.f);
    const auto add_op = fk::Add<float>::build(5.f);
    const auto write_op = fk::PerThreadWrite<fk::_1D, float>::build(d_data_out);

    std::vector<fk::JIT_Operation_pp> pipeline = buildOperationPipeline(
        read_op, // Read operation
        mul_op,  // First operation (Op1)
        add_op,  // Second operation (Op2)
        write_op // Write operation
    );

    // --- 3. Dynamically build the kernel name expression ---
    std::string name_expression_str = buildNameExpression(pipeline);
    const char* name_expression = name_expression_str.c_str();
    std::cout << "Dynamically generated name expression: " << name_expression << std::endl;

    // --- 4. CUDA Init & NVRTC Setup ---
    gpuErrchk(cuInit(0));
    CUdevice device;
    gpuErrchk(cuDeviceGet(&device, 0));
    CUcontext context;
    gpuErrchk(cuCtxCreate(&context, 0, device));

    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, main_source_content, "pipeline.cu", 2, nullptr, nullptr));

    // --- 5. Compile ---
    NVRTC_CHECK(nvrtcAddNameExpression(prog, name_expression));
    const char* options[] = { "--std=c++17", "-ID:/include", "-IE:/GitHub/FKL/include", "-DNVRTC_COMPILER"};
    nvrtcResult compile_result = nvrtcCompileProgram(prog, 4, options);
    size_t log_size;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1) {
        std::vector<char> log(log_size);
        NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));
        std::cout << "NVRTC Log:\n" << log.data() << std::endl;
    }
    NVRTC_CHECK(compile_result);

    // --- 6. Get PTX, Mangled Name, and Kernel Handle ---
    const char* mangled_name;
    NVRTC_CHECK(nvrtcGetLoweredName(prog, name_expression, &mangled_name));
    std::cout << "Name expression: " << name_expression << std::endl;
    std::cout << "Mangled kernel name: " << mangled_name << std::endl;
    size_t ptx_size;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
    std::vector<char> ptx(ptx_size);
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    CUmodule module;
    CUfunction kernel_func;
    gpuErrchk(cuModuleLoadData(&module, ptx.data()));
    gpuErrchk(cuModuleGetFunction(&kernel_func, module, mangled_name));

    // --- 7. Prepare Data and Launch ---
    const int N = 256;
    std::vector<void*> kernel_args_vec = buildKernelArgumentsFKL(pipeline);//buildKernelArguments(d_data_in, d_data_out, pipeline);

    std::cout << "Launching dynamically constructed kernel..." << std::endl;
    gpuErrchk(cuLaunchKernel(kernel_func, 1, 1, 1, N, 1, 1, 0, reinterpret_cast<CUstream>(stream.getCUDAStream()), kernel_args_vec.data(), nullptr));

    // --- 8. Verify & Cleanup ---
    d_data_out.download(stream);
    stream.sync();
    std::cout << "Result of op1(factor=2.0) then op2(offset=5.0) on data[3]: " << d_data_out.at(fk::Point(3)) << " (expected 11)" << std::endl;

    gpuErrchk(cuModuleUnload(module));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
    gpuErrchk(cuCtxDestroy(context));

    // --- 9. No manual cleanup needed! The vector's and JIT_Operation_pp's destructors handle it. ---
    std::cout << "Cleanup handled automatically by C++ destructors." << std::endl;

    return 0;
}
#else

int launch() {
    std::cerr << "NVRTC is not enabled. Skipping JIT compilation test." << std::endl;
    return 0;
}

#endif // NVRTC_ENABLED

#endif // FK_TEST_NVRTC