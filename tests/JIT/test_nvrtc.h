/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Hguet)

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

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <cstring> // For memcpy

   // CUDA headers
#include <cuda.h>
#include <nvrtc.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(result) { \
    if (result != CUDA_SUCCESS) { \
        const char* error_name; \
        cuGetErrorName(result, &error_name); \
        std::cerr << "CUDA Error: " << error_name << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("CUDA API call failed."); \
    } \
}

// Helper macro for NVRTC error checking
#define NVRTC_CHECK(result) { \
    if (result != NVRTC_SUCCESS) { \
        std::cerr << "NVRTC Error: " << nvrtcGetErrorString(result) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("NVRTC API call failed."); \
    } \
}

// --- Host-side Structs (must match device-side definitions) ---
struct Op1 { float factor; };
struct Op2 { float offset; };

// --- Abstract Operation Definition (Hybrid C++ class) ---
class JIT_Operation_pp {
private:
    std::string opType; // The C++ typename of the operation struct
    void* opData;       // A pointer to an internal copy of the data (owned)
    size_t dataSize;    // The size of the data block

public:
    // Constructor: Performs a deep copy of the provided data.
    JIT_Operation_pp(std::string type, const void* data, size_t size)
        : opType(type), dataSize(size) { // Changed std::move(type) to type
        // Allocate memory and copy the parameter data
        opData = new char[dataSize];
        memcpy(opData, data, dataSize);
    }

    // Copy Constructor: Essential for use in std::vector.
    JIT_Operation_pp(const JIT_Operation_pp& other)
        : opType(other.opType), dataSize(other.dataSize) {
        // Allocate and copy data for the new object
        opData = new char[dataSize];
        memcpy(opData, other.opData, dataSize);
    }

    // Move Constructor
    JIT_Operation_pp(JIT_Operation_pp&& other) noexcept
        : opType(std::move(other.opType)), opData(other.opData), dataSize(other.dataSize) {
        // Take ownership of the other object's resources
        other.opData = nullptr;
        other.dataSize = 0;
    }

    // Copy Assignment Operator
    JIT_Operation_pp& operator=(const JIT_Operation_pp& other) {
        if (this == &other) {
            return *this;
        }
        // Free old resources
        delete[] static_cast<char*>(opData);

        // Copy new resources
        opType = other.opType;
        dataSize = other.dataSize;
        opData = new char[dataSize];
        memcpy(opData, other.opData, dataSize);

        return *this;
    }

    // Move Assignment Operator
    JIT_Operation_pp& operator=(JIT_Operation_pp&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        delete[] static_cast<char*>(opData);

        opType = std::move(other.opType);
        opData = other.opData;
        dataSize = other.dataSize;

        other.opData = nullptr;
        other.dataSize = 0;

        return *this;
    }


    // Destructor: Frees the owned memory using RAII.
    ~JIT_Operation_pp() {
        // Cast to char* to ensure correct byte-wise deletion with delete[]
        delete[] static_cast<char*>(opData);
    }

    // Public accessors
    const std::string& getType() const { return opType; }
    void* getData() const { return opData; }
};

// --- Helper Functions for Dynamic Pipeline Construction ---
std::string buildNameExpression(const std::vector<JIT_Operation_pp>& pipeline) {
    std::stringstream ss;
    ss << "&genericKernel<";
    for (size_t i = 0; i < pipeline.size(); ++i) {
        ss << pipeline[i].getType();
        if (i < pipeline.size() - 1) {
            ss << ", ";
        }
    }
    ss << ">";
    return ss.str();
}

std::vector<void*> buildKernelArguments(CUdeviceptr& d_data_in, CUdeviceptr& d_data_out, const std::vector<JIT_Operation_pp>& pipeline) {
    std::vector<void*> args;
    args.push_back(&d_data_in);
    args.push_back(&d_data_out);
    for (const auto& op : pipeline) {
        args.push_back(op.getData());
    }
    return args;
}

int launch() {
    // --- 1. Mock Header Content ---
    const char* operations_h_content = "struct Op1 { float factor; }; struct Op2 { float offset; };";
    const char* generic_kernel_h_content = R"(
        #include "operations.h"
        #include <cuda/std/type_traits>

        // Base case for recursion: return the final value.
        __device__ __forceinline__ float apply_ops(float in) {
            return in;
        }

        template<typename H, typename... T> 
        __device__ __forceinline__ float apply_ops(float in, const H& h, const T&... t) {
            float result;
            if constexpr (cuda::std::is_same_v<H, Op1>) { result = in * h.factor; }
            else if constexpr (cuda::std::is_same_v<H, Op2>) { result = in + h.offset; }
            else { result = in; } // Default case if type is unknown
            
            // Pass the result of this operation to the next one.
            return apply_ops(result, t...);
        }

        template<typename... Ops> 
        __global__ void genericKernel(const float* __restrict__ dataIn, float* dataOut, const Ops... ops) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            dataOut[tid] = apply_ops(dataIn[tid], ops...);
        }
    )";
    const char* main_source_content = "#include \"generic_kernel.h\"";

    // --- 2. Define the Runtime Pipeline using JIT_Operation_pp ---
    std::cout << "Defining runtime pipeline with C++ JIT_Operation_pp..." << std::endl;
    Op1 op1_params = { 2.0f };
    Op2 op2_params = { 5.0f };

    std::vector<JIT_Operation_pp> pipeline;
    pipeline.emplace_back("Op1", &op1_params, sizeof(Op1));
    pipeline.emplace_back("Op2", &op2_params, sizeof(Op2));

    // --- 3. Dynamically build the kernel name expression ---
    std::string name_expression_str = buildNameExpression(pipeline);
    const char* name_expression = name_expression_str.c_str();
    std::cout << "Dynamically generated name expression: " << name_expression << std::endl;

    // --- 4. CUDA Init & NVRTC Setup ---
    CUDA_CHECK(cuInit(0));
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUcontext context;
    CUDA_CHECK(cuCtxCreate(&context, 0, device));

    nvrtcProgram prog;
    const char* header_contents[] = { operations_h_content, generic_kernel_h_content };
    const char* header_names[] = { "operations.h", "generic_kernel.h" };
    NVRTC_CHECK(nvrtcCreateProgram(&prog, main_source_content, "pipeline.cu", 2, header_contents, header_names));

    // --- 5. Compile ---
    NVRTC_CHECK(nvrtcAddNameExpression(prog, name_expression));
    const char* options[] = { "--std=c++17", "-ID:/include" };
    nvrtcResult compile_result = nvrtcCompileProgram(prog, 2, options);
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
    CUDA_CHECK(cuModuleLoadData(&module, ptx.data()));
    CUDA_CHECK(cuModuleGetFunction(&kernel_func, module, mangled_name));

    // --- 7. Prepare Data and Launch ---
    const int N = 256;
    CUdeviceptr d_data_in;
    CUdeviceptr d_data_out;
    CUDA_CHECK(cuMemAlloc(&d_data_in, N * sizeof(float)));
    CUDA_CHECK(cuMemAlloc(&d_data_out, N * sizeof(float)));
    std::vector<float> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }
    CUDA_CHECK(cuMemcpyHtoD(d_data_in, h_data.data(), N * sizeof(float)));

    std::vector<void*> kernel_args_vec = buildKernelArguments(d_data_in, d_data_out, pipeline);

    std::cout << "Launching dynamically constructed kernel..." << std::endl;
    CUDA_CHECK(cuLaunchKernel(kernel_func, 1, 1, 1, N, 1, 1, 0, nullptr, kernel_args_vec.data(), nullptr));
    CUDA_CHECK(cuCtxSynchronize());

    // --- 8. Verify & Cleanup ---
    std::vector<float> h_result(N);
    CUDA_CHECK(cuMemcpyDtoH(h_result.data(), d_data_out, N * sizeof(float)));
    std::cout << "Result of op1(factor=2.0) then op2(offset=5.0) on data[3]: " << h_result[3] << " (expected 11)" << std::endl;

    CUDA_CHECK(cuMemFree(d_data_in));
    CUDA_CHECK(cuMemFree(d_data_out));
    CUDA_CHECK(cuModuleUnload(module));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
    CUDA_CHECK(cuCtxDestroy(context));

    // --- 9. No manual cleanup needed! The vector's and JIT_Operation_pp's destructors handle it. ---
    std::cout << "Cleanup handled automatically by C++ destructors." << std::endl;

    return 0;
}


#endif // FK_TEST_NVRTC