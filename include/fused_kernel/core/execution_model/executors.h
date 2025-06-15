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

// In the future we will change all .h files to .h files, since we are going to be multi platform
#ifndef FK_EXECUTORS_CUH
#define FK_EXECUTORS_CUH

#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/execution_model/data_parallel_patterns.h>
#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/set.h>
#include <fused_kernel/core/execution_model/stream.h>

#if defined(NVRTC_ENABLED)
#include <unordered_map>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <fused_kernel/core/utils/type_to_string.h>
#endif

#if defined(__NVCC__) || defined(__HIP__)
#include <fused_kernel/core/execution_model/executor_kernels.h>
#endif

namespace fk {

#if defined(__NVCC__) || defined(__HIP__) || defined(NVRTC_ENABLED)
    struct CtxDim3 {
        uint x;
        uint y;
        uint z;
        constexpr CtxDim3(const dim3& dims) : x(dims.x), y(dims.y), z(dims.z) {}
        constexpr CtxDim3() : x(1), y(1), z(1) {}
        constexpr CtxDim3(const uint& x) : x(x), y(1), z(1) {}
        constexpr CtxDim3(const uint& x, const uint& y) : x(x), y(y), z(1) {}
        constexpr CtxDim3(const uint& x, const uint& y, const uint& z) : x(x), y(y), z(z) {}
    };
#endif

    template <typename Child>
    struct BaseExecutor {
        FK_STATIC_STRUCT_SELFTYPE(BaseExecutor, BaseExecutor)
        template <enum ParArch PA, typename... IOps>
        FK_HOST_FUSE void executeOperations(Stream_<PA>& stream, const IOps&... iOps) {
            Child::executeOperations_helper(stream, iOps...);
        }

        template <enum ParArch PA, typename I, typename... IOps>
        FK_HOST_FUSE void executeOperations(const Ptr2D<I>& input, Stream_<PA>& stream,
                                            const IOps&... iOps) {
            Child::executeOperations_helper(stream, PerThreadRead<_2D, I>::build({ input }), iOps...);
        }

        template <enum ParArch PA, typename I, typename O, typename... IOps>
        FK_HOST_FUSE void executeOperations(const Ptr2D<I>& input, const Ptr2D<O>& output,
                                            Stream_<PA>& stream, const IOps&... iOps) {
            Child::executeOperations_helper(stream,
            PerThreadRead<_2D, I>::build({ input }), iOps..., PerThreadWrite<_2D, O>::build({ output }));
        }

        template <enum ParArch PA, typename I, size_t BATCH, typename... IOps>
        FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, const int& activeBatch, const I& defaultValue,
                                            Stream_<PA>& stream, const IOps&... iOps) {
            const auto batchReadIOp = PerThreadRead<_2D, I>::build(activeBatch, defaultValue, input);
            Child::executeOperations_helper(stream, batchReadIOp, iOps...);
        }

        template <enum ParArch PA, typename I, size_t BATCH, typename... IOps>
        FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, BATCH>& input,
                                            Stream_<PA>& stream, const IOps&... iOps) {
            const auto batchReadIOp = PerThreadRead<_2D, I>::build(input);
            Child::executeOperations_helper(stream, batchReadIOp, iOps...);
        }

        template <enum ParArch PA, typename I, typename O, size_t Batch, typename... IOps>
        FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const I& defaultValue,
                                            const Tensor<O>& output, Stream_<PA>& stream, const IOps&... iOps) {
            const auto batchReadIOp = PerThreadRead<_2D, I>::build(activeBatch, defaultValue, input);
            const auto writeOp = PerThreadWrite<_3D, O>::build(output);
            Child::executeOperations_helper(stream, batchReadIOp, iOps..., writeOp);
        }

        template <enum ParArch PA, typename I, typename O, size_t Batch, typename... IOps>
        FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const Tensor<O>& output,
                                            Stream_<PA>& stream, const IOps&... iOps) {
            const auto batchReadIOp = PerThreadRead<_2D, I>::build(input);
            const auto writeOp = PerThreadWrite<_3D, O>::build(output);
            Child::executeOperations_helper(stream, batchReadIOp, iOps..., writeOp);
        }
    };

#define DECLARE_EXECUTOR_PARENT_IMPL \
friend class BaseExecutor<Child>; \
template <enum ParArch PA, typename... IOps> \
FK_HOST_FUSE void executeOperations(Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(stream, iOps...); \
} \
template <enum ParArch PA, typename I, typename... IOps> \
FK_HOST_FUSE void executeOperations(const Ptr2D<I>& input, Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, stream, iOps...); \
} \
template <enum ParArch PA, typename I, typename O, typename... IOps> \
FK_HOST_FUSE void executeOperations(const Ptr2D<I>& input, const Ptr2D<O>& output, Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, output, stream, iOps...); \
} \
template <enum ParArch PA, typename I, size_t BATCH, typename... IOps> \
FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, const int& activeBatch, const I& defaultValue, \
                                    Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, activeBatch, defaultValue, stream, iOps...); \
} \
template <enum ParArch PA, typename I, size_t BATCH, typename... IOps> \
FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, stream, iOps...); \
} \
template <enum ParArch PA, typename I, typename O, size_t Batch, typename... IOps> \
FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const I& defaultValue, \
                                    const Tensor<O>& output, Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, activeBatch, defaultValue, output, stream, iOps...); \
} \
template <enum ParArch PA, typename I, typename O, size_t Batch, typename... IOps> \
FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const Tensor<O>& output, \
                                    Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, output, stream, iOps...); \
}

    template <typename DataParallelPattern>
    struct Executor {
        FK_STATIC_STRUCT_SELFTYPE(Executor, Executor)
        static_assert(DataParallelPattern::PAR_ARCH == ParArch::GPU_NVIDIA ||
                      DataParallelPattern::PAR_ARCH == ParArch::CPU ||
                      DataParallelPattern::PAR_ARCH == ParArch::GPU_NVIDIA_JIT, "Only GPU_NVIDIA, CPU and GPU_NVIDIA_JIT are supported for now");
    };

    template <enum TF TFEN>
    struct Executor<TransformDPP<ParArch::CPU, TFEN, void>> {
        FK_STATIC_STRUCT_SELFTYPE(Executor, Executor)
    private:
        using Child = Executor<TransformDPP<ParArch::CPU, TFEN>>;
        using Parent = BaseExecutor<Child>;
        template <typename... IOps>
        FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::CPU>& stream, const IOps&... iOps) {
            constexpr ParArch PA = ParArch::CPU;
            const auto tDetails = TransformDPP<PA, TFEN>::build_details(iOps...);
            using TDPPDetails = std::decay_t<decltype(tDetails)>;
            if constexpr (TDPPDetails::TFI::ENABLED) {
                if (!tDetails.threadDivisible) {
                    TransformDPP<PA, TFEN, TDPPDetails, false>::exec(tDetails, iOps...);
                } else {
                    TransformDPP<PA, TFEN, TDPPDetails, true>::exec(tDetails, iOps...);
                }
            } else {
                TransformDPP<PA, TFEN, TDPPDetails, true>::exec(tDetails, iOps...);
            }
        }
    public:
        FK_HOST_FUSE ParArch parArch() {
            return ParArch::CPU;
        }
        DECLARE_EXECUTOR_PARENT_IMPL
    };

#if defined(__NVCC__) || defined(__HIP__) || defined(NVRTC_ENABLED)
    struct ComputeBestSolutionBase {
        FK_HOST_FUSE uint computeDiscardedThreads(const uint width, const uint height, const uint blockDimx, const uint blockDimy) {
            const uint modX = width % blockDimx;
            const uint modY = height % blockDimy;
            const uint th_disabled_in_X = modX == 0 ? 0 : blockDimx - modX;
            const uint th_disabled_in_Y = modY == 0 ? 0 : blockDimy - modY;
            return (th_disabled_in_X * (modY == 0 ? height : (height + blockDimy)) + th_disabled_in_Y * width);
        }
    };

    template <uint bxS_t, uint byS_t>
    struct computeBestSolution {};

    template <uint bxS_t>
    struct computeBestSolution<bxS_t, 0> final : public ComputeBestSolutionBase {
        FK_HOST_FUSE void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
            const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[bxS_t], blockDimY[0][bxS_t]);
            if (minDiscardedThreads > currentDiscardedThreads) {
                minDiscardedThreads = currentDiscardedThreads;
                bxS = bxS_t;
                byS = 0;
                if (minDiscardedThreads == 0) return;
            }
            computeBestSolution<bxS_t, 1>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);
        }
    };

    template <uint bxS_t>
    struct computeBestSolution<bxS_t, 1> final : public ComputeBestSolutionBase {
        FK_HOST_FUSE void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
            const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[bxS_t], blockDimY[1][bxS_t]);
            if (minDiscardedThreads > currentDiscardedThreads) {
                minDiscardedThreads = currentDiscardedThreads;
                bxS = bxS_t;
                byS = 1;
                if constexpr (bxS_t == 3) return;
                if (minDiscardedThreads == 0) return;
            }
            computeBestSolution<bxS_t + 1, 0>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);
        }
    };

    template <>
    struct computeBestSolution<3, 1> final : public ComputeBestSolutionBase {
        FK_HOST_FUSE void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
            const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[3], blockDimY[1][3]);
            if (minDiscardedThreads > currentDiscardedThreads) {
                minDiscardedThreads = currentDiscardedThreads;
                bxS = 3;
                byS = 1;
            }
        }
    };
#endif
#if defined(__NVCC__) || defined(__HIP__)
    template <enum TF TFEN>
    struct Executor<TransformDPP<ParArch::GPU_NVIDIA, TFEN>> {
        FK_STATIC_STRUCT_SELFTYPE(Executor, Executor)
    private:
        using Child = Executor<TransformDPP<ParArch::GPU_NVIDIA, TFEN>>;
        using Parent = BaseExecutor<Child>;

        FK_HOST_FUSE CtxDim3 getDefaultBlockSize(const uint& width, const uint& height) {
            constexpr uint blockDimX[4] = { 32, 64, 128, 256 };  // Possible block sizes in the x axis
            constexpr uint blockDimY[2][4] = { { 8,  4,   2,   1},
                                              { 6,  3,   3,   2} };  // Possible block sizes in the y axis according to blockDim.x

            uint minDiscardedThreads = UINT_MAX;
            uint bxS = 0; // from 0 to 3
            uint byS = 0; // from 0 to 1

            computeBestSolution<0, 0>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);

            return CtxDim3(blockDimX[bxS], blockDimY[byS][bxS]);
        }
        template <typename... IOps>
        FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::GPU_NVIDIA>& stream_, const IOps&... iOps) {
            const cudaStream_t stream = stream_.getCUDAStream();
            constexpr ParArch PA = ParArch::GPU_NVIDIA;
            const auto tDetails = TransformDPP<PA, TFEN>::build_details(iOps...);
            if constexpr (decltype(tDetails)::TFI::ENABLED) {
                const ActiveThreads activeThreads = tDetails.activeThreads;

                const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

                const dim3 block{ ctx_block.x, ctx_block.y, 1 };
                const dim3 grid{ static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                                 static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))),
                                 activeThreads.z };
                if (!tDetails.threadDivisible) {
                    launchTransformDPP_Kernel<PA, TFEN, false><<<grid, block, 0, stream>>>(tDetails, iOps...);
                    gpuErrchk(cudaGetLastError());
                } else {
                    launchTransformDPP_Kernel<PA, TFEN, true><<<grid, block, 0, stream>>>(tDetails, iOps...);
                    gpuErrchk(cudaGetLastError());
                }
            } else {
                const auto readOp = get<0>(iOps...);

                const ActiveThreads activeThreads = readOp.getActiveThreads();

                const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

                const dim3 block{ ctx_block.x, ctx_block.y, 1 };
                const dim3 grid{ static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                                 static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))),
                                 activeThreads.z };
                launchTransformDPP_Kernel<PA, TFEN, true><<<grid, block, 0, stream>>>(tDetails, iOps...);
                gpuErrchk(cudaGetLastError());
            }
        }
    public:
        FK_HOST_FUSE ParArch parArch() {
            return ParArch::GPU_NVIDIA;
        }
        DECLARE_EXECUTOR_PARENT_IMPL
    };
#endif

#if defined(NVRTC_ENABLED)
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

    namespace jit_internal {
        // --- Helper Functions for Dynamic Pipeline Construction ---
        std::string buildNameExpression(const std::string& kernelName, const std::vector<JIT_Operation_pp>& pipeline) {
            std::stringstream ss;
            ss << "&" << kernelName;
            for (size_t i = 0; i < pipeline.size(); ++i) {
                ss << pipeline[i].getType();
                if (i < pipeline.size() - 1) {
                    ss << ", ";
                }
            }
            ss << ">";
            return ss.str();
        }

        std::vector<void*> buildKernelArguments(const std::vector<JIT_Operation_pp>& pipeline) {
            std::vector<void*> args;
            for (const auto& op : pipeline) {
                args.push_back(op.getData());
            }
            return args;
        }

        template <typename... IOps>
        std::vector<JIT_Operation_pp> buildOperationPipeline(const IOps&... iOps) {
            std::vector<JIT_Operation_pp> pipeline;
            (pipeline.emplace_back(typeToString<IOps>(), &iOps, sizeof(IOps)), ...);
            return pipeline;
        }
    } // jit_internal


    class JitFkKernel {
        CUmodule m_module;
        CUfunction m_kernelFunc;
        std::string m_nameExpression;
        std::string m_includes{
            R"( 
                #include <fused_kernel/core/execution_model/executor_kernels.h>
                #include <fused_kernel/algorithms/algorithms.h>
                #include <fused_kernel/core/execution_model/data_parallel_patterns.h>
            )"
        };
    public:
        JitFkKernel(const std::string& kernelName,
                    const std::vector<JIT_Operation_pp>& pipeline) {
            m_nameExpression = jit_internal::buildNameExpression(kernelName, pipeline);
            nvrtcProgram fklProg;
            gpuErrchk(nvrtcCreateProgram(&fklProg, m_includes.c_str(), m_nameExpression.c_str(), 0, nullptr, nullptr));
            gpuErrchk(nvrtcAddNameExpression(fklProg, m_nameExpression.c_str()));
            const char* options[] = { "--std=c++17", "-ID:/include", "-IE:/GitHub/FKL/include", "-DNVRTC_COMPILER" };
            nvrtcResult compile_result = nvrtcCompileProgram(fklProg, 4, options);
            size_t log_size;
            gpuErrchk(nvrtcGetProgramLogSize(fklProg, &log_size));
            if (log_size > 1) {
                std::stringstream nvrtc_log;
                std::vector<char> log(log_size);
                const char* error_str = nvrtcGetErrorString(compile_result);
                gpuErrchk(nvrtcGetProgramLog(fklProg, log.data()));
                nvrtc_log << "NVRTC Error: " << error_str << std::endl;
                nvrtc_log << "NVRTC Log:\n" << log.data() << std::endl;
                throw std::runtime_error(nvrtc_log.str());
            }
            const char* mangled_name;
            gpuErrchk(nvrtcGetLoweredName(fklProg, m_nameExpression.c_str(), &mangled_name));
            size_t ptx_size;
            gpuErrchk(nvrtcGetPTXSize(fklProg, &ptx_size));
            std::vector<char> ptx(ptx_size);
            gpuErrchk(nvrtcGetPTX(fklProg, ptx.data()));
            gpuErrchk(cuModuleLoadData(&m_module, ptx.data()));
            gpuErrchk(cuModuleGetFunction(&m_kernelFunc, m_module, mangled_name));
        }

        CUfunction getKernelFunction() const {
            return m_kernelFunc;
        }

        std::string getNameExpression() const {
            return m_nameExpression;
        }

        ~JitFkKernel() {
            gpuErrchk(cuModuleUnload(m_module));
        }
    };

    // Singleton class to avoid having to create instances of Executors
    // Rightnow it is not thread safe, it will be in the future.
    class JITExecutorSingleton {
        CUdevice m_device;
        CUcontext m_context;
        std::string m_includes;
        std::unordered_map<std::string, JitFkKernel> m_kernelCache;
        void addJITKernel(const JitFkKernel& fkKernel) {
            m_kernelCache[fkKernel.getNameExpression()] = fkKernel;
        }
        bool hasJITKernel(const std::string& kernelName) const {
            return m_kernelCache.find(kernelName) != m_kernelCache.end();
        }
        CUfunction getCUfunction(const std::string& kernelNameWithDetails) const {
            auto it = m_kernelCache.find(kernelNameWithDetails);
            if (it != m_kernelCache.end()) {
                return it->second.getKernelFunction();
            } else {
                throw std::runtime_error("JIT Kernel not found: " + kernelNameWithDetails);
            }
        }
    public:
        JITExecutorSingleton() {
            // Initialize the NVRTC context and device
            gpuErrchk(cuInit(0));
            gpuErrchk(cuDeviceGet(&m_device, 0));
            gpuErrchk(cuCtxCreate(&m_context, 0, m_device));
            m_includes =
                std::string(R"( 
                    #include <fused_kernel/core/execution_model/executor_kernels.h>
                    #include <fused_kernel/algorithms/algorithms.h>
                    #include <fused_kernel/core/execution_model/data_parallel_patterns.h>
                )");
        }
        ~JITExecutorSingleton() {
            // Clean up the NVRTC context
            gpuErrchk(cuCtxDestroy(m_context));
        }

        static JITExecutorSingleton& getInstance() {
            static JITExecutorSingleton instance;
            return instance;
        }

        CUfunction addKernel(const std::string& kernelName, const std::vector<JIT_Operation_pp>& pipeline) {
            const auto completeKernelExpression = jit_internal::buildNameExpression(kernelName, pipeline);
            if (!hasJITKernel(completeKernelExpression)) {
                JitFkKernel fkKernel(kernelName, pipeline);
                addJITKernel(fkKernel);
            }
            return getCUfunction(completeKernelExpression);
        }
    };

    template <enum TF TFEN>
    struct Executor<TransformDPP<ParArch::GPU_NVIDIA_JIT, TFEN, void>> {
        FK_STATIC_STRUCT_SELFTYPE(Executor, Executor)
    private:
        using Child = Executor<TransformDPP<ParArch::GPU_NVIDIA_JIT, TFEN>>;
        using Parent = BaseExecutor<Child>;
        template <typename... IOps>
        FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::GPU_NVIDIA_JIT>& stream, const IOps&... iOps) {
            constexpr ParArch PA = ParArch::GPU_NVIDIA;
            const auto tDetails = TransformDPP<PA, TFEN>::build_details(iOps...);
            using TDPPDetails = std::decay_t<decltype(tDetails)>;
            std::string detailsType = fk::typeToString<TDPPDetails>();
            std::string kernelName{ "launchTransformDPP_Kernel<ParArch::GPU_NVIDIA, " };
            std::string tfi;
            ActiveThreads activeThreads;
            std::string threadDivisible;
            if constexpr (TDPPDetails::TFI::ENABLED) {
                tfi = std::string("TF::ENABLED");
                activeThreads = tDetails.activeThreads;
                if (!tDetails.threadDivisible) {
                    threadDivisible = std::string("false");
                } else {
                    threadDivisible = std::string("true");
                }
            } else {
                tfi = std::string("TF::DISABLED");
                activeThreads = get<0>(iOps...).getActiveThreads();
                threadDivisible = std::string("true");
            }
            const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

            const dim3 block{ ctx_block.x, ctx_block.y, 1 };
            const dim3 grid{ static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                             static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))),
                             activeThreads.z };
            
            std::string kernelNameWithDetails = kernelName + tfi + ", " + threadDivisible + ", " + typeToString<TDPPDetails>() + ", ";
            std::vector<JIT_Operation_pp> pipeline = jit_internal::buildOperationPipeline(iOps...);
            CUfunction kernelFunc = JITExecutorSingleton::getInstance().addKernel(kernelNameWithDetails, pipeline);
            std::vector<void*> args = jit_internal::buildKernelArguments(pipeline);
            args.insert(args.begin(), &tDetails);
            gpuErrchk(cuLaunchKernel(kernelFunc, grid.x, grid.y, grid.z,
                                     block.x, block.y, block.z, 0,
                                     reinterpret_cast<CUstream>(stream.getCUDAStream()), args.data(), nullptr));
        }
    public:
        FK_HOST_FUSE ParArch parArch() {
            return ParArch::GPU_NVIDIA_JIT;
        }
        DECLARE_EXECUTOR_PARENT_IMPL
    };
#endif // NVRTC_ENABLED

#undef DECLARE_EXECUTOR_PARENT_IMPL
    template <enum ParArch PA, enum ND D, typename T>
    inline void setTo(const T& value, Ptr<D, T>& outputPtr, Stream_<PA>& stream) {
        RawPtr<D, T> output = outputPtr.ptr();
#if defined(__NVCC__) || defined(__HIP__)
        if constexpr (PA == ParArch::GPU_NVIDIA) {
            if (outputPtr.getMemType() == MemType::Device || outputPtr.getMemType() == MemType::DeviceAndPinned) {
                Executor<TransformDPP<ParArch::GPU_NVIDIA>>::executeOperations(stream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(output));
                if (outputPtr.getMemType() == MemType::DeviceAndPinned) {
                    Stream_<ParArch::CPU> cpuStream;
                    Executor<TransformDPP<ParArch::CPU>>::executeOperations(cpuStream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(outputPtr.ptrPinned()));
                }
            } else {
                Executor<TransformDPP<ParArch::GPU_NVIDIA>>::executeOperations(stream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(output));
            }
        } else {
            Executor<TransformDPP<PA>>::executeOperations(stream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(output));
        }
#else
        Executor<TransformDPP<PA>>::executeOperations(stream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(outputPtr));
#endif
    }

} // namespace fk

#endif // FK_EXECUTORS_CUH