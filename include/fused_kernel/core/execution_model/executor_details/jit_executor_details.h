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

#ifndef FK_JIT_EXECUTOR_DETAILS_H
#define FK_JIT_EXECUTOR_DETAILS_H

#if defined(NVRTC_ENABLED)
namespace fk {
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
        // Default constructor
        JitFkKernel() : m_module(nullptr), m_kernelFunc(nullptr) {}
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

        // Copy constructor
        JitFkKernel(const JitFkKernel& other)
            : m_nameExpression(other.m_nameExpression), m_includes(other.m_includes),
            m_module(other.m_module), m_kernelFunc(other.m_kernelFunc) {}

        // Move constructor
        JitFkKernel(JitFkKernel&& other) noexcept
            : m_module(other.m_module),
            m_kernelFunc(other.m_kernelFunc),
            m_nameExpression(std::move(other.m_nameExpression)),
            m_includes(std::move(other.m_includes)) {
            other.m_module = nullptr;
            other.m_kernelFunc = nullptr;
        }

        // Copy assignment operator
        JitFkKernel& operator=(const JitFkKernel& other) {
            if (this != &other) {
                m_nameExpression = other.m_nameExpression;
                m_includes = other.m_includes;
                m_module = other.m_module;
                m_kernelFunc = other.m_kernelFunc;
            }
            return *this;
        }

        // Move assignment operator
        JitFkKernel& operator=(JitFkKernel&& other) noexcept {
            if (this != &other) {
                m_module = other.m_module;
                m_kernelFunc = other.m_kernelFunc;
                m_nameExpression = std::move(other.m_nameExpression);
                m_includes = std::move(other.m_includes);

                other.m_module = nullptr;
                other.m_kernelFunc = nullptr;
            }
            return *this;
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

    // --- Singleton Executor for JIT Compilation ---// --- Helper Functions for Dynamic Pipeline Construction ---
    std::string buildNameExpression(const std::vector<fk::JIT_Operation_pp>& pipeline) {
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

    std::vector<void*> buildKernelArguments(CUdeviceptr& d_data_in, CUdeviceptr& d_data_out, const std::vector<fk::JIT_Operation_pp>& pipeline) {
        std::vector<void*> args;
        args.push_back(&d_data_in);
        args.push_back(&d_data_out);
        for (const auto& op : pipeline) {
            args.push_back(op.getData());
        }
        return args;
    }
    std::vector<void*> buildKernelArgumentsFKL(const std::vector<fk::JIT_Operation_pp>& pipeline) {
        std::vector<void*> args;
        for (const auto& op : pipeline) {
            args.push_back(op.getData());
        }
        return args;
    }

    template <typename... IOps>
    std::vector<fk::JIT_Operation_pp> buildOperationPipeline(const IOps&... iOps) {
        std::vector<fk::JIT_Operation_pp> pipeline;
        (pipeline.emplace_back(fk::typeToString<IOps>(), &iOps, sizeof(IOps)), ...);
        return pipeline;
    }

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

    template <typename Read, typename Next, typename... IOps>
    constexpr inline std::vector<JIT_Operation_pp> fuseBack(const Read& read, const Next& nextOp, const IOps&... iOps) {
        static_assert(!isReadType<Next>, "A Read Operation can not go after another Read Operation, it has to be ReadBack");
        if constexpr (sizeof...(iOps) > 0) {
            constexpr bool nextIsReadBack = isReadBackType<Next>;
            constexpr bool iOpsContainsReadBack = (isReadBackType<IOps> || ...);
            constexpr bool nextIsComputeOrMidWrite = isComputeType<Next> || isMidWriteType<Next>;
            if constexpr (nextIsReadBack || (nextIsComputeOrMidWrite && iOpsContainsReadBack)) {
                return fuseBack(fuse(read, nextOp), iOps...);
            } else {
                return buildOperationPipeline(read, nextOp, iOps...);
            }
        } else {
            static_assert(isWriteType<Next>, "Last IOp must be WriteType");
            return buildOperationPipeline(read, nextOp);
        }
    }

} // namespace fk
#endif // NVRTC_ENABLED

#endif // FK_JIT_EXECUTOR_DETAILS_H
