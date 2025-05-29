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

#ifndef FK_STREAM
#define FK_STREAM

#include <fused_kernel/core/execution_model/parallel_architectures.h>

#if defined(__NVCC__)
#include <fused_kernel/core/utils/utils.h>
#elif defined(__HIP__)
#include <hip/hip_runtime.h>
#endif

namespace fk {

    class BaseStream {
    public:
        virtual ~BaseStream() = default;
        virtual void sync() = 0;
    };

    template <enum ParArch PA>
    class Stream_;

#if defined(__NVCC__) || defined(__HIP__)
    template <>
    class Stream_<ParArch::GPU_NVIDIA> final : public BaseStream {
        cudaStream_t m_stream;
        bool m_isMine{ false };
    public:
        Stream_<ParArch::GPU_NVIDIA>() {
            gpuErrchk(cudaStreamCreate(&m_stream));
            m_isMine = true;
        }
        // Intentionally not set as explicit, to allow implicit conversion from cudaStream_t
        explicit Stream_<ParArch::GPU_NVIDIA>(const cudaStream_t& stream) : m_stream(stream) {}
        ~Stream_<ParArch::GPU_NVIDIA>() final {
            if (m_stream != 0 && m_isMine) {
                sync();
                gpuErrchk(cudaStreamDestroy(m_stream));
            }
        }
        operator cudaStream_t() const {
            return m_stream;
        }
        inline cudaStream_t getCUDAStream() const {
            return m_stream;
        }
        inline void sync() final {
            gpuErrchk(cudaStreamSynchronize(m_stream));
        }
        constexpr inline enum ParArch getParArch() const {
            return ParArch::GPU_NVIDIA;
        };
        static constexpr inline enum ParArch parArch() {
            return ParArch::GPU_NVIDIA;
        }
    };
#endif

    template <>
    class Stream_<ParArch::CPU> final : public BaseStream {
    public:
        Stream_<ParArch::CPU>() {}
        ~Stream_<ParArch::CPU>() {}
        inline void sync() final {}
        constexpr inline enum ParArch getParArch() const {
            return ParArch::CPU;
        };
        static constexpr inline enum ParArch parArch() {
            return ParArch::CPU;
        }
    };

    using Stream = Stream_<defaultParArch>;
} // namespace fk

#endif
