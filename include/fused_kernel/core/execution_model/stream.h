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

    template <enum ParArch PA>
    class BaseStream {
    public:
        virtual ~BaseStream() = default;
        virtual void sync() = 0;
        constexpr inline enum ParArch getParArch() const {
            return PA;
        };
        static constexpr inline enum ParArch parArch() {
            return PA;
        }
    };

#if defined(__NVCC__) || defined(__HIP__)
    template <enum ParArch PA = ParArch::GPU_NVIDIA>
    class Stream_;
    template <>
    class Stream_<ParArch::GPU_NVIDIA> final : public BaseStream<ParArch::GPU_NVIDIA> {
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
    };
#else
    template <enum ParArch PA = ParArch::CPU>
    class Stream_;
#endif

    template <>
    class Stream_<ParArch::CPU> final : public BaseStream<ParArch::CPU> {
    public:
        Stream_<ParArch::CPU>() {}
        ~Stream_<ParArch::CPU>() {}
        inline void sync() final {}
    };

    using Stream = Stream_<>;
} // namespace fk

#endif
