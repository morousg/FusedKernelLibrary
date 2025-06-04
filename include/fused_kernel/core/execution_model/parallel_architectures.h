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

#ifndef PARALLEL_ARCHITECTURES_H
#define PARALLEL_ARCHITECTURES_H

namespace fk {

    enum class ParArch {
        CPU,
        GPU_NVIDIA,
        GPU_NVIDIA_JIT,
        GPU_AMD,
        CPU_OMP,
        CPU_OMPSS,
        MULTI_GPU_NVIDIA,
        CLUSTER_GPU_NVIDIA,
        None
    };

#if defined(__NVCC__) || defined(__HIP__)
    constexpr ParArch defaultParArch = ParArch::GPU_NVIDIA;
#else
    constexpr ParArch defaultParArch = ParArch::CPU;
#endif

} // namespace fk

#endif