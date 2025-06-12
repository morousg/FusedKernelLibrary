/* Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_FUSED_KERNEL
#define FK_FUSED_KERNEL

#include <fused_kernel/core/execution_model/executors.h>

namespace fk {

    template <typename DPPType, typename... Args>
    inline void executeOperations(Stream_<DPPType::PAR_ARCH>& stream, const Args&... args) {
        Executor<DPPType>::executeOperations(stream, args...);
    }
    template <typename DPPType, enum ND D, typename I, typename... Args>
    inline void executeOperations(const Ptr<D, I>& input, Stream_<DPPType::PAR_ARCH>& stream, const Args&... args) {
        Executor<DPPType>::executeOperations(input, stream, args...);
    }
    template <typename DPPType, enum ND D, typename I, typename O, typename... Args>
    inline void executeOperations(const Ptr<D, I>& input, const Ptr<D, O>& output,
                                  Stream_<DPPType::PAR_ARCH>& stream, const Args&... args) {
        Executor<DPPType>::executeOperations(input, output, stream, args...);
    }
    template <typename DPPType, typename I, size_t BATCH, typename... Args>
    inline void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, const int& activeBatch, const I& defaultValue,
                                  Stream_<DPPType::PAR_ARCH>& stream, const Args&... args) {
        Executor<DPPType>::executeOperations(input, activeBatch, defaultValue, stream, args...);
    }
    template <typename DPPType, typename I, size_t BATCH, typename... Args>
    inline void executeOperations(const std::array<Ptr2D<I>, BATCH>& input,
                                  Stream_<DPPType::PAR_ARCH>& stream, const Args&... args) {
        Executor<DPPType>::executeOperations(input, stream, args...);
    }
    template <typename DPPType, typename I, typename O, size_t BATCH, typename... Args>
    inline void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, const int& activeBatch, const I& defaultValue,
                                  const Tensor<O>& output, Stream_<DPPType::PAR_ARCH>& stream, const Args&... args) {
        Executor<DPPType>::executeOperations(input, activeBatch, defaultValue, output, stream, args...);
    }
    template <typename DPPType, typename I, typename O, size_t BATCH, typename... Args>
    inline void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, const Tensor<O>& output,
                                  Stream_<DPPType::PAR_ARCH>& stream, const Args&... args) {
        Executor<DPPType>::executeOperations(input, output, stream, args...);
    }
} // namespace fk

#endif
