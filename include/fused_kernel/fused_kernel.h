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

    template <enum TF TFEN, typename... Args>
    inline void executeOperations(Args&... args) {
        using Executor_t = Executor<TransformDPP<defaultParArch, TFEN>>;
        Executor_t::executeOperations(args...);
    }

    template <typename... Args>
    inline void executeOperations(Args&... args) {
        using Executor_t = Executor<TransformDPP<defaultParArch>>;
        Executor_t::executeOperations(args...);
    }

    template <typename DPPType, typename... Args>
    inline void executeOperations(Args&... args) {
        using Executor_t = Executor<DPPType>;
        Executor_t::executeOperations(args...);
    }

} // namespace fk

#endif
