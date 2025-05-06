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

#ifndef FK_CAST
#define FK_CAST

#include <fused_kernel/core/execution_model/default_operations.cuh>
#include <fused_kernel/core/execution_model/vector_operations.cuh>
#include <fused_kernel/algorithms/basic_ops/cast_base.cuh>

namespace fk {
    template <typename I, typename O>
    struct Cast final : UnaryOperation<I, O, Cast<I, O>>{
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return UnaryV<CastBase<VBase<I>, VBase<O>>, I, O>::exec(input);
        }
        using Parent = UnaryOperation<I, O, Cast<I, O>>;
        UNARY_PARENT_FUNCTIONS
    };
} // namespace fk

#endif
