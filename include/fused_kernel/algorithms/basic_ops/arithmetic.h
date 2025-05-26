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

#ifndef FK_ARITHMETIC
#define FK_ARITHMETIC

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/utils/cuda_vector_utils.h>

namespace fk {
    template <typename I1, typename I2 = I1, typename O = I1, typename IT = BinaryType>
    struct Add {};

    template <typename I, typename P, typename O>
    struct Add<I, P, O, BinaryType> {
        using Parent = BinaryOperation<I, P, O, Add<I, P, O, BinaryType>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return input + params;
        }
    };

    template <typename I1, typename I2, typename O>
    struct Add<I1, I2, O, UnaryType> {
        using Parent = UnaryOperation<Tuple<I1, I2>, O, Add<I1, I2, O, UnaryType>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return get<0>(input) + get<1>(input);
        }
    };

    template <typename I, typename P = I, typename O = I>
    struct Sub {
        using Parent = BinaryOperation<I, P, O, Sub<I, P, O>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return input - params;
        }
    };

    template <typename I, typename P = I, typename O = I>
    struct Mul {
        using Parent = BinaryOperation<I, P, O, Mul<I, P, O>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return input * params;
        }
    };

    template <typename I, typename P = I, typename O = I>
    struct Div {
        using Parent = BinaryOperation<I, P, O, Div<I, P, O>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return input / params;
        }
    };
} // namespace fk

#endif
