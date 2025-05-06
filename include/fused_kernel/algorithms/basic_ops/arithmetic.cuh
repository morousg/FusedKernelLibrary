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

#include <fused_kernel/core/execution_model/default_operations.cuh>
#include <fused_kernel/core/utils/cuda_vector_utils.h>

namespace fk {
    template <typename I1, typename I2 = I1, typename O = I1, typename IT = BinaryType>
    struct Add {};

    template <typename I, typename P, typename O>
    struct Add<I, P, O, BinaryType> final : public BinaryOperation<I, P, O, Add<I, P, O, BinaryType>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return input + params;
        }
        using Parent = BinaryOperation<I, P, O, Add<I, P, O, BinaryType>>;
        BINARY_PARENT_FUNCTIONS
    };

    template <typename I1, typename I2, typename O>
    struct Add<I1, I2, O, UnaryType> final : public UnaryOperation<Tuple<I1, I2>, O, Add<I1, I2, O, UnaryType>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return get<0>(input) + get<1>(input);
        }
        using Parent = UnaryOperation<Tuple<I1, I2>, O, Add<I1, I2, O, UnaryType>>;
        UNARY_PARENT_FUNCTIONS
    };

    template <typename I, typename P = I, typename O = I>
    struct Sub final : public BinaryOperation<I, P, O, Sub<I, P, O>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return input - params;
        }
        using Parent = BinaryOperation<I, P, O, Sub<I, P, O>>;
        BINARY_PARENT_FUNCTIONS
    };

    template <typename I, typename P = I, typename O = I>
    struct Mul final : public BinaryOperation<I, P, O, Mul<I, P, O>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return input * params;
        }
        using Parent = BinaryOperation<I, P, O, Mul<I, P, O>>;
        BINARY_PARENT_FUNCTIONS
    };

    template <typename I, typename P = I, typename O = I>
    struct Div final : public BinaryOperation<I, P, O, Div<I, P, O>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return input / params;
        }
        using Parent = BinaryOperation<I, P, O, Div<I, P, O>>;
        BINARY_PARENT_FUNCTIONS
    };
} // namespace fk

#endif
