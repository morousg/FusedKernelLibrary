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

#ifndef FK_LOGICAL
#define FK_LOGICAL

#include <fused_kernel/core/execution_model/vector_operations.cuh>
#include <fused_kernel/core/data/tuple.cuh>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.cuh>
#include <fused_kernel/core/execution_model/default_operations.cuh>

namespace fk {
    enum ShiftDirection { Left, Right };

    template <typename T, ShiftDirection SD>
    struct ShiftBase final : BinaryOperation<T, uint, T, ShiftBase<T, SD>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            static_assert(!validCUDAVec<T>, "Shift can't work with cuda vector types.");
            static_assert(std::is_unsigned_v<T>, "Shift only works with unsigned integers.");
            if constexpr (SD == Left) {
                return input << params;
            } else if constexpr (SD == Right) {
                return input >> params;
            }
        }
        using Parent = BinaryOperation<T, uint, T, ShiftBase<T, SD>>;
        BINARY_PARENT_FUNCTIONS
    };

    template <typename T, ShiftDirection SD>
    struct Shift final : public BinaryOperation<T, uint, T, Shift<T, SD>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return BinaryV<ShiftBase<VBase<T>, SD>, T, uint>::exec(input, { params });
        }
        using Parent = BinaryOperation<T, uint, T, Shift<T, SD>>;
        BINARY_PARENT_FUNCTIONS
    };

    template <typename T>
    using ShiftLeft = Shift<T, ShiftDirection::Left>;
    template <typename T>
    using ShiftRight = Shift<T, ShiftDirection::Right>;

    template <typename I>
    struct IsEven final : public UnaryOperation<I, bool, IsEven<I>> {
        using AcceptedTypes = TypeList<uchar, ushort, uint, ulong, ulonglong>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(one_of_v<I, AcceptedTypes>, "Input type not valid for UnaryIsEven");
            return (input & 1u) == 0;
        }
        using Parent = UnaryOperation<I, bool, IsEven<I>>;
        UNARY_PARENT_FUNCTIONS
    };

    template <typename I, typename P = I, typename O = I, typename IType = BinaryType>
    struct MaxBase final : public BinaryOperation<I, P, O, MaxBase<I, P, O, BinaryType>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>,
                "Max_ can't work with cuda vector types.");
            return cxp::max(input, params);
        }
        using Parent = BinaryOperation<I, P, O, MaxBase<I, P, O, BinaryType>>;
        BINARY_PARENT_FUNCTIONS
    };

    template <typename I, typename P, typename O>
    struct MaxBase<I, P, O, UnaryType> final : public UnaryOperation<Tuple<I, P>, O, MaxBase<I, P, O, UnaryType>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>,
                "Max_ can't work with cuda vector types.");
            return cxp::max(get<0>(input), get<1>(input));
        }
        using Parent = UnaryOperation<Tuple<I, P>, O, MaxBase<I, P, O, UnaryType>>;
        UNARY_PARENT_FUNCTIONS
    };

    template <typename I, typename P = I, typename O = I, typename IType = BinaryType>
    struct Max final : public BinaryOperation<I, P, O, Max<I, P, O, BinaryType>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return BinaryV<MaxBase<VBase<I>, VBase<P>, VBase<O>>, I, P, O>::exec(input, params);
        }
        using Parent = BinaryOperation<I, P, O, Max<I, P, O, BinaryType>>;
        BINARY_PARENT_FUNCTIONS
    };

    template <typename I, typename P, typename O>
    struct Max<I, P, O, UnaryType> final : public UnaryOperation<Tuple<I, P>, O, Max<I, P, O, UnaryType>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return UnaryV<MaxBase<VBase<I>, VBase<P>, VBase<O>, UnaryType>, I, O>::exec(input);
        }
        using Parent = UnaryOperation<Tuple<I, P>, O, Max<I, P, O, UnaryType>>;
        UNARY_PARENT_FUNCTIONS
    };

    template <typename I, typename P = I, typename O = I, typename IType = BinaryType>
    struct MinBase final : public BinaryOperation<I, P, O, MinBase<I, P, O, BinaryType>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>,
                "Min_ can't work with cuda vector types.");
            return cxp::min(input, params);
        }
        using Parent = BinaryOperation<I, P, O, MinBase<I, P, O, BinaryType>>;
        BINARY_PARENT_FUNCTIONS
    };

    template <typename I, typename P, typename O>
    struct MinBase<I, P, O, UnaryType> final : public UnaryOperation<Tuple<I, P>, O, MinBase<I, P, O, UnaryType>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>,
                "Min_ can't work with cuda vector types.");
            return cxp::min(get<0>(input), get<1>(input));
        }
        using Parent = UnaryOperation<Tuple<I, P>, O, MinBase<I, P, O, UnaryType>>;
        UNARY_PARENT_FUNCTIONS
    };

    template <typename I, typename P = I, typename O = I, typename IType = BinaryType>
    struct Min final : public BinaryOperation<I, P, O, Min<I, P, O, BinaryType>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return BinaryV<MinBase<VBase<I>, VBase<P>, VBase<O>>, I, P, O>::exec(input, params);
        }
        using Parent = BinaryOperation<I, P, O, Min<I, P, O, BinaryType>>;
        BINARY_PARENT_FUNCTIONS
    };

    template <typename I, typename P, typename O>
    struct Min<I, P, O, UnaryType> final : public UnaryOperation<Tuple<I, P>, O, Min<I, P, O, UnaryType>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return UnaryV<MinBase<VBase<I>, VBase<P>, VBase<O>, UnaryType>, I, O>::exec(input);
        }
        using Parent = UnaryOperation<Tuple<I, P>, O, Min<I, P, O, UnaryType>>;
        UNARY_PARENT_FUNCTIONS
    };

    template <typename I1, typename I2=I1>
    struct Equal final : public UnaryOperation<Tuple<I1, I2>, bool, Equal<I1, I2>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return get<0>(input) == get<1>(input);
        }
        using Parent = UnaryOperation<Tuple<I1, I2>, bool, Equal<I1, I2>>;
        UNARY_PARENT_FUNCTIONS
    };
} //namespace fk

#endif
