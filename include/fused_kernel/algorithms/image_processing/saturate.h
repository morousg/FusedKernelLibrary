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

#ifndef FK_SATURATE
#define FK_SATURATE

#include <cmath>

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/core/utils/vlimits.h>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>

namespace fk {

    template <typename I, typename O>
    struct SaturateCastBase {
    private:
        using SelfType = SaturateCastBase<I, O>;
        using Parent = UnaryOperation<I, O, SelfType>;
    public:
        FK_STATIC_STRUCT(SaturateCastBase, SelfType)
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            constexpr auto maxValOutput = maxValue<OutputType>;
            constexpr auto minValueOutput = minValue<OutputType>;
            if (cxp::cmp_greater(input, maxValOutput)) {
                return maxValOutput;
            } else if (cxp::cmp_less(input, minValueOutput)) {
                return minValueOutput;
            } else {
                // We know that the value of input is within the
                // numerical range of OutputType.
                if constexpr (std::is_floating_point_v<I> && std::is_integral_v<O>) {
                    // For floating point to integral conversion, we need to round
                    return static_cast<OutputType>(cxp::round(input));
                } else {
                    // For any other case, we can cast directly
                    return static_cast<OutputType>(input);
                }
            }
        }
    };

    template <typename I, typename O>
    struct SaturateCast {
    private:
        using SelfType = SaturateCast<I, O>;
    public:
        FK_STATIC_STRUCT(SaturateCast, SelfType)
        using Parent = UnaryOperation<I, O, SaturateCast<I, O>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return UnaryV<SaturateCastBase<VBase<I>, VBase<O>>, I, O>::exec(input);
        }
    };

    struct SaturateFloatBase {
        FK_STATIC_STRUCT(SaturateFloatBase, SaturateFloatBase)
        using Parent = UnaryOperation<float, float, SaturateFloatBase>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return Max<float, float, float, UnaryType>::exec({ 0.f, Min<float,float,float,UnaryType>::exec({ input, 1.f }) });
        }
    };

    template <typename T>
    struct Saturate {
    private:
        using SelfType = Saturate<T>;
    public:
        FK_STATIC_STRUCT(Saturate, SelfType)
        using Parent = BinaryOperation<T, VectorType_t<VBase<T>, 2>, T, Saturate<T>>;
        DECLARE_BINARY_PARENT
        using Base = typename VectorTraits<T>::base;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            static_assert(!validCUDAVec<T>, "Saturate only works with non cuda vector types");
            return Max<Base>::exec(Min<Base>::exec(input, { params.y }), { params.x });
        }
    };

    template <typename T>
    struct SaturateFloat {
    private:
        using SelfType = SaturateFloat<T>;
    public:
        FK_STATIC_STRUCT(SaturateFloat, SelfType)
        using Parent = UnaryOperation<T, T, SaturateFloat<T>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(std::is_same_v<VBase<T>, float>, "Saturate float only works with float base types.");
            return UnaryV<SaturateFloatBase, T, T>::exec(input);
        }
    };

} // namespace fk

#endif
