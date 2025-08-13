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

#ifndef FK_DEINTERLACE
#define FK_DEINTERLACE

#include <fused_kernel/core/data/size.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/algorithms/image_processing/interpolation.h>

namespace fk {
    enum class DeinterlaceType {
        BLEND = 2,
        BOB = 3,
        WEAVE = 4
    };

    template <enum DeinterlaceType DEINT_T>
    struct DeinterlaceParameters {};

    template <>
    struct DeinterlaceParameters<DeinterlaceType::BLEND> {
        // No additional parameters needed for simple blend
    };

    template <>
    struct DeinterlaceParameters<DeinterlaceType::BOB> {
        Size src_size;
    };

    template <>
    struct DeinterlaceParameters<DeinterlaceType::WEAVE> {
        // No additional parameters needed for weave
    };

    template <enum DeinterlaceType DEINT_T, typename BackFunction_ = void>
    struct Deinterlace;

    template <typename BackFunction_>
    struct Deinterlace<DeinterlaceType::BLEND, BackFunction_> {
    private:
        using SelfType = Deinterlace<DeinterlaceType::BLEND, BackFunction_>;
        using ReadOutputType = typename BackFunction_::Operation::OutputType;
    public:
        FK_STATIC_STRUCT(Deinterlace, SelfType)
        using Parent = TernaryOperation<Point, DeinterlaceParameters<DeinterlaceType::BLEND>,
                                        BackFunction_, VectorType_t<float, cn<ReadOutputType>>,
                                        Deinterlace<DeinterlaceType::BLEND, BackFunction_>>;
        DECLARE_TERNARY_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params, const BackFunction& back_function) {
            const int x = input.x;
            const int y = input.y;

            // For blend deinterlacing, we blend the current line with adjacent lines
            const Point current_point(x, y);
            const Point prev_point(x, y - 1);
            const Point next_point(x, y + 1);

            const BackFunction readIOp = back_function;
            using ReadOperation = typename BackFunction::Operation;

            const ReadOutputType current_pixel = ReadOperation::exec(current_point, readIOp);
            const ReadOutputType prev_pixel = ReadOperation::exec(prev_point, readIOp);
            const ReadOutputType next_pixel = ReadOperation::exec(next_point, readIOp);

            // Simple blend: average of current line with adjacent lines
            return (current_pixel + prev_pixel + next_pixel) * (1.0f / 3.0f);
        }
    };

    template <typename BackFunction_>
    struct Deinterlace<DeinterlaceType::BOB, BackFunction_> {
    private:
        using SelfType = Deinterlace<DeinterlaceType::BOB, BackFunction_>;
        using ReadOutputType = typename BackFunction_::Operation::OutputType;
    public:
        FK_STATIC_STRUCT(Deinterlace, SelfType)
        using Parent = TernaryOperation<Point, DeinterlaceParameters<DeinterlaceType::BOB>,
                                        BackFunction_, VectorType_t<float, cn<ReadOutputType>>,
                                        Deinterlace<DeinterlaceType::BOB, BackFunction_>>;
        DECLARE_TERNARY_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params, const BackFunction& back_function) {
            const int x = input.x;
            const int y = input.y;

            // For bob deinterlacing, we use interpolation for missing lines
            const float src_x = static_cast<float>(x);
            const float src_y = static_cast<float>(y) * 0.5f; // Scale by 0.5 for field to frame conversion

            const float2 interpolation_point = { src_x, src_y };
            
            // Use linear interpolation for bob deinterlacing
            return Interpolate<InterpolationType::INTER_LINEAR, BackFunction_>::exec(
                interpolation_point, { params.src_size }, back_function);
        }
    };

    template <DeinterlaceType DEINT_T>
    struct Deinterlace<DEINT_T, void> {
    private:
        using SelfType = Deinterlace<DEINT_T, void>;
    public:
        FK_STATIC_STRUCT(Deinterlace, SelfType)
        template <typename RealBackFunction>
        FK_HOST_DEVICE_FUSE
            auto build(const OperationData<Deinterlace<DEINT_T, RealBackFunction>>& opData) {
            return Deinterlace<DEINT_T, RealBackFunction>::build(opData);
        }
    };
} // namespace fk
#endif