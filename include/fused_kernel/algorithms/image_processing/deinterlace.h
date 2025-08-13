/* Copyright 2025 Oscar Amoros Huguet

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
        BLEND = 0, 
        INTER_LINEAR = 1
    };

    template <enum DeinterlaceType DEINT_T>
    struct DeinterlaceParameters {};

    template <>
    struct DeinterlaceParameters<DeinterlaceType::BLEND> {
        Size src_size;
    };

    template <>
    struct DeinterlaceParameters<DeinterlaceType::INTER_LINEAR> {
        Size src_size;
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
                                        BackFunction_, ReadOutputType,
                                        Deinterlace<DeinterlaceType::BLEND, BackFunction_>>;
        DECLARE_TERNARY_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params, const BackFunction& back_function) {
            const int x = input.x;
            const int y = input.y;
            
            const Size srcSize = params.src_size;
            
            // For blend deinterlacing, we blend adjacent lines
            const int y_above = (y > 0) ? (y - 1) : y;
            const int y_below = (y < srcSize.height - 1) ? (y + 1) : y;
            
            const BackFunction readIOp = back_function;
            using ReadOperation = typename BackFunction::Operation;
            
            const ReadOutputType pixel_above = ReadOperation::exec(Point(x, y_above), readIOp);
            const ReadOutputType pixel_below = ReadOperation::exec(Point(x, y_below), readIOp);
            
            // Simple blend: average of adjacent lines
            return (pixel_above + pixel_below) * 0.5f;
        }
    };

    template <typename BackFunction_>
    struct Deinterlace<DeinterlaceType::INTER_LINEAR, BackFunction_> {
    private:
        using SelfType = Deinterlace<DeinterlaceType::INTER_LINEAR, BackFunction_>;
        using ReadOutputType = typename BackFunction_::Operation::OutputType;
    public:
        FK_STATIC_STRUCT(Deinterlace, SelfType)
        using Parent = TernaryOperation<Point, DeinterlaceParameters<DeinterlaceType::INTER_LINEAR>,
                                        BackFunction_, ReadOutputType,
                                        Deinterlace<DeinterlaceType::INTER_LINEAR, BackFunction_>>;
        DECLARE_TERNARY_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params, const BackFunction& back_function) {
            const int x = input.x;
            const int y = input.y;
            
            const Size srcSize = params.src_size;
            
            // For inter-linear deinterlacing, we use linear interpolation between lines
            const int y_above = (y > 0) ? (y - 1) : y;
            const int y_below = (y < srcSize.height - 1) ? (y + 1) : y;
            
            const BackFunction readIOp = back_function;
            using ReadOperation = typename BackFunction::Operation;
            
            const ReadOutputType pixel_above = ReadOperation::exec(Point(x, y_above), readIOp);
            const ReadOutputType pixel_below = ReadOperation::exec(Point(x, y_below), readIOp);
            const ReadOutputType pixel_current = ReadOperation::exec(Point(x, y), readIOp);
            
            // Linear interpolation with current line consideration
            if (y_above == y_below) {
                // Edge case: use current pixel
                return pixel_current;
            } else {
                // Interpolate between above and below lines
                const float weight = 0.5f;
                return pixel_above * (1.0f - weight) + pixel_below * weight;
            }
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