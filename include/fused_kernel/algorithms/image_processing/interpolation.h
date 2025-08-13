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

#ifndef FK_INTERPOLATION
#define FK_INTERPOLATION

#include <cmath>

#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/core/data/size.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>

namespace fk {
    template <typename T>
    struct Slice2x2 {
        T _0x0;
        T _1x0;
        T _0x1;
        T _1x1;
    };

    enum class InterpolationType {
        INTER_LINEAR = 1,
        DEINTERLACE_BLEND = 2,
        DEINTERLACE_INTER_LINEAR = 3,
        NONE = 17
    };

    template <enum InterpolationType INTER_T>
    struct InterpolationParameters {};

    template <>
    struct InterpolationParameters<InterpolationType::INTER_LINEAR> {
        Size src_size;
    };

    template <>
    struct InterpolationParameters<InterpolationType::DEINTERLACE_BLEND> {
        Size src_size;
    };

    template <>
    struct InterpolationParameters<InterpolationType::DEINTERLACE_INTER_LINEAR> {
        Size src_size;
        int field_select; // 0 for even rows (interpolate odd), 1 for odd rows (interpolate even)
    };

    template <enum InterpolationType INTER_T, typename BackFunction_ = void>
    struct Interpolate;

    template <typename BackFunction_>
    struct Interpolate<InterpolationType::INTER_LINEAR, BackFunction_> {
    private:
        using SelfType = Interpolate<InterpolationType::INTER_LINEAR, BackFunction_>;
        using ReadOutputType = typename BackFunction_::Operation::OutputType;
    public:
        FK_STATIC_STRUCT(Interpolate, SelfType)
        using Parent = TernaryOperation<float2, InterpolationParameters<InterpolationType::INTER_LINEAR>,
                                        BackFunction_, VectorType_t<float, cn<ReadOutputType>>,
                                        Interpolate<InterpolationType::INTER_LINEAR, BackFunction_>>;
        DECLARE_TERNARY_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params, const BackFunction& back_function) {
            const float src_x = input.x;
            const float src_y = input.y;

#ifdef __CUDA_ARCH__
            const int x1 = __float2int_rd(src_x);
            const int y1 = __float2int_rd(src_y);
#else
            const int x1 = static_cast<int>(std::floor(src_x));
            const int y1 = static_cast<int>(std::floor(src_x));
#endif
            const int x2 = x1 + 1;
            const int y2 = y1 + 1;

            const Size srcSize = params.src_size;
            const int x2_read = Min<int>::exec(x2, { srcSize.width - 1 });
            const int y2_read = Min<int>::exec(y2, { srcSize.height - 1 });

            const Slice2x2<Point> readPoints{ Point(x1, y1),
                                              Point(x2_read, y1),
                                              Point(x1, y2_read),
                                              Point(x2_read, y2_read) };

            const BackFunction readIOp = back_function;
            using ReadOperation = typename BackFunction::Operation;

            const ReadOutputType src_reg0x0 = ReadOperation::exec(readPoints._0x0, readIOp);
            const ReadOutputType src_reg1x0 = ReadOperation::exec(readPoints._1x0, readIOp);
            const ReadOutputType src_reg0x1 = ReadOperation::exec(readPoints._0x1, readIOp);
            const ReadOutputType src_reg1x1 = ReadOperation::exec(readPoints._1x1, readIOp);

            return (src_reg0x0 * ((x2 - src_x) * (y2 - src_y))) +
                   (src_reg1x0 * ((src_x - x1) * (y2 - src_y))) +
                   (src_reg0x1 * ((x2 - src_x) * (src_y - y1))) +
                   (src_reg1x1 * ((src_x - x1) * (src_y - y1)));
        }
    };

    template <typename BackFunction_>
    struct Interpolate<InterpolationType::DEINTERLACE_BLEND, BackFunction_> {
    private:
        using SelfType = Interpolate<InterpolationType::DEINTERLACE_BLEND, BackFunction_>;
        using ReadOutputType = typename BackFunction_::Operation::OutputType;
    public:
        FK_STATIC_STRUCT(Interpolate, SelfType)
        using Parent = TernaryOperation<Point, InterpolationParameters<InterpolationType::DEINTERLACE_BLEND>,
                                        BackFunction_, ReadOutputType,
                                        Interpolate<InterpolationType::DEINTERLACE_BLEND, BackFunction_>>;
        DECLARE_TERNARY_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params, const BackFunction& back_function) {
            const int x = input.x;
            const int y = input.y;
            const Size srcSize = params.src_size;
            
            const BackFunction readIOp = back_function;
            using ReadOperation = typename BackFunction::Operation;
            
            // For the first line (y=0), copy directly from input
            if (y == 0) {
                return ReadOperation::exec(Point(x, 0), readIOp);
            }
            
            // For other lines, blend current line with previous line
            // output_pixel_line1 = (input_pixel_line0 + input_pixel_line1 + 1) / 2
            const ReadOutputType pixel_prev = ReadOperation::exec(Point(x, y - 1), readIOp);
            const ReadOutputType pixel_curr = ReadOperation::exec(Point(x, y), readIOp);
            
            // Blend using (a + b + 1) / 2 for proper rounding with unsigned integer division
            return (pixel_prev + pixel_curr + make_set<ReadOutputType>(1)) / make_set<ReadOutputType>(2);
        }
    };

    template <typename BackFunction_>
    struct Interpolate<InterpolationType::DEINTERLACE_INTER_LINEAR, BackFunction_> {
    private:
        using SelfType = Interpolate<InterpolationType::DEINTERLACE_INTER_LINEAR, BackFunction_>;
        using ReadOutputType = typename BackFunction_::Operation::OutputType;
    public:
        FK_STATIC_STRUCT(Interpolate, SelfType)
        using Parent = TernaryOperation<Point, InterpolationParameters<InterpolationType::DEINTERLACE_INTER_LINEAR>,
                                        BackFunction_, ReadOutputType,
                                        Interpolate<InterpolationType::DEINTERLACE_INTER_LINEAR, BackFunction_>>;
        DECLARE_TERNARY_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params, const BackFunction& back_function) {
            const int x = input.x;
            const int y = input.y;
            const Size srcSize = params.src_size;
            const int field_select = params.field_select;
            
            const BackFunction readIOp = back_function;
            using ReadOperation = typename BackFunction::Operation;
            
            if (field_select == 0) {
                // Use even rows (starting from 0) and interpolate odd rows
                if (y == 0) {
                    // Copy line 0
                    return ReadOperation::exec(Point(x, 0), readIOp);
                } else if (y >= srcSize.height - 2) {
                    // Copy lines n-1 and n for the last rows
                    return ReadOperation::exec(Point(x, y), readIOp);
                } else if (y % 2 == 1) {
                    // Interpolate odd rows: output_pixel_line1 = (input_pixel_line0 + input_pixel_line2 + 1) / 2
                    const ReadOutputType pixel_above = ReadOperation::exec(Point(x, y - 1), readIOp);
                    const ReadOutputType pixel_below = ReadOperation::exec(Point(x, y + 1), readIOp);
                    return (pixel_above + pixel_below + make_set<ReadOutputType>(1)) / make_set<ReadOutputType>(2);
                } else {
                    // Copy even rows directly
                    return ReadOperation::exec(Point(x, y), readIOp);
                }
            } else {
                // Use odd rows and interpolate even rows
                if (y <= 1) {
                    // Copy lines 0 and 1
                    return ReadOperation::exec(Point(x, y), readIOp);
                } else if (y == srcSize.height - 1) {
                    // Copy line n
                    return ReadOperation::exec(Point(x, y), readIOp);
                } else if (y % 2 == 0) {
                    // Interpolate even rows: output_pixel_line2 = (input_pixel_line1 + input_pixel_line3 + 1) / 2
                    const ReadOutputType pixel_above = ReadOperation::exec(Point(x, y - 1), readIOp);
                    const ReadOutputType pixel_below = ReadOperation::exec(Point(x, y + 1), readIOp);
                    return (pixel_above + pixel_below + make_set<ReadOutputType>(1)) / make_set<ReadOutputType>(2);
                } else {
                    // Copy odd rows directly
                    return ReadOperation::exec(Point(x, y), readIOp);
                }
            }
        }
    };

    template <InterpolationType INTER_T>
    struct Interpolate<INTER_T, void> {
    private:
        using SelfType = Interpolate<INTER_T, void>;
    public:
        FK_STATIC_STRUCT(Interpolate, SelfType)
        template <typename RealBackFunction>
        FK_HOST_DEVICE_FUSE
            auto build(const OperationData<Interpolate<InterpolationType::INTER_LINEAR, RealBackFunction>>& opData) {
            return Interpolate<INTER_T, RealBackFunction>::build(opData);
        }
        template <typename RealBackFunction>
        FK_HOST_DEVICE_FUSE
            auto build(const OperationData<Interpolate<InterpolationType::DEINTERLACE_BLEND, RealBackFunction>>& opData) {
            return Interpolate<INTER_T, RealBackFunction>::build(opData);
        }
        template <typename RealBackFunction>
        FK_HOST_DEVICE_FUSE
            auto build(const OperationData<Interpolate<InterpolationType::DEINTERLACE_INTER_LINEAR, RealBackFunction>>& opData) {
            return Interpolate<INTER_T, RealBackFunction>::build(opData);
        }
    };
} // namespace fk
#endif
