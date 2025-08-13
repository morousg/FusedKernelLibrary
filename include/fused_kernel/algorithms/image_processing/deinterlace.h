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

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/interpolation.h>
#include <fused_kernel/core/data/size.h>

namespace fk {
    enum class DeinterlaceType { BLEND, INTER_LINEAR };

    template <enum DeinterlaceType DType>
    struct DeinterlaceParameters {};

    template <>
    struct DeinterlaceParameters<DeinterlaceType::BLEND> {
        Size src_size;
    };

    template <>
    struct DeinterlaceParameters<DeinterlaceType::INTER_LINEAR> {
        Size src_size;
    };

    template <enum DeinterlaceType DType, typename BackFunction_ = void>
    struct Deinterlace {
    private:
        using SelfType = Deinterlace<DType, BackFunction_>;
    public:
        FK_STATIC_STRUCT(Deinterlace, SelfType)
        using Parent = ReadBackOperation<typename BackFunction_::Operation::ReadDataType,
                                         DeinterlaceParameters<DType>,
                                         BackFunction_,
                                         typename BackFunction_::Operation::ReadDataType,
                                         Deinterlace<DType, BackFunction_>>;
        DECLARE_READBACK_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            if constexpr (DType == DeinterlaceType::BLEND) {
                return exec_blend(thread, params, back_function);
            } else { // INTER_LINEAR
                return exec_inter_linear(thread, params, back_function);
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.src_size.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.src_size.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

        FK_HOST_FUSE InstantiableType build(const BackFunction& backFunction) {
            const Size srcSize = Num_elems<BackFunction>::size(Point(), backFunction);
            const ParamsType deinterlaceParams{ srcSize };
            return { {deinterlaceParams, backFunction} };
        }

    private:
        FK_HOST_DEVICE_FUSE OutputType exec_blend(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            const int x = thread.x;
            const int y = thread.y;
            const int z = thread.z;
            
            // For blend deinterlacing, we average the current line with adjacent lines
            using ReadOperation = typename BackFunction::Operation;
            
            // Read current pixel
            const OutputType current = ReadOperation::exec(Point(x, y, z), back_function);
            
            // For odd lines (field 1), blend with line above
            // For even lines (field 0), blend with line below
            if (y % 2 == 1) {
                // Odd line - blend with line above if available
                if (y > 0) {
                    const OutputType above = ReadOperation::exec(Point(x, y - 1, z), back_function);
                    return (current + above) * 0.5f;
                } else {
                    return current;
                }
            } else {
                // Even line - blend with line below if available
                if (y < params.src_size.height - 1) {
                    const OutputType below = ReadOperation::exec(Point(x, y + 1, z), back_function);
                    return (current + below) * 0.5f;
                } else {
                    return current;
                }
            }
        }

        FK_HOST_DEVICE_FUSE OutputType exec_inter_linear(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            const int x = thread.x;
            const int y = thread.y;
            const int z = thread.z;
            
            using ReadOperation = typename BackFunction::Operation;
            
            // Read current pixel
            const OutputType current = ReadOperation::exec(Point(x, y, z), back_function);
            
            // For inter-linear deinterlacing, we interpolate missing lines
            // Assume we're dealing with interlaced fields where every other line needs interpolation
            if (y % 2 == 1) {
                // Odd lines - interpolate between adjacent even lines
                if (y > 0 && y < params.src_size.height - 1) {
                    const OutputType above = ReadOperation::exec(Point(x, y - 1, z), back_function);
                    const OutputType below = ReadOperation::exec(Point(x, y + 1, z), back_function);
                    return (above + below) * 0.5f;
                } else if (y > 0) {
                    const OutputType above = ReadOperation::exec(Point(x, y - 1, z), back_function);
                    return above;
                } else if (y < params.src_size.height - 1) {
                    const OutputType below = ReadOperation::exec(Point(x, y + 1, z), back_function);
                    return below;
                } else {
                    return current;
                }
            } else {
                // Even lines - keep original
                return current;
            }
        }
    };

    template <enum DeinterlaceType DType>
    struct Deinterlace<DType, void> {
    private:
        using SelfType = Deinterlace<DType, void>;
    public:
        FK_STATIC_STRUCT(Deinterlace, SelfType)
        using Parent = ReadBackOperation<NullType, DeinterlaceParameters<DType>,
                                         NullType, NullType, Deinterlace<DType, void>>;
        DECLARE_READBACK_PARENT_INCOMPLETE

        template <typename BF>
        FK_HOST_FUSE std::enable_if_t<isAnyReadType<BF>, ReadBack<Deinterlace<DType, BF>>>
        build(const BF& backFunction) {
            return Deinterlace<DType, BF>::build(backFunction);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_FUSE auto build(const ParamsType& params) {
            return ReadBack<Deinterlace<DType, void>>{{params, {}}};
        }

        template <typename BackIOp>
        FK_HOST_FUSE auto build(const BackIOp& backIOp, const InstantiableType& iOp) {
            return ReadBack<Deinterlace<DType, BackIOp>>{ {iOp.params, backIOp} };
        }
    };

} // namespace fk

#endif