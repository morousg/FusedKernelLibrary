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

#ifndef FK_CROP_OP
#define FK_CROP_OP

#include <fused_kernel/core/execution_model/parent_operations.cuh>
#include <fused_kernel/core/data/rect.h>
#include <fused_kernel/core/data/point.h>

namespace fk {

    template <typename BackIOp = void>
    struct Crop {
        using Parent = ReadBackOperation<typename BackIOp::Operation::OutputType,
                                         Rect,
                                         BackIOp,
                                         typename BackIOp::Operation::OutputType,
                                         Crop<BackIOp>>;
        DECLARE_READBACK_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            const Point newThread(thread.x + params.x, thread.y + params.y);
            return BackFunction::Operation::exec(newThread, back_function);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

        FK_HOST_FUSE InstantiableType build(const BackFunction& backFunction, const Rect& rect) {
            return InstantiableType{ { rect, backFunction } };
        }
    };

    template <>
    struct Crop<void> {
        using Parent = ReadBackOperation<NullType, Rect, NullType, NullType, Crop<void>>;
        DECLARE_READBACK_PARENT_INCOMPLETE

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_FUSE auto build(const Rect& rectCrop) {
            return InstantiableType{ { rectCrop, {} } };
        }

        template <typename RealBackIOp>
        FK_HOST_FUSE auto build(const RealBackIOp& realBIOp, const InstantiableType& iOp) {
            return Crop<RealBackIOp>::build(realBIOp, iOp.params);
        }
    };

} // namespace fk

#endif