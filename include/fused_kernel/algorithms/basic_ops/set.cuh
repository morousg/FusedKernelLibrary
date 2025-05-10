/* Copyright 2024-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_SET
#define FK_SET

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>

namespace fk {

    template <typename T>
    struct ReadSetParams {
        T value;
        ActiveThreads size;
    };

    template <typename T>
    struct ReadSet {
        using Parent = ReadOperation<T, ReadSetParams<T>, T, TF::DISABLED, ReadSet<T>>;
        DECLARE_READ_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params) {
            return params.value;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.size.x;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.size.y;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return opData.params.size.z;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

        FK_HOST_FUSE auto build(const T& value, const ActiveThreads& activeThreads) {
            return InstantiableType{ OperationDataType{{ value, activeThreads }} };
        }
    };
} // namespace fk

#endif
