/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Grup Mediapro S.L.U

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_WARPING
#define FK_WARPING

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/algorithms/image_processing/interpolation.h>

namespace fk {
    enum WarpType { Affine = 0, Perspective = 1 };

    template<enum WarpType WType>
    struct WarpingParameters;

    template<>
    struct WarpingParameters<Affine> {
        StaticRawPtr<StaticPtrDims2D<3, 2>, float> transformMatrix;
        Size dstSize;
    };

    template<>
    struct WarpingParameters<Perspective> {
        StaticRawPtr<StaticPtrDims2D<3, 3>, float> transformMatrix;
        Size dstSize;
    };

    template <enum WarpType WT>
    struct WarpingCoords {
        using Parent = BinaryOperation<Point, WarpingParameters<WT>, float2, WarpingCoords<WT>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& thread, const ParamsType& params) {
            const int x = thread.x;
            const int y = thread.y;
            const auto& transMatRaw = params.transformMatrix.data;
            if constexpr (WT == WarpType::Perspective) {
                const float coeff = 1.0f / (transMatRaw[2][0] * x + transMatRaw[2][1] * y + transMatRaw[2][2]);

                const float xcoo = coeff * (transMatRaw[0][0] * x + transMatRaw[0][1] * y + transMatRaw[0][2]);
                const float ycoo = coeff * (transMatRaw[1][0] * x + transMatRaw[1][1] * y + transMatRaw[1][2]);

                return make_<float2>(xcoo, ycoo);
            } else {
                const float xcoo = transMatRaw[0][0] * x + transMatRaw[0][1] * y + transMatRaw[0][2];
                const float ycoo = transMatRaw[1][0] * x + transMatRaw[1][1] * y + transMatRaw[1][2];

                return make_<float2>(xcoo, ycoo);
            }
        }
    };

    template<enum WarpType WT, typename BackIOp = void>
    struct Warping {
        using Parent = ReadBackOperation<typename BackIOp::Operation::ReadDataType,
                                         WarpingParameters<WT>,
                                         BackIOp,
                                         VectorType_t<float, cn<typename BackIOp::Operation::ReadDataType>>,
                                         Warping<WT, BackIOp>>;
        DECLARE_READBACK_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            const float2 coord = WarpingCoords<WT>::exec(thread, params);
            const Size sourceSize(BackFunction::Operation::num_elems_x(thread, back_function),
                                  BackFunction::Operation::num_elems_y(thread, back_function));
            if ((coord.x >= 0.f && coord.x < sourceSize.width) && (coord.y >= 0.f && coord.y < sourceSize.height)) {
                return Interpolate<INTER_LINEAR, BackIOp>::exec(coord, {sourceSize}, back_function);
            } else {
                return make_set<OutputType>(0.f);
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }
    };

    template<enum WarpType WT>
    struct Warping<WT, void> {
        using Parent = ReadBackOperation<NullType, WarpingParameters<WT>,
                                         NullType, NullType, Warping<WT, void>>;
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

        FK_HOST_FUSE auto build(const ParamsType& params) {
            return ReadBack<Warping<WT, void>>{{params, {}}};
        }

        template <typename BackIOp>
        FK_HOST_FUSE auto build(const BackIOp& backIOp, const InstantiableType& iOp) {
            return ReadBack<Warping<WT, BackIOp>>{ {iOp.params, backIOp} };
        }
    };
} // namespace fk
#endif
