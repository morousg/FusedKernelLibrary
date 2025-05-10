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

   // In this file we have the basic defintion of the Operations

#ifndef FK_DEFAULT_OPERATIONS
#define FK_DEFAULT_OPERATIONS

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>

namespace fk {
// WriteOperation declared in instantiable_operations.cuh
#define DECLARE_WRITE_PARENT_BASIC \
using ParamsType = typename Parent::ParamsType; \
using InputType = typename Parent::InputType; \
using WriteDataType = typename Parent::WriteDataType; \
using InstanceType = typename Parent::InstanceType; \
using OperationDataType = typename Parent::OperationDataType; \
using InstantiableType = typename Parent::InstantiableType; \
static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION; \
template <uint ELEMS_PER_THREAD=1> \
FK_HOST_DEVICE_FUSE void exec(const Point& thread, \
                              const ThreadFusionType<InputType, ELEMS_PER_THREAD>& input, \
                              const OperationDataType& opData) { \
    Parent::template exec<ELEMS_PER_THREAD>(thread, input, opData); \
} \
FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) { \
    return Parent::build(opData); \
} \
FK_HOST_DEVICE_FUSE auto build(const ParamsType& params) { \
    return Parent::build(params); \
}

    template <typename RT, typename P, typename B, typename O, typename RBOperationImpl>
    struct ReadBackOperation {
        using Child = RBOperationImpl;
        using ReadDataType = RT;
        using OutputType = O;
        using ParamsType = P;
        using BackFunction = B;
        using InstanceType = ReadBackType;
        using OperationDataType = OperationData<RBOperationImpl>;
        using InstantiableType = ReadBackInstantiableOperation<RBOperationImpl>;
        static constexpr bool THREAD_FUSION = false;

        template <typename BF = BackFunction>
        FK_DEVICE_FUSE std::enable_if_t<!std::is_same_v<BF, NullType>, OutputType>
        exec(const Point& thread, const OperationDataType& opData) {
            return RBOperationImpl::exec(thread, opData.params, opData.back_function);
        }
        FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) {
            return InstantiableType{ opData };
        }
        FK_HOST_DEVICE_FUSE auto build(const ParamsType& params, const BackFunction& backFunc) {
            return InstantiableType{ { params, backFunc } };
        };
    };

#define DECLARE_READBACK_PARENT_ALIAS \
using ReadDataType = typename Parent::ReadDataType; \
using OutputType = typename Parent::OutputType; \
using ParamsType = typename Parent::ParamsType; \
using BackFunction = typename Parent::BackFunction; \
using InstanceType = typename Parent::InstanceType; \
using OperationDataType = typename Parent::OperationDataType; \
using InstantiableType = typename Parent::InstantiableType; \
static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION;

// DECLARE_READBACK_PARENT
#define DECLARE_READBACK_PARENT \
DECLARE_READBACK_PARENT_ALIAS \
FK_DEVICE_FUSE OutputType exec(const Point& thread, const OperationDataType& opData) { \
    return Parent::exec(thread, opData); \
} \
FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) { \
    return Parent::build(opData); \
} \
FK_HOST_DEVICE_FUSE auto build(const ParamsType& params, const BackFunction& back_function) { \
    return Parent::build(params, back_function); \
} \
DECLARE_READ_PARENT_BATCH

    template <typename Parent>
    struct ReadBackIncompleteOperationBatchBuilders {
        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance,
                                const ArrayTypes&... arrays) {
            const auto arrayOfIOps = BatchOperation::build_batch<typename Parent::Child>(firstInstance, arrays...);
            return BatchRead<BATCH_N>::build(arrayOfIOps);
        }
        template <size_t BATCH_N, typename DefaultValueType, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue,
                                const std::array<FirstType, BATCH_N>& firstInstance,
                                const ArrayTypes&... arrays) {
            const auto arrayOfIOps = BatchOperation::build_batch<typename Parent::Child>(firstInstance, arrays...);
            return BatchRead<BATCH_N, CONDITIONAL_WITH_DEFAULT>::build(arrayOfIOps, usedPlanes, defaultValue);
        }
        template <size_t BATCH_N, typename DefaultValueType, typename FirstType>
        FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue,
                                const std::array<FirstType, BATCH_N>& firstInstance) {
            if constexpr (isAnyReadType<FirstType>) {
                return BatchRead<BATCH_N, CONDITIONAL_WITH_DEFAULT>::build(firstInstance, usedPlanes, defaultValue);
            } else if constexpr (!isAnyReadType<FirstType>) {
                const auto arrayOfIOps = BatchOperation::build_batch<typename Parent::Child>(firstInstance);
                return BatchRead<BATCH_N, CONDITIONAL_WITH_DEFAULT>::build(arrayOfIOps, usedPlanes, defaultValue);
            } else {
                static_assert(false, "BatchRead: FirstType is not a valid read type");
            }
        }
    };
    template <typename Parent>
    using RBIncompleteOpBB = ReadBackIncompleteOperationBatchBuilders<Parent>;

#define DECLARE_READBACK_PARENT_BATCH_INCOMPLETE \
template <size_t BATCH_N, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N>& firstInstance, \
                              const ArrayTypes&... arrays) { \
    return BatchOperation::build_batch<typename Parent::Child>(firstInstance, arrays...); \
} \
template <size_t BATCH_N, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance, \
                        const ArrayTypes&... arrays) { \
    return RBIncompleteOpBB<Parent>::build(firstInstance, arrays...); \
} \
template <size_t BATCH_N, typename DefaultValueType, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue, \
                        const std::array<FirstType, BATCH_N>& firstInstance, \
                        const ArrayTypes&... arrays) { \
    return RBIncompleteOpBB<Parent>::build(usedPlanes, defaultValue, firstInstance, arrays...); \
}\
template <size_t BATCH_N, typename DefaultValueType, typename FirstType> \
FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue, \
                        const std::array<FirstType, BATCH_N>& firstInstance) { \
    return RBIncompleteOpBB<Parent>::build(usedPlanes, defaultValue, firstInstance); \
}

    // DECLARE_READBACK_PARENT_INCOMPLETE
#define DECLARE_READBACK_PARENT_INCOMPLETE \
DECLARE_READBACK_PARENT_ALIAS \
FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) { \
    return Parent::build(opData); \
} \
FK_HOST_DEVICE_FUSE auto build(const ParamsType& params, const BackFunction& back_function) { \
    return Parent::build(params, back_function); \
} \
DECLARE_READBACK_PARENT_BATCH_INCOMPLETE

} // namespace fk

#endif
