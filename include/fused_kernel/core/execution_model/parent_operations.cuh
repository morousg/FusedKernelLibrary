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
    template <typename I, typename O, typename UOperationImpl>
    struct UnaryOperation {
        using InputType = I;
        using OutputType = O;
        using InstanceType = UnaryType;
        using InstantiableType = UnaryInstantiableOperation<UOperationImpl>;
        // build() is fine, it only refers to UOperationImpl::InstantiableType
        // within the function body/return type, which is instantiated later.
        FK_HOST_DEVICE_FUSE auto build() {
            return typename UOperationImpl::InstantiableType{};
        }
    };

#define DECLARE_UNARY_PARENT \
using InputType = typename Parent::InputType; \
using OutputType = typename Parent::OutputType; \
using InstanceType = typename Parent::InstanceType; \
using InstantiableType = typename Parent::InstantiableType; \
FK_HOST_DEVICE_FUSE InstantiableType build() { \
    return Parent::build(); \
}

    template <typename I, typename P, typename O, typename BOperationImpl>
    struct BinaryOperation {
        // --- REMOVE using ALIASES that depend on BOperationImpl ---
        // These caused the incomplete type error during base class instantiation.
        // We will refer to BOperationImpl::TypeNeeded directly in methods.
        using InputType = I;
        using OutputType = O; // Needed for the static exec signature
        using ParamsType = P; // Needed by OperationData and build(params)
        using InstanceType = BinaryType;
        using OperationDataType = OperationData<BOperationImpl>; // Needed by exec/build(opData)
        using InstantiableType = BinaryInstantiableOperation<BOperationImpl>;
        // --- exec Method ---
        // Accesses types only in signature/body -> OK
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) {
            // Calls the static exec of the derived class (CRTP)
            return BOperationImpl::exec(input, opData.params);
            // Return type deduced via 'auto'
        }
        // --- build Methods ---
        // Accesses types only in signature/body -> OK
        FK_HOST_DEVICE_FUSE InstantiableType build(const OperationDataType& opData) {
            // Return type deduced via 'auto'
            return InstantiableType{ opData };
        }
        FK_HOST_DEVICE_FUSE InstantiableType build(const ParamsType& params) {
            // Return type deduced via 'auto'
            return InstantiableType{ {params} };
        }
    };

#define DECLARE_BINARY_PARENT \
using InputType = typename Parent::InputType; \
using OutputType = typename Parent::OutputType; \
using ParamsType = typename Parent::ParamsType; \
using InstanceType = typename Parent::InstanceType; \
using OperationDataType = typename Parent::OperationDataType; \
using InstantiableType = typename Parent::InstantiableType; \
FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, \
                                       const OperationDataType& opData) { \
    return Parent::exec(input, opData); \
} \
FK_HOST_DEVICE_FUSE InstantiableType build(const OperationDataType& opData) { \
    return Parent::build(opData); \
} \
FK_HOST_DEVICE_FUSE InstantiableType build(const ParamsType& params) { \
    return Parent::build(params); \
}

    template <typename RT, typename P, typename O, enum class TF TFE, typename ROperationImpl>
    struct ReadOperation {
        using ParamsType = P;
        using ReadDataType = RT;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ static_cast<bool>(TFE) };
        using OutputType = O;
        using OperationDataType = OperationData<ROperationImpl>;
        using InstantiableType = Read<ROperationImpl>;

        template <uint ELEMS_PER_THREAD=1>
        FK_DEVICE_FUSE ThreadFusionType<ReadDataType, ELEMS_PER_THREAD> exec(const Point& thread, const OperationDataType& opData) {
            if constexpr (THREAD_FUSION) {
                return ROperationImpl::exec<ELEMS_PER_THREAD>(thread, opData.params);
            } else {
                return ROperationImpl::exec(thread, opData.params);
            }
        }

        FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) {
            return InstantiableType{ opData };
        }

        FK_HOST_DEVICE_FUSE auto build(const ParamsType& params) {
            return InstantiableType{ {params} };
        };

        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes&... arrays) {

            static_assert(allArraysSameSize_v<BATCH_N, ArrayTypes...>,
                "Not all arrays have the same size as BATCH");
            return build_helper_generic(std::make_index_sequence<BATCH_N>(), firstInstance, arrays...);
        }

        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance,
            const ArrayTypes&... arrays) {
            const auto arrayOfIOps = build_batch(firstInstance, arrays...);
            return BatchRead<BATCH_N>::build(arrayOfIOps);
        }

        template <size_t BATCH_N, typename DefaultValueType, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue,
            const std::array<FirstType, BATCH_N>& firstInstance,
            const ArrayTypes&... arrays) {
            const auto arrayOfIOps = build_batch(firstInstance, arrays...);
            return BatchRead<BATCH_N, CONDITIONAL_WITH_DEFAULT>::build(arrayOfIOps, usedPlanes, defaultValue);
        }
    private:
        template <size_t Idx, typename Array>
        FK_HOST_FUSE auto get_element_at_index(const Array& paramArray) -> decltype(paramArray[Idx]) {
            return paramArray[Idx];
        }
        template <size_t Idx, typename... Arrays>
        FK_HOST_FUSE auto call_build_at_index(const Arrays&... arrays) {

            return build(get_element_at_index<Idx>(arrays)...);
        }
        template <size_t... Idx, typename... Arrays>
        FK_HOST_FUSE auto build_helper_generic(const std::index_sequence<Idx...>&,
                                               const Arrays&... arrays) {
            using OutputArrayType = decltype(call_build_at_index<0>(std::declval<Arrays>()...));
            return std::array<OutputArrayType, sizeof...(Idx)>{ call_build_at_index<Idx>(arrays...)... };
        }
    };

#define DECLARE_READ_PARENT \
using ParamsType = typename Parent::ParamsType; \
using ReadDataType = typename Parent::ReadDataType; \
using InstanceType = typename Parent::InstanceType; \
using OutputType = typename Parent::OutputType; \
using OperationDataType = typename Parent::OperationDataType; \
using InstantiableType = typename Parent::InstantiableType; \
static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION; \
template <uint ELEMS_PER_THREAD=1> \
FK_DEVICE_FUSE ThreadFusionType<ReadDataType, ELEMS_PER_THREAD> exec(const Point& thread, const OperationDataType& opData) { \
    return Parent::template exec<ELEMS_PER_THREAD>(thread, opData); \
} \
FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) { \
    return Parent::build(opData); \
} \
FK_HOST_DEVICE_FUSE auto build(const ParamsType& params) { \
    return Parent::build(params); \
} \
template <size_t BATCH_N, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N>& firstInstance, \
    const ArrayTypes&... arrays) { \
    return Parent::build_batch(firstInstance, arrays...); \
} \
template <size_t BATCH_N, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes&... arrays) { \
    return Parent::build(firstInstance, arrays...); \
} \
template <size_t BATCH_N, typename DefaultValueType, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue, \
    const std::array<FirstType, BATCH_N>& firstInstance, \
    const ArrayTypes&... arrays) { \
    return Parent::build(usedPlanes, defaultValue, firstInstance, arrays...); \
}

    template <typename I, typename P, typename WT, enum class TF TFE, typename WOperationImpl>
    struct WriteOperation {
        using ParamsType = P;
        using InputType = I;
        using WriteDataType = WT;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ static_cast<bool>(TFE) };
        using OperationDataType = OperationData<WOperationImpl>;
        using InstantiableType = Write<WOperationImpl>;
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<InputType, ELEMS_PER_THREAD>& input, const OperationDataType& opData) {
            if constexpr (THREAD_FUSION) {
                WOperationImpl::exec<ELEMS_PER_THREAD>(thread, input, opData.params);
            } else {
                WOperationImpl::exec(thread, input, opData.params);
            }
        }
        FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) {
            return InstantiableType{ opData };
        }
        FK_HOST_DEVICE_FUSE auto build(const ParamsType& params) {
            return InstantiableType{ {params} };
        };
        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N>& firstInstance,
            const ArrayTypes&... arrays) {
            static_assert(allArraysSameSize_v<BATCH_N, ArrayTypes...>, "Not all arrays have the same size as BATCH");
            return build_helper_generic(std::make_index_sequence<BATCH_N>(), firstInstance, arrays...);
        }
        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance,
            const ArrayTypes&... arrays) {
            const auto arrayOfIOps = build_batch(firstInstance, arrays...);
            return BatchWrite<BATCH_N>::build(arrayOfIOps);
        }
    private:
        template <size_t Idx, typename Array>
        FK_HOST_FUSE auto get_element_at_index(const Array& paramArray) -> decltype(paramArray[Idx]) {
        
            return paramArray[Idx]; 
        } 
        template <size_t Idx, typename... Arrays> 
            FK_HOST_FUSE auto call_build_at_index(const Arrays&... arrays) {
            
                return build(get_element_at_index<Idx>(arrays)...); 
        } 
        template <size_t... Idx, typename... Arrays> 
            FK_HOST_FUSE auto build_helper_generic(const std::index_sequence<Idx...>&, 
                const Arrays&... arrays) {
                
                using OutputArrayType = decltype(call_build_at_index<0>(std::declval<Arrays>()...)); 
                return std::array<OutputArrayType, sizeof...(Idx)>{ call_build_at_index<Idx>(arrays...)... };
        }
    };

#define DECLARE_WRITE_PARENT \
using ParamsType = typename Parent::ParamsType; \
using InputType = typename Parent::InputType; \
using WriteDataType = typename Parent::WriteDataType; \
using InstanceType = typename Parent::InstanceType; \
using OperationDataType = typename Parent::OperationDataType; \
using InstantiableType = typename Parent::InstantiableType; \
static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION; \
template <uint ELEMS_PER_THREAD=1> \
FK_HOST_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<InputType, ELEMS_PER_THREAD>& input, const OperationDataType& opData) { \
    Parent::template exec<ELEMS_PER_THREAD>(thread, input, opData); \
} \
FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) { \
    return Parent::build(opData); \
} \
FK_HOST_DEVICE_FUSE auto build(const ParamsType& params) { \
    return Parent::build(params); \
} \
template <size_t BATCH_N, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N>& firstInstance, \
    const ArrayTypes&... arrays) { \
    return Parent::build_batch(firstInstance, arrays...); \
} \
template <size_t BATCH_N, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes&... arrays) { \
    return Parent::build(firstInstance, arrays...); \
}

} // namespace fk

#endif
