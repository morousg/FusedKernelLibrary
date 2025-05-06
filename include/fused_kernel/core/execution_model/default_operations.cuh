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

#define UNARY_PARENT_FUNCTIONS \
FK_HOST_DEVICE_FUSE typename Parent::InstantiableType build() { \
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

#define BINARY_PARENT_FUNCTIONS \
FK_HOST_DEVICE_FUSE typename Parent::OutputType exec(const typename Parent::InputType& input, \
                                       const typename Parent::OperationDataType& opData) { \
    return Parent::exec(input, opData); \
} \
FK_HOST_DEVICE_FUSE typename Parent::InstantiableType build(const typename Parent::OperationDataType& opData) { \
    return Parent::build(opData); \
} \
FK_HOST_DEVICE_FUSE typename Parent::InstantiableType build(const typename Parent::ParamsType& params) { \
    return Parent::build(params); \
}

    template <typename ROperationImpl>
    struct ReadOperation {
        using IT = typename ROperationImpl::InputType;
        using OT = typename ROperationImpl::OutputType;
        using PT = typename ROperationImpl::ParamsType;
        using ODT = typename ROperationImpl::OperationDataType;
        using IType = typename ROperationImpl::InstantiableType;

        FK_DEVICE_FUSE OT exec(const Point& thread, const ODT& opData) {
            return ROperationImpl::exec(thread, opData.params);
        }

        FK_HOST_DEVICE_FUSE auto build(const ODT& opData) {
            return IType{ opData };
        }

        FK_HOST_DEVICE_FUSE auto build(const PT& params) {
            return IType{ {params} };
        };

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
    public:
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
    };


} // namespace fk

#endif
