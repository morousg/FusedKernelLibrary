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

#ifndef FK_OPERATION_DATA_CUH
#define FK_OPERATION_DATA_CUH

#include <type_traits>
#include <fused_kernel/core/execution_model/operation_model/operation_types.cuh>

namespace fk {
    //Operation type traits
    // hasParams trait
    template <typename, typename = std::void_t<>>
    struct hasParams : std::false_type {};

    template <typename T>
    struct hasParams<T, std::void_t<typename T::ParamsType>> : std::true_type {};

    template <typename T>
    constexpr bool hasParams_v = hasParams<T>::value;

    // Primary template: assumes T does not have member 'next'
    template <typename, typename = std::void_t<>>
    struct has_next : std::false_type {};

    // Specialized template: this will be chosen if T has member 'next'
    template <typename T>
    struct has_next<T, std::void_t<decltype(std::declval<T>().next)>> : std::true_type {};

    // Helper variable template for easier usage
    template <typename T>
    constexpr bool has_next_v = has_next<T>::value;

    using BFList = TypeList<ReadBackType, TernaryType>;
    template <typename OpOrDF>
    constexpr bool hasNoBackFunction_v = !one_of_v<typename OpOrDF::InstanceType, BFList>;

    // hasBackFunction trait
    template <typename, typename = std::void_t<>>
    struct hasBackFunction : std::false_type {};

    template <typename T>
    struct hasBackFunction<T, std::void_t<typename T::BackFunction>> : std::true_type {};

    template <typename T>
    constexpr bool hasBackFunction_v = hasBackFunction<T>::value;

    // hasParamsAndBackFunction trait
    template <typename, typename = std::void_t<>>
    struct hasParamsAndBackFunction : std::false_type {};

    template <typename T>
    struct hasParamsAndBackFunction<T, std::void_t<typename T::ParamsType,
        typename T::BackFunction>> : std::true_type {};

    template <typename T>
    constexpr bool hasParamsAndBackFunction_v = hasParamsAndBackFunction<T>::value;

    // OperationData implementation selectors
    template <typename Operation>
    constexpr bool hasParamsNoArray =
        hasParams_v<Operation> && !std::is_array_v<typename Operation::ParamsType>;
    template <typename Operation>
    constexpr bool hasParamsArray =
        hasParams_v<Operation> && std::is_array_v<typename Operation::ParamsType>;

    // OperationData implementations
    template <typename Operation, typename Enabler = void>
    struct OperationData;

    template <typename Operation>
    struct OperationData<Operation, std::enable_if_t<hasParamsNoArray<Operation>&& hasNoBackFunction_v<Operation>, void>> {
#ifdef COPYABLE_IOPS
        FK_HOST_DEVICE_CNST OperationData() {};
        FK_HOST_DEVICE_CNST OperationData(const typename Operation::ParamsType& params_) : params(params_) {}
#endif
        typename Operation::ParamsType params;
    };

    template <typename Operation>
    struct OperationData<Operation, std::enable_if_t<hasParamsArray<Operation>&& hasNoBackFunction_v<Operation>, void>> {
#ifdef COPYABLE_IOPS
        FK_HOST_DEVICE_CNST OperationData() {};
        __host__ __forceinline__ OperationData(const typename Operation::ParamsType& params_) {
            std::copy(std::begin(params_), std::end(params_), std::begin(params));
        }
        __host__ __forceinline__ OperationData<Operation>& operator=(const OperationData<Operation>& other) {
            if (this != &other) {
                std::copy(std::begin(other.params), std::end(other.params), std::begin(params));
            }
            return *this;
        }
#endif
        typename Operation::ParamsType params;
    };

    template <typename Operation>
    struct OperationData<Operation, std::enable_if_t<hasParamsAndBackFunction_v<Operation> &&
        !std::is_array_v<typename Operation::ParamsType> &&
        !std::is_array_v<typename Operation::BackFunction>, void>> {
#ifdef COPYABLE_IOPS
        FK_HOST_DEVICE_CNST OperationData() {};
        FK_HOST_DEVICE_CNST OperationData(const typename Operation::ParamsType& params_,
            const typename Operation::BackFunction& back_function_) :
            params(params_), back_function(back_function_) {}
#endif
        typename Operation::ParamsType params;
        typename Operation::BackFunction back_function;
    };

    template <typename Operation>
    struct OperationData<Operation, std::enable_if_t<hasParamsAndBackFunction_v<Operation> &&
        (std::is_array_v<typename Operation::ParamsType> ||
            std::is_array_v<typename Operation::BackFunction>), void>> {
#ifdef COPYABLE_IOPS
        __host__ __forceinline__ OperationData() {};
        __host__ __forceinline__ OperationData(const typename Operation::ParamsType& params_,
            const typename Operation::BackFunction& back_function_) {
            if constexpr (std::is_array_v<typename Operation::ParamsType>) {
                std::copy(std::begin(params_), std::end(params_), std::begin(params));
            } else {
                params = params_;
            }
            if constexpr (std::is_array_v<typename Operation::BackFunction>) {
                std::copy(std::begin(back_function_), std::end(back_function_), std::begin(back_function));
            } else {
                back_function = back_function_;
            }
        }
        __host__ __forceinline__ OperationData<Operation>& operator=(const OperationData<Operation>& other) {
            if (this != &other) {
                if constexpr (std::is_array_v<typename Operation::ParamsType>) {
                    std::copy(std::begin(other.params), std::end(other.params), std::begin(params));
                } else {
                    params = other.params;
                }
                if constexpr (std::is_array_v<typename Operation::BackFunction>) {
                    std::copy(std::begin(other.back_function), std::end(other.back_function), std::begin(back_function));
                } else {
                    back_function = other.back_function;
                }
            }
            return *this;
        }
#endif
        typename Operation::ParamsType params;
        typename Operation::BackFunction back_function;
    };
} // namespace fk

#endif