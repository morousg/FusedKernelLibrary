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

#ifndef FK_CUDA_VECTOR
#define FK_CUDA_VECTOR

#include <fused_kernel/core/execution_model/default_operations.cuh>
#include <fused_kernel/algorithms/basic_ops/logical.cuh>

namespace fk {
    template <typename I, typename O>
    struct Discard final : public UnaryOperation<I, O, Discard<I, O>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(cn<I> > cn<O>, "Output type should at least have one channel less");
            static_assert(std::is_same_v<typename VectorTraits<I>::base,
                typename VectorTraits<O>::base>,
                "Base types should be the same");
            if constexpr (cn<O> == 1) {
                if constexpr (std::is_aggregate_v<O>) {
                    return { input.x };
                } else {
                    return input.x;
                }
            } else if constexpr (cn<O> == 2) {
                return { input.x, input.y };
            } else if constexpr (cn<O> == 3) {
                return { input.x, input.y, input.z };
            }
        }
        using Parent = UnaryOperation<I, O, Discard<I, O>>;
        UNARY_PARENT_FUNCTIONS
    };

    template <typename T, int... Idx>
    struct VectorReorder final : public UnaryOperation<T, T, VectorReorder<T, Idx...>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: UnaryVectorReorder");
            static_assert(cn<T> >= 2, "Minimum number of channels is 2: UnaryVectorReorder");
            return {VectorAt<Idx>(input)...};
        }
        using Parent = UnaryOperation<T, T, VectorReorder<T, Idx...>>;
        UNARY_PARENT_FUNCTIONS
    };

    template <typename T>
    struct VectorReorderRT final : public BinaryOperation<T, VectorType_t<int, cn<T>>, T, VectorReorderRT<T>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type");
            static_assert(cn<T> >= 2, "Minimum number of channels is 2");
            if constexpr (cn<T> == 2) {
                const fk::Array<VBase<T>, 2> temp{ input.x, input.y };
                return { temp.at[params.x], temp.at[params.y] };
            } else if constexpr (cn<T> == 3) {
                const fk::Array<VBase<T>, 3> temp{ input.x, input.y, input.z };
                return { temp.at[params.x], temp.at[params.y], temp.at[params.z] };
            } else {
                const fk::Array<VBase<T>, 4> temp{ input.x, input.y, input.z, input.w };
                return { temp.at[params.x], temp.at[params.y], temp.at[params.z], temp.at[params.w] };
            }
        }
        using Parent = BinaryOperation<T, VectorType_t<int, cn<T>>, T, VectorReorderRT<T>>;
        BINARY_PARENT_FUNCTIONS
    };

    template <typename T, typename Operation>
    struct VectorReduce final : public UnaryOperation<T, VBase<T>, VectorReduce<T, Operation>> { 
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if constexpr (std::is_same_v<typename Operation::InstanceType, UnaryType>) {
                if constexpr (cn<T> == 1) {
                    if constexpr (validCUDAVec<T>) {
                        return input.x;
                    } else {
                        return input;
                    }
                } else if constexpr (cn<T> == 2) {
                    return Operation::exec({ input.x, input.y });
                } else if constexpr (cn<T> == 3) {
                    return Operation::exec({ Operation::exec({ input.x, input.y }), input.z });
                } else if constexpr (cn<T> == 4) {
                    return Operation::exec({ Operation::exec({ Operation::exec({ input.x, input.y }), input.z }), input.w });
                }
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, BinaryType>) {
                if constexpr (cn<T> == 1) {
                    if constexpr (validCUDAVec<T>) {
                        return input.x;
                    } else {
                        return input;
                    }
                } else if constexpr (cn<T> == 2) {
                    return Operation::exec(input.x, input.y);
                } else if constexpr (cn<T> == 3) {
                    return Operation::exec(Operation::exec(input.x, input.y), input.z);
                } else if constexpr (cn<T> == 4) {
                    return Operation::exec(Operation::exec(Operation::exec(input.x, input.y), input.z), input.w);
                }
            }
        }
        using Parent = UnaryOperation<T, T, VectorReduce<T, Operation>>;
        UNARY_PARENT_FUNCTIONS
    };

    template <typename I, typename O>
    struct AddLast final : public BinaryOperation<I, typename VectorTraits<I>::base, O, AddLast<I, O>> {
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            static_assert(cn<I> == cn<O> -1, "Output type should have one channel more");
            static_assert(std::is_same_v<typename VectorTraits<I>::base, typename VectorTraits<O>::base>,
                "Base types should be the same");
            if constexpr (cn<I> == 1) {
                if constexpr (std::is_aggregate_v<I>) {
                    return { input.x, params };
                } else {
                  return {input, params};
                }
            } else if constexpr (cn<I> == 2) {
              return {input.x, input.y, params};
            } else if constexpr (cn<I> == 3) {
              return {input.x, input.y, input.z, params};
            }
        }
        using Parent = BinaryOperation<I, typename VectorTraits<I>::base, O, AddLast<I, O>>;
        BINARY_PARENT_FUNCTIONS
    };

    template <typename T>
    struct VectorAnd final : UnaryOperation<T, T, VectorAnd<T>>{
        static_assert(std::is_same_v<VBase<T>, bool>, "VectorAnd only works with boolean vectors");
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return VectorReduce<T, Equal<bool, bool>>::exec(input);
        }
        using Parent = UnaryOperation<T, T, VectorAnd<T>>;
        UNARY_PARENT_FUNCTIONS
    };
} // namespace fk

#endif
