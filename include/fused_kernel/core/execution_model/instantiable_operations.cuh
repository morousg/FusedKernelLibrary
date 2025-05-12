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

#ifndef FK_INSTANTIABLE_OPERATIONS
#define FK_INSTANTIABLE_OPERATIONS

#include <vector_types.h>
#include <fused_kernel/core/execution_model/operation_tuple.cuh>
#include <fused_kernel/core/execution_model/thread_fusion.cuh>
#include <fused_kernel/core/utils/parameter_pack_utils.cuh>
#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/data/size.h>
#include <fused_kernel/core/data/array.cuh>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.cuh>
#include <fused_kernel/core/execution_model/vector_operations.cuh>
#include <fused_kernel/algorithms/basic_ops/cast_base.cuh>

namespace fk { // namespace FusedKernel
    struct ActiveThreads {
        uint x, y, z;
        FK_HOST_DEVICE_CNST ActiveThreads(const uint& vx = 1,
            const uint& vy = 1,
            const uint& vz = 1) : x(vx), y(vy), z(vz) {}
    };

#define DEVICE_FUNCTION_DETAILS_IS(instance_type) \
    using Operation = Operation_t; \
    using InstanceType = instance_type; \
    template <typename IT> \
    static constexpr bool is{ std::is_same_v<IT, InstanceType> };

#define DEVICE_FUNCTION_DETAILS_IS_ASSERT(instance_type) \
    DEVICE_FUNCTION_DETAILS_IS(instance_type) \
    static_assert(std::is_same_v<typename Operation::InstanceType, instance_type>, "Operation is not " #instance_type );

    // Helper template to check for existence of static constexpr int BATCH 
    template <typename T, typename = void>
    struct has_batch : std::false_type {};
    template <typename T>
    struct has_batch<T, std::void_t<decltype(T::BATCH)>> : std::is_integral<decltype(T::BATCH)> {};
    // Helper template to check for existence of type alias Operation
    template <typename T, typename = void>
    struct has_operation : std::false_type {};
    template <typename T>
    struct has_operation<T, std::void_t<typename T::Operation>> : std::true_type {};
    // Combine checks into a single struct
    template <typename T> struct IsBatchOperation :
        std::integral_constant<bool, has_batch<T>::value&& has_operation<T>::value> {};
    // Helper variable template
    template <typename T>
    static constexpr bool isBatchOperation = IsBatchOperation<T>::value;

    enum PlanePolicy { PROCESS_ALL = 0, CONDITIONAL_WITH_DEFAULT = 1 };

    struct Fuser;

    template <typename Operation_t>
    struct ReadInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadType)

        template <typename ContinuationIOp, typename Fuser_t = Fuser>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            return Fuser_t::fuse(*this, cIOp);
        }

        FK_HOST_DEVICE_CNST ActiveThreads getActiveThreads() const {
            return Operation::getActiveThreads(*this);
        }
    };

    template <typename Operation_t>
    struct ReadBackInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadBackType)

        template <typename ContinuationIOp, typename Fuser_t = Fuser>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            return Fuser_t::fuse(*this, cIOp);
        }

        FK_HOST_DEVICE_CNST ActiveThreads getActiveThreads() const {
            return Operation::getActiveThreads(*this);
        }
    };

    /**
    * @brief BinaryInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
    * (which will reside on GPU registers) and an additional parameter that contains data not generated during the execution
    * of the current kernel.
    * It generates an output and returns it in register memory.
    * It can be composed of a single Operation or of a chain of Operations, in which case it wraps them into an
    * FusedOperation.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input, const OperationData<Operation_t>& opDat)
    */
    template <typename Operation_t>
    struct BinaryInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(BinaryType)

        template <typename ContinuationIOp, typename Fuser_t = Fuser>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            return Fuser_t::fuse(*this, cIOp);
        }

        template <typename ContinuationIOp, typename... ContinuationIOps>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp, const ContinuationIOps&... cIOps) const {
            return then(cIOp).then(cIOps...);
        }
    };

    /**
    * @brief TernaryInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
    * (which will reside on GPU registers) plus two additional parameters.
    * Second parameter (params): represents the same as in a BinaryFunction, data thas was not generated during the execution
    * of this kernel.
    * Third parameter (back_function): it's a IOp that will be used at some point in the implementation of the
    * Operation. It can be any kind of IOp.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input, const OperationData<Operation_t>& opData)
    */
    template <typename Operation_t>
    struct TernaryInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(TernaryType)

        template <typename ContinuationIOp, typename Fuser_t = Fuser>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            return Fuser_t::fuse(*this, cIOp);
        }

        template <typename ContinuationIOp, typename... ContinuationIOps>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp, const ContinuationIOps&... cIOps) const {
            return then(cIOp).then(cIOps...);
        }
    };

    /**
    * @brief UnaryInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
    * (which will reside on GPU registers).
    * It allows to execute the Operation (or chain of Unary Operations) on the input, and returns the result as output
    * in register memory.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input)
    */
    template <typename Operation_t>
    struct UnaryInstantiableOperation {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(UnaryType)

        template <typename ContinuationIOp, typename Fuser_t = Fuser>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            return Fuser_t::fuse(*this, cIOp);
        }

        template <typename ContinuationIOp, typename... ContinuationIOps>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp, const ContinuationIOps&... cIOps) const {
            return then(cIOp).then(cIOps...);
        }
    };

    /**
    * @brief MidWriteInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
    * (which will reside on GPU registers) and writes it into device memory. The way that the data is written, is definded
    * by the implementation of Operation_t.
    * It returns the input data without modification, so that another IOp can be executed after it, using the same data.
    */
    template <typename Operation_t>
    struct MidWriteInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS(MidWriteType)
            static_assert(std::is_same_v<typename Operation::InstanceType, WriteType> ||
                std::is_same_v<typename Operation::InstanceType, MidWriteType>,
                "Operation is not WriteType or MidWriteType");

        template <typename ContinuationIOp, typename Fuser_t = Fuser>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            return Fuser_t::fuse(*this, cIOp);
        }

        template <typename ContinuationIOp, typename... ContinuationIOps>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp, const ContinuationIOps&... cIOps) const {
            return then(cIOp).then(cIOps...);
        }
    };

    /**
    * @brief WriteInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
    * (which will reside on GPU registers) and writes it into device memory. The way that the data is written, is definded
    * by the implementation of Operation_t.
    * It can only be the last IOp in a sequence of InstantiableOperations.
    */
    template <typename Operation_t>
    struct WriteInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(WriteType)
    };

#undef DEVICE_FUNCTION_DETAILS_IS_ASSERT
#undef IS_ASSERT
#undef DEVICE_FUNCTION_DETAILS_IS
#undef DEVICE_FUNCTION_DETAILS
#undef ASSERT
#undef IS

    template <typename Operation>
    using Read = ReadInstantiableOperation<Operation>;
    template <typename Operation>
    using ReadBack = ReadBackInstantiableOperation<Operation>;
    template <typename Operation>
    using Unary = UnaryInstantiableOperation<Operation>;
    template <typename Operation>
    using Binary = BinaryInstantiableOperation<Operation>;
    template <typename Operation>
    using Ternary = TernaryInstantiableOperation<Operation>;
    template <typename Operation>
    using MidWrite = MidWriteInstantiableOperation<Operation>;
    template <typename Operation>
    using Write = WriteInstantiableOperation<Operation>;

    template <typename Operation, typename Enabler = void>
    struct InstantiableOperationType;

    // Single Operation cases
    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isReadType<Operation>>> {
        using type = Read<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isReadBackType<Operation>>> {
        using type = ReadBack<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isUnaryType<Operation>>> {
        using type = Unary<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isBinaryType<Operation>>> {
        using type = Binary<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isTernaryType<Operation>>> {
        using type = Ternary<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isWriteType<Operation>>> {
        using type = Write<Operation>;
    };

    template <typename Operation>
    using Instantiable = typename InstantiableOperationType<Operation>::type;

    // PARENT OPERATIONS
    template <typename RT, typename P, typename O, enum class TF TFE, typename ROperationImpl, bool IS_FUSED = false>
    struct ReadOperation {
        using Child = ROperationImpl;
        using ParamsType = P;
        using ReadDataType = RT;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ static_cast<bool>(TFE) };
        using OutputType = O;
        using OperationDataType = OperationData<ROperationImpl>;
        using InstantiableType = Read<ROperationImpl>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;

        template <uint ELEMS_PER_THREAD = 1, typename CH = Child>
        FK_DEVICE_FUSE ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType>
            exec(const Point& thread, const OperationDataType& opData) {
            if constexpr (std::bool_constant<THREAD_FUSION>::value) {
                return ROperationImpl::template exec<ELEMS_PER_THREAD>(thread, opData.params);
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
    };

#define DECLARE_READ_PARENT_BASIC \
using ParamsType = typename Parent::ParamsType; \
using ReadDataType = typename Parent::ReadDataType; \
using InstanceType = typename Parent::InstanceType; \
using OutputType = typename Parent::OutputType; \
using OperationDataType = typename Parent::OperationDataType; \
using InstantiableType = typename Parent::InstantiableType; \
static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP; \
static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION; \
template <uint ELEMS_PER_THREAD=1> \
FK_DEVICE_FUSE ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType> \
exec(const Point& thread, const OperationDataType& opData) { \
    if constexpr (std::bool_constant<THREAD_FUSION>::value) { \
        return Parent::template exec<ELEMS_PER_THREAD>(thread, opData); \
    } else { \
        return Parent::exec(thread, opData); \
    } \
} \
FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) { \
    return Parent::build(opData); \
} \
FK_HOST_DEVICE_FUSE auto build(const ParamsType& params) { \
    return Parent::build(params); \
}

    template <typename I, typename P, typename WT, enum class TF TFE, typename WOperationImpl, bool IS_FUSED = false>
    struct WriteOperation {
        using Child = WOperationImpl;
        using ParamsType = P;
        using InputType = I;
        using WriteDataType = WT;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ static_cast<bool>(TFE) };
        using OperationDataType = OperationData<WOperationImpl>;
        using InstantiableType = Write<WOperationImpl>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<InputType, ELEMS_PER_THREAD, InputType>& input, const OperationDataType& opData) {
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
    };

#define DECLARE_WRITE_PARENT_BASIC \
using ParamsType = typename Parent::ParamsType; \
using InputType = typename Parent::InputType; \
using WriteDataType = typename Parent::WriteDataType; \
using InstanceType = typename Parent::InstanceType; \
using OperationDataType = typename Parent::OperationDataType; \
using InstantiableType = typename Parent::InstantiableType; \
static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP; \
static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION; \
template <uint ELEMS_PER_THREAD=1> \
FK_HOST_DEVICE_FUSE void exec(const Point& thread, \
                              const ThreadFusionType<InputType, ELEMS_PER_THREAD, InputType>& input, \
                              const OperationDataType& opData) { \
    Parent::template exec<ELEMS_PER_THREAD>(thread, input, opData); \
} \
FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) { \
    return Parent::build(opData); \
} \
FK_HOST_DEVICE_FUSE auto build(const ParamsType& params) { \
    return Parent::build(params); \
}
    // END PARENT OPERATIONS

    // BATCH OPERATIONS
    struct BatchOperation {
        template <typename InstantiableType>
        FK_HOST_FUSE auto toArray(const InstantiableType& batchIOp) {
            static_assert(isBatchOperation<typename InstantiableType::Operation>,
                "The IOp passed as parameter is not a batch operation");
            constexpr size_t BATCH = InstantiableType::Operation::BATCH;
            return toArray_helper(std::make_index_sequence<BATCH>{}, batchIOp);
        }
        template <typename Operation, size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N>& firstInstance,
            const ArrayTypes&... arrays) {
            static_assert(allArraysSameSize_v<BATCH_N, ArrayTypes...>,
                "Not all arrays have the same size as BATCH");
            return build_helper_generic<Operation>(std::make_index_sequence<BATCH_N>(), firstInstance, arrays...);
        }
    private:
        template <typename InstantiableType, size_t... Idx>
        FK_HOST_FUSE auto toArray_helper(const std::index_sequence<Idx...>&, const InstantiableType& batchIOp) {
            using Operation = typename InstantiableType::Operation::Operation;
            using OutputArrayType = std::array<Instantiable<Operation>, sizeof...(Idx)>;
            if constexpr (InstantiableType::template is<ReadType>) {
                return OutputArrayType{ Operation::build(batchIOp.params.opData[Idx])... };
            } else if constexpr (InstantiableType::template is<WriteType>) {
                return OutputArrayType{ Operation::build(batchIOp.params[Idx])... };
            } else {
                static_assert(false, "The IOp passed as parameter is not a batch operation");
            }
        }
        template <size_t Idx, typename Array>
        FK_HOST_FUSE auto get_element_at_index(const Array& paramArray) -> decltype(paramArray[Idx]) {
            return paramArray[Idx];
        }
        template <typename Operation, size_t Idx, typename... Arrays>
        FK_HOST_FUSE auto call_build_at_index(const Arrays&... arrays) {
            return Operation::build(get_element_at_index<Idx>(arrays)...);
        }
        template <typename Operation, size_t... Idx, typename... Arrays>
        FK_HOST_FUSE auto build_helper_generic(const std::index_sequence<Idx...>&,
            const Arrays&... arrays) {
            using OutputArrayType = decltype(call_build_at_index<Operation, 0>(std::declval<Arrays>()...));
            return std::array<OutputArrayType, sizeof...(Idx)>{ call_build_at_index<Operation, Idx>(arrays...)... };
        }
    };

    template <size_t BATCH, enum PlanePolicy PP, typename OpParamsType, typename DefaultType>
    struct BatchReadParams;

    template <size_t BATCH, typename Operation, typename DefaultType>
    struct BatchReadParams<BATCH, CONDITIONAL_WITH_DEFAULT, Operation, DefaultType> {
        OperationData<Operation> opData[BATCH];
        int usedPlanes;
        DefaultType default_value;
        ActiveThreads activeThreads;
    };

    template <size_t BATCH, typename Operation, typename DefaultType>
    struct BatchReadParams<BATCH, PROCESS_ALL, Operation, DefaultType> {
        OperationData<Operation> opData[BATCH];
        ActiveThreads activeThreads;
    };

    template <size_t BATCH, typename Operation>
    struct BatchReadBase {
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationData<Operation>& opData) {
            return Operation::num_elems_x(thread, opData.params.opData[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationData<Operation>& opData) {
            return Operation::num_elems_y(thread, opData.params.opData[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationData<Operation>& opData) {
            return BATCH;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationData<Operation>& opData) {
            return Operation::pitch(thread, opData.params.opData[thread.z]);
        }
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationData<Operation>& opData) {
            return opData.params.activeThreads;
        }
    };

    template <size_t BATCH, enum PlanePolicy PP = PROCESS_ALL, typename Operation = void, typename OutputType = NullType>
    struct BatchRead;

    /// @brief struct BatchRead
    /// @tparam BATCH: number of thread planes and number of data planes to process
    /// @tparam Operation: the read Operation to perform on the data
    /// @tparam PP: enum to select if all planes will be processed equally, or only some
    /// with the remainder not reading and returning a default value
    template <size_t BATCH_, typename Operation_, typename OutputType_>
    struct BatchRead<BATCH_, PROCESS_ALL, Operation_, OutputType_> final :
        public BatchReadBase<BATCH_, BatchRead<BATCH_, PROCESS_ALL, Operation_, OutputType_>> {
        using Operation = Operation_;
        static constexpr size_t BATCH = BATCH_;
        static constexpr PlanePolicy PP = PROCESS_ALL;
        using Parent = ReadOperation<typename Operation::ReadDataType,
            BatchReadParams<BATCH, PP, Operation, typename Operation::OutputType>,
            typename Operation::OutputType,
            Operation::THREAD_FUSION ? TF::ENABLED : TF::DISABLED,
            BatchRead<BATCH, PROCESS_ALL, Operation_, OutputType_>>;
        DECLARE_READ_PARENT_BASIC

        static_assert(isAnyReadType<Operation>, "The Operation is not of any Read type");

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE const auto exec(const Point& thread, const ParamsType& params) {
            return exec_helper<ELEMS_PER_THREAD>(thread, params.opData);
        }
        // Build BatchRead from an array of InstantiableOperations
        template <typename IOp>
        FK_HOST_FUSE std::enable_if_t<isAnyReadType<IOp>, InstantiableType>
            build(const std::array<IOp, BATCH>& instantiableOperations) {
            static_assert(isAnyReadType<IOp>);
            return build_helper(instantiableOperations, std::make_integer_sequence<int, BATCH>{});
        }
        // Build BatchRead from arrays of Operation::build() parameters
        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE std::enable_if_t<!isAnyReadType<FirstType>, InstantiableType>
            build(const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes&... arrays) {
            const auto arrayOfIOps = Operation::build_batch(firstInstance, arrays...);
            return BatchRead<BATCH_N, PP>::build(arrayOfIOps);
        }
    private:
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE const auto exec_helper(const Point& thread,
            const OperationData<Operation>(&opData)[BATCH]) {
            if constexpr (THREAD_FUSION) {
                return Operation::exec<ELEMS_PER_THREAD>(thread, opData[thread.z]);
            } else {
                return Operation::exec(thread, opData[thread.z]);
            }
        }
        template <int... Idx>
        FK_HOST_FUSE InstantiableType
            build_helper(const std::array<Instantiable<Operation>, BATCH>& instantiableOperations,
                const std::integer_sequence<int, Idx...>&) {
            const uint max_width =
                cxp::max(Operation::num_elems_x(Point(0u, 0u, 0u), instantiableOperations[Idx])...);
            const uint max_height =
                cxp::max(Operation::num_elems_y(Point(0u, 0u, 0u), instantiableOperations[Idx])...);
            return { {{{static_cast<OperationData<Operation>>(instantiableOperations[Idx])...}, ActiveThreads{max_width, max_height, BATCH}}} };
        }
    };

    template <typename T>
    using NullTypeToUchar = std::conditional_t<std::is_same_v<T, NullType>, uchar, T>;
    template <typename Nullable, typename Alternative>
    using NullTypeToAlternative = std::conditional_t<std::is_same_v<Nullable, NullType>, Alternative, Nullable>;

    template <size_t BATCH_, typename Operation_, typename OutputType_>
    struct BatchRead<BATCH_, CONDITIONAL_WITH_DEFAULT, Operation_, OutputType_> final :
        public BatchReadBase<BATCH_, BatchRead<BATCH_, CONDITIONAL_WITH_DEFAULT, Operation_, OutputType_>> {
        using Operation = Operation_;
        static constexpr int BATCH = BATCH_;
        static constexpr PlanePolicy PP = CONDITIONAL_WITH_DEFAULT;
        using Parent = ReadOperation<NullTypeToUchar<typename Operation::ReadDataType>,
            BatchReadParams<BATCH, PP, Operation, NullTypeToAlternative<typename Operation::OutputType, OutputType_>>,
            NullTypeToAlternative<typename Operation::OutputType, OutputType_>,
            Operation::THREAD_FUSION ? TF::ENABLED : TF::DISABLED,
            BatchRead<BATCH, PP, Operation, OutputType_>>;
        DECLARE_READ_PARENT_BASIC
        static_assert(isAnyReadType<Operation>, "The Operation is not of any Read type");
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE auto exec(const Point& thread, const ParamsType& params) {
            if (params.usedPlanes <= thread.z) {
                return params.default_value;
            } else {
                return Operation::exec(thread, params.opData[thread.z]);
            }
        }

        template <typename IOp, typename DefaultValueType>
        FK_HOST_FUSE auto build(const std::array<IOp, BATCH>& instantiableOperations,
            const int& usedPlanes, const DefaultValueType& defaultValue) {
            static_assert(isAnyReadType<IOp>);
            if constexpr (std::is_same_v<OutputType, NullType>) {
                return BatchRead<BATCH, PP, typename IOp::Operation, DefaultValueType>::build_helper(instantiableOperations, usedPlanes, defaultValue,
                    std::make_integer_sequence<int, BATCH>{});
            } else {
                return build_helper(instantiableOperations, usedPlanes, defaultValue, std::make_integer_sequence<int, BATCH>{});
            }
        }
        template <size_t BATCH_N, typename DefaultValueType, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue,
            const std::array<FirstType, BATCH_N>& firstInstance,
            const ArrayTypes&... arrays) {
            const auto arrayOfIOps = Operation::build_batch(firstInstance, arrays...);
            return BatchRead<BATCH_N, PP>::build(arrayOfIOps, usedPlanes, defaultValue);
        }
    private:
        template <int... Idx>
        FK_HOST_FUSE InstantiableType build_helper(const std::array<Instantiable<Operation>, BATCH>& instantiableOperations,
            const int& usedPlanes, const OutputType& defaultValue,
            const std::integer_sequence<int, Idx...>&) {
            const uint max_width =
                cxp::max(Operation::num_elems_x(Point(0u, 0u, 0u), instantiableOperations[Idx])...);
            const uint max_height =
                cxp::max(Operation::num_elems_y(Point(0u, 0u, 0u), instantiableOperations[Idx])...);
            return { {{{static_cast<OperationData<Operation>>(instantiableOperations[Idx])...}, usedPlanes, defaultValue, ActiveThreads{max_width, max_height, BATCH}}} };
        }
    };

    template <size_t BATCH>
    struct BatchRead<BATCH, PROCESS_ALL, void> {
        template <typename IOp>
        FK_HOST_FUSE auto build(const std::array<IOp, BATCH>& instantiableOperations) {
            return BatchRead<BATCH, PROCESS_ALL, typename IOp::Operation>::build(instantiableOperations);
        }
    };

    template <size_t BATCH>
    struct BatchRead<BATCH, CONDITIONAL_WITH_DEFAULT, void> {
        template <typename IOp, typename DefaultValueType>
        FK_HOST_FUSE auto build(const std::array<IOp, BATCH>& instantiableOperations,
            const int& usedPlanes, const DefaultValueType& defaultValue) {
            if constexpr (std::is_same_v<typename IOp::Operation::OutputType, NullType>) {
                return BatchRead<BATCH, CONDITIONAL_WITH_DEFAULT, typename IOp::Operation, DefaultValueType>::build(instantiableOperations, usedPlanes, defaultValue);
            } else {
                return BatchRead<BATCH, CONDITIONAL_WITH_DEFAULT, typename IOp::Operation>::build(instantiableOperations, usedPlanes, defaultValue);
            }
        }
    };

    template <size_t BATCH, typename Operation = void>
    struct BatchWrite {
        using Parent = WriteOperation<typename Operation::InputType,
            typename Operation::ParamsType[BATCH],
            typename Operation::WriteDataType,
            Operation::THREAD_FUSION ? TF::ENABLED : TF::DISABLED,
            BatchWrite<BATCH, Operation>>;
        DECLARE_WRITE_PARENT_BASIC

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread,
            const ThreadFusionType<InputType, ELEMS_PER_THREAD, InputType>& input,
            const ParamsType& params) {
            if constexpr (THREAD_FUSION) {
                Operation::exec<ELEMS_PER_THREAD>(thread, input, params[thread.z]);
            } else {
                Operation::exec(thread, input, params[thread.z]);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return Operation::num_elems_x(thread, opData.params[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return Operation::pitch(thread, opData.params[thread.z]);
        }
        // Build WriteBatch from array of IOps
        template <typename IOp>
        FK_HOST_FUSE std::enable_if_t<isWriteType<IOp>, InstantiableType>
            build(const std::array<IOp, BATCH>& iOps) {
            static_assert(isWriteType<IOp>, "The IOps in the array are not WriteType");
            return build_helper(iOps, std::make_integer_sequence<int, BATCH>{});
        }
        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE std::enable_if_t<!isWriteType<FirstType>, InstantiableType>
            build(const std::array<FirstType, BATCH_N>& firstInstance,
                const ArrayTypes&... arrays) {
            const auto arrayOfIOps = Operation::build_batch(firstInstance, arrays...);
            return BatchWrite<BATCH_N>::build(arrayOfIOps);
        }
    private:
        template <int... Idx>
        FK_HOST_FUSE InstantiableType build_helper(const std::array<Instantiable<Operation>, BATCH>& iOps,
            const std::integer_sequence<int, Idx...>&) {
            return { {{(iOps[Idx].params)...}} };
        }
    };

    template <size_t BATCH>
    struct BatchWrite<BATCH, void> {
        using InstaceType = WriteType;
        template <typename IOp>
        FK_HOST_FUSE auto build(const std::array<IOp, BATCH>& iOps) {
            return BatchWrite<BATCH, typename IOp::Operation>::build(iOps);
        }
    };

    template <typename Parent>
    struct ReadOperationBatchBuilders {
        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes&... arrays) {
            return BatchRead<BATCH_N, PROCESS_ALL, typename Parent::Child>::build(firstInstance, arrays...);
        }
        template <size_t BATCH_N, typename DefaultValueType, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue,
            const std::array<FirstType, BATCH_N>& firstInstance,
            const ArrayTypes&... arrays) {
            return BatchRead<BATCH_N, CONDITIONAL_WITH_DEFAULT, typename Parent::Child>::build(usedPlanes, defaultValue, firstInstance, arrays...);
        }
        template <size_t BATCH_N, typename DefaultValueType, typename FirstType>
        FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue,
            const std::array<FirstType, BATCH_N>& firstInstance) {
            if constexpr (isAnyReadType<FirstType>) {
                return BatchRead<BATCH_N, CONDITIONAL_WITH_DEFAULT, typename Parent::Child>::build(firstInstance, usedPlanes, defaultValue);
            } else if constexpr (!isAnyReadType<FirstType>) {
                return BatchRead<BATCH_N, CONDITIONAL_WITH_DEFAULT, typename Parent::Child>::build(usedPlanes, defaultValue, firstInstance);
            } else {
                static_assert(false, "BatchRead: FirstType is not a valid read type");
            }
        }
    };

#define DECLARE_READ_PARENT_BATCH \
template <size_t BATCH_N, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N>& firstInstance, \
                              const ArrayTypes&... arrays) { \
    return BatchOperation::build_batch<typename Parent::Child>(firstInstance, arrays...); \
} \
template <size_t BATCH_N, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes&... arrays) { \
    return ReadOperationBatchBuilders<Parent>::build(firstInstance, arrays...); \
} \
template <size_t BATCH_N, typename DefaultValueType, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue, \
                        const std::array<FirstType, BATCH_N>& firstInstance, \
                        const ArrayTypes&... arrays) { \
    return ReadOperationBatchBuilders<Parent>::build(usedPlanes, defaultValue, firstInstance, arrays...); \
} \
template <size_t BATCH_N, typename DefaultValueType, typename FirstType> \
FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue, \
                        const std::array<FirstType, BATCH_N>& firstInstance) { \
    return ReadOperationBatchBuilders<Parent>::build(usedPlanes, defaultValue, firstInstance); \
}

#define DECLARE_READ_PARENT \
DECLARE_READ_PARENT_BASIC \
DECLARE_READ_PARENT_BATCH

#define DECLARE_WRITE_PARENT_BATCH \
template <size_t BATCH_N, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N>& firstInstance, \
    const ArrayTypes&... arrays) { \
    return BatchOperation::build_batch<typename Parent::Child>(firstInstance, arrays...); \
} \
template <size_t BATCH_N, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance, \
    const ArrayTypes&... arrays) { \
    return BatchWrite<BATCH_N, typename Parent::Child>::build(firstInstance, arrays...); \
}
#define DECLARE_WRITE_PARENT \
DECLARE_WRITE_PARENT_BASIC \
DECLARE_WRITE_PARENT_BATCH

    template <typename RT, typename P, typename B, typename O, typename RBOperationImpl, bool IS_FUSED = false>
    struct ReadBackOperation {
        using Child = RBOperationImpl;
        using ReadDataType = RT;
        using OutputType = O;
        using ParamsType = P;
        using BackFunction = B;
        using InstanceType = ReadBackType;
        using OperationDataType = OperationData<RBOperationImpl>;
        using InstantiableType = ReadBackInstantiableOperation<RBOperationImpl>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;
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
static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP; \
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

#define DECLARE_READBACK_PARENT_INCOMPLETE \
DECLARE_READBACK_PARENT_ALIAS \
FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) { \
    return Parent::build(opData); \
} \
FK_HOST_DEVICE_FUSE auto build(const ParamsType& params, const BackFunction& back_function) { \
    return Parent::build(params, back_function); \
} \
DECLARE_READBACK_PARENT_BATCH_INCOMPLETE
    // END BATCH OPERATIONS
    // PARENT COMPUTE OPERATIONS
    template <typename I, typename O, typename UOperationImpl, bool IS_FUSED = false>
    struct UnaryOperation {
        using InputType = I;
        using OutputType = O;
        using InstanceType = UnaryType;
        using InstantiableType = UnaryInstantiableOperation<UOperationImpl>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;
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
static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP; \
FK_HOST_DEVICE_FUSE InstantiableType build() { \
    return Parent::build(); \
}

    template <typename I, typename P, typename O, typename BOperationImpl, bool IS_FUSED = false>
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
        static constexpr bool IS_FUSED_OP = IS_FUSED;
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
static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP; \
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
    template <typename I, typename P, typename BF, typename O, typename TOperationImpl, bool IS_FUSED = false>
    struct TernaryOperation {
        using InputType = I;
        using OutputType = O;
        using ParamsType = P;
        using BackFunction = BF;
        using InstanceType = TernaryType;
        using OperationDataType = OperationData<TOperationImpl>;
        using InstantiableType = TernaryInstantiableOperation<TOperationImpl>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) {
            return TOperationImpl::exec(input, opData.params, opData.back_function);
        }
        FK_HOST_DEVICE_FUSE InstantiableType build(const OperationDataType& opData) {
            return InstantiableType{ opData };
        }
        FK_HOST_DEVICE_FUSE InstantiableType build(const ParamsType& params, const BackFunction& backFunc) {
            return InstantiableType{ {params, backFunc} };
        }
    };

#define DECLARE_TERNARY_PARENT \
using InputType = typename Parent::InputType; \
using OutputType = typename Parent::OutputType; \
using ParamsType = typename Parent::ParamsType; \
using BackFunction = typename Parent::BackFunction; \
using InstanceType = typename Parent::InstanceType; \
using OperationDataType = typename Parent::OperationDataType; \
using InstantiableType = typename Parent::InstantiableType; \
static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP; \
FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) { \
    return Parent::exec(input, opData); \
} \
FK_HOST_DEVICE_FUSE InstantiableType build(const OperationDataType& opData) { \
    return Parent::build(opData); \
} \
FK_HOST_DEVICE_FUSE InstantiableType build(const ParamsType& params, const BackFunction& backFunc) { \
    return Parent::build(params, backFunc); \
}
    // END PARENT COMPUTE OPERATIONS
} // namespace fk

#endif
