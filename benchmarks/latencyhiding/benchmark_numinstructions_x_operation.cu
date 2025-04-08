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

#include <tests/utils/main.h>
#include <tests/utils/fkTestsCommon.h>
#include <tests/utils/twoExecutionsBenchmark.h>

#include <fused_kernel/fused_kernel.cuh>
#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>
#include <fused_kernel/algorithms/basic_ops/static_loop.cuh>

constexpr char VARIABLE_DIMENSION_NAME[]{ "Number of instructions per Operation" };

constexpr size_t NUM_EXPERIMENTS = 100;
constexpr size_t FIRST_VALUE = 1;
constexpr size_t INCREMENT = 5;

constexpr std::array<size_t, NUM_EXPERIMENTS> variableDimensionValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

constexpr int NUM_ELEMENTS = 3840 * 2160 * 8;

constexpr int TOTAL_INSTRUCTIONS = 500;
constexpr std::string_view FIRST_LABEL{ "Separated Ops" };
constexpr std::string_view SECOND_LABEL{ "All Ops" };

constexpr float3 init_val_input{ 1.f, 2.f ,3.f };
constexpr float3 init_val_output{ 0.f, 0.f ,0.f };
constexpr float3 add_val{ 1.f, 1.f, 1.f };
constexpr auto singleOp = fk::Add<float3>::build(add_val);
constexpr auto allInstr = fk::StaticLoop<typename decltype(singleOp)::Operation, TOTAL_INSTRUCTIONS>::build(add_val);

template <typename T>
__global__ void init_values(const T val, fk::RawPtr<fk::_1D, T> pointer_to_init) {
    const int x = threadIdx.x + (blockDim.x * blockIdx.x);
    if (x < pointer_to_init.dims.width) {
        *fk::PtrAccessor<fk::_1D>::point(fk::Point(x), pointer_to_init) = val;
    }
}

template <typename InputType, typename OutputType, size_t NumOps, typename IOp>
struct VerticalFusion {
    static inline void execute(const fk::Ptr1D<InputType>& input, const cudaStream_t& stream,
        const fk::Ptr1D<OutputType>& output, const IOp& dFunc) {
        const fk::ActiveThreads activeThreads{ output.ptr().dims.width };
        fk::Read<fk::PerThreadRead<fk::_1D, InputType>> readDF{ {input.ptr()} };
        using Loop = fk::Binary<fk::StaticLoop<fk::StaticLoop<typename IOp::Operation, INCREMENT>, NumOps / INCREMENT>>;
        Loop loop;
        loop.params = dFunc.params;

        fk::executeOperations<false>(stream, readDF, loop, fk::Write<fk::PerThreadWrite<fk::_1D, OutputType>>{ {output.ptr()} });
    }
};

template <int Idx>
inline bool testNumInstPerOp(cudaStream_t& stream, const fk::Ptr1D<float3>& inputFirst,
                                                   const fk::Ptr1D<float3>& inputSecond,
                                                   const fk::Ptr1D<float3>& outputFirst, 
                                                   const fk::Ptr1D<float3>& outputSecond) {
    // Hack to make the benchmark macros work
    constexpr auto BATCH = variableDimensionValues[Idx];
    // End of hack

    constexpr auto manyInst = fk::StaticLoop<typename decltype(singleOp)::Operation, variableDimensionValues[Idx]>::build(add_val);

    constexpr bool exactDivision = (TOTAL_INSTRUCTIONS % variableDimensionValues[Idx]) == 0;

    const dim3 block(256);
    const dim3 grid(ceil(NUM_ELEMENTS / (float)block.x));
    init_values<<<grid, block, 0, stream>>>(init_val_input, inputFirst.ptr());
    init_values<<<grid, block, 0, stream>>>(init_val_input, inputSecond.ptr());
    init_values<<<grid, block, 0, stream>>>(init_val_output, outputFirst.ptr());
    init_values<<<grid, block, 0, stream>>>(init_val_output, outputSecond.ptr());

    const auto readDF = fk::PerThreadRead<fk::_1D, float3>::build(inputFirst);
    const auto readDF2 = fk::PerThreadRead<fk::_1D, float3>::build(inputSecond);
    const auto writeDF = fk::PerThreadWrite<fk::_1D, float3>::build(outputFirst.ptr());
    const auto writeDF2 = fk::PerThreadWrite<fk::_1D, float3>::build(outputSecond.ptr());

    if constexpr (exactDivision) {
        START_FIRST_BENCHMARK
            // Executing as many kernels as Operations made of variableDimensionValues[Idx] number of instructions,
            // for a total number of instructions equal to TOTAL_INSTRUCTIONS
            for (int i = 0; i < TOTAL_INSTRUCTIONS / variableDimensionValues[Idx]; ++i) {
                fk::executeOperations(stream, readDF, manyInst, writeDF);
            }
        STOP_FIRST_START_SECOND_BENCHMARK
            fk::executeOperations(stream, readDF, allInstr, writeDF);
        STOP_SECOND_BENCHMARK
    } else {
        constexpr int remaining = TOTAL_INSTRUCTIONS % variableDimensionValues[Idx];
        constexpr auto lastOp = fk::StaticLoop<typename decltype(singleOp)::Operation, remaining>::build(add_val);
        START_FIRST_BENCHMARK
            // Executing as many kernels as Operations made of variableDimensionValues[Idx] number of instructions,
            // for a total number of instructions equal to TOTAL_INSTRUCTIONS
            for (int i = 0; i < TOTAL_INSTRUCTIONS / variableDimensionValues[Idx]; ++i) {
                fk::executeOperations(stream, readDF, manyInst, writeDF);
            }
            // Executing the remaining instructions to ger to TOTAL_INSTRUCTIONS
            fk::executeOperations(stream, readDF, lastOp, writeDF);
        STOP_FIRST_START_SECOND_BENCHMARK
            fk::executeOperations(stream, readDF, allInstr, writeDF);
        STOP_SECOND_BENCHMARK
    }

    

    return true;
}

template <int... Idx>
bool testNumInstPerOp_helper(const std::integer_sequence<int, Idx...>&,
                             cudaStream_t& stream,
                             const fk::Ptr1D<float3>& inputFirst,
                             const fk::Ptr1D<float3>& inputSecond,
                             const fk::Ptr1D<float3>& outputFirst,
                             const fk::Ptr1D<float3>& outputSecond) {
    return (testNumInstPerOp<Idx>(stream, inputFirst, inputSecond, outputFirst, outputSecond) && ...);
}

int launch() {
    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    const fk::Ptr1D<float3> inputFirst(NUM_ELEMENTS);
    const fk::Ptr1D<float3> outputFirst(NUM_ELEMENTS);
    const fk::Ptr1D<float3> inputSecond(NUM_ELEMENTS);
    const fk::Ptr1D<float3> outputSecond(NUM_ELEMENTS);
    
    const bool result =
        testNumInstPerOp_helper(std::make_integer_sequence<int, NUM_EXPERIMENTS>{},
            stream,
            inputFirst,
            inputSecond,
            outputFirst,
            outputSecond);


    gpuErrchk(cudaStreamSynchronize(stream));

    return result;
}