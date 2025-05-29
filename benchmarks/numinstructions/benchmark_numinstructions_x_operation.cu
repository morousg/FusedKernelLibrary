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

#include <tests/main.h>
#include <benchmarks/fkBenchmarksCommon.h>
#include <benchmarks/twoExecutionsBenchmark.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/static_loop.h>
#include <fused_kernel/core/execution_model/executors.h>

constexpr char VARIABLE_DIMENSION_NAME[]{ "Number of instructions per Operation" };

constexpr size_t NUM_EXPERIMENTS = 10; // Used 100 in the paper
constexpr size_t FIRST_VALUE = 1;
constexpr size_t INCREMENT = 5;

constexpr std::array<size_t, NUM_EXPERIMENTS> variableDimensionValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

constexpr int NUM_ELEMENTS = 3840 * 2160 * 8;

constexpr int TOTAL_INSTRUCTIONS = 50; // Used 500 and 1000 in the paper
constexpr std::string_view FIRST_LABEL{ "Separated Ops" };
constexpr std::string_view SECOND_LABEL{ "All Ops" };

constexpr float init_val_input{ 1.f };
constexpr float init_val_output{ 0.f };
constexpr float add_val{ 1.f };
constexpr auto singleOp = fk::Add<float>::build(add_val);
constexpr auto allInstr = fk::StaticLoop<typename decltype(singleOp)::Operation, TOTAL_INSTRUCTIONS>::build(add_val);

template <typename T>
__global__ void init_values(const T val, fk::RawPtr<fk::_1D, T> pointer_to_init) {
    const int x = threadIdx.x + (blockDim.x * blockIdx.x);
    if (x < pointer_to_init.dims.width) {
        *fk::PtrAccessor<fk::_1D>::point(fk::Point(x), pointer_to_init) = val;
    }
}

template <int Idx>
inline bool testNumInstPerOp(fk::Stream& stream, fk::Ptr1D<float>& inputFirst,
                                                 fk::Ptr1D<float>& inputSecond,
                                                 fk::Ptr1D<float>& outputFirst,
                                                 fk::Ptr1D<float>& outputSecond) {
    // Hack to make the benchmark macros work
    constexpr auto BATCH = variableDimensionValues[Idx];
    // End of hack

    constexpr auto manyInst = fk::StaticLoop<typename decltype(singleOp)::Operation, variableDimensionValues[Idx]>::build(add_val);

    constexpr bool exactDivision = (TOTAL_INSTRUCTIONS % variableDimensionValues[Idx]) == 0;

    fk::setTo(init_val_input, inputFirst, stream);
    fk::setTo(init_val_input, inputSecond, stream);
    fk::setTo(init_val_output, outputFirst, stream);
    fk::setTo(init_val_output, outputSecond, stream);

    const auto readDF = fk::PerThreadRead<fk::_1D, float>::build(inputFirst);
    const auto readDF2 = fk::PerThreadRead<fk::_1D, float>::build(inputSecond);
    const auto writeDF = fk::PerThreadWrite<fk::_1D, float>::build(outputFirst);
    const auto writeDF2 = fk::PerThreadWrite<fk::_1D, float>::build(outputSecond);

    if constexpr (exactDivision) {
        // Wramming up the GPU
        fk::executeOperations<false>(stream, readDF, manyInst, writeDF);
        fk::executeOperations<false>(stream, readDF, allInstr, writeDF);
        START_FIRST_BENCHMARK
            // Executing as many kernels as Operations made of variableDimensionValues[Idx] number of instructions,
            // for a total number of instructions equal to TOTAL_INSTRUCTIONS
            for (int i = 0; i < TOTAL_INSTRUCTIONS / variableDimensionValues[Idx]; ++i) {
                fk::executeOperations<false>(stream, readDF, manyInst, writeDF);
            }
        STOP_FIRST_START_SECOND_BENCHMARK
            fk::executeOperations<false>(stream, readDF, allInstr, writeDF);
        STOP_SECOND_BENCHMARK
    } else {
        constexpr int remaining = TOTAL_INSTRUCTIONS % variableDimensionValues[Idx];
        constexpr auto lastOp = fk::StaticLoop<typename decltype(singleOp)::Operation, remaining>::build(add_val);
        // Wramming up the GPU
        fk::executeOperations<false>(stream, readDF, manyInst, writeDF);
        fk::executeOperations<false>(stream, readDF, lastOp, writeDF);
        fk::executeOperations<false>(stream, readDF, allInstr, writeDF);
        START_FIRST_BENCHMARK
            // Executing as many kernels as Operations made of variableDimensionValues[Idx] number of instructions,
            // for a total number of instructions equal to TOTAL_INSTRUCTIONS
            for (int i = 0; i < TOTAL_INSTRUCTIONS / variableDimensionValues[Idx]; ++i) {
                fk::executeOperations<false>(stream, readDF, manyInst, writeDF);
            }
            // Executing the remaining instructions to ger to TOTAL_INSTRUCTIONS
            fk::executeOperations<false>(stream, readDF, lastOp, writeDF);
        STOP_FIRST_START_SECOND_BENCHMARK
            fk::executeOperations<false>(stream, readDF, allInstr, writeDF);
        STOP_SECOND_BENCHMARK
    }

    return true;
}

template <int... Idx>
bool testNumInstPerOp_helper(const std::integer_sequence<int, Idx...>&,
                             fk::Stream& stream,
                             fk::Ptr1D<float>& inputFirst,
                             fk::Ptr1D<float>& inputSecond,
                             fk::Ptr1D<float>& outputFirst,
                             fk::Ptr1D<float>& outputSecond) {
    return (testNumInstPerOp<Idx>(stream, inputFirst, inputSecond, outputFirst, outputSecond) && ...);
}

int launch() {
    fk::Stream stream;

    fk::Ptr1D<float> inputFirst(NUM_ELEMENTS);
    fk::Ptr1D<float> outputFirst(NUM_ELEMENTS);
    fk::Ptr1D<float> inputSecond(NUM_ELEMENTS);
    fk::Ptr1D<float> outputSecond(NUM_ELEMENTS);
    
    const bool result =
        testNumInstPerOp_helper(std::make_integer_sequence<int, NUM_EXPERIMENTS>{},
            stream,
            inputFirst,
            inputSecond,
            outputFirst,
            outputSecond);

    stream.sync();

    return result ? 0 : -1;
}