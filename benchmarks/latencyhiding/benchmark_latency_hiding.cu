﻿/* Copyright 2024-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <testing/main.h>
#include <testing/fkTestsCommon.h>
#include <testing/fkbenchmark.h>
 

#include <fused_kernel/fused_kernel.cuh>
#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>
#include <fused_kernel/algorithms/basic_ops/static_loop.cuh>

constexpr char VARIABLE_DIMENSION_NAME[]{ "Number of Operations" };

constexpr size_t NUM_EXPERIMENTS = 30;
constexpr size_t FIRST_VALUE = 1;
constexpr size_t INCREMENT = 20;

constexpr std::array<size_t, NUM_EXPERIMENTS> variableDimanesionValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

constexpr int NUM_ELEMENTS = 3840 * 2160 * 8;

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
        using Loop = fk::Binary<fk::StaticLoop<fk::StaticLoop<typename IOp::Operation, INCREMENT>, NumOps/INCREMENT>>;
        Loop loop;
        loop.params = dFunc.params;

        fk::executeOperations<false>(stream, readDF, loop, fk::Write<fk::PerThreadWrite<fk::_1D, OutputType>>{ {output.ptr()} });
    }
};

template <int VARIABLE_DIMENSION>
inline int testLatencyHiding(cudaStream_t stream) {

    const fk::Ptr1D<float3> input(NUM_ELEMENTS);
    const fk::Ptr1D<float3> output(NUM_ELEMENTS);

    constexpr float3 init_val{ 1,2,3 };

    dim3 block(256);
    dim3 grid(ceil(NUM_ELEMENTS / (float)block.x));
    init_values<<<grid, block, 0, stream>>>(init_val, input.ptr());

    using IOp = fk::Binary<fk::Add<float3>>;
    IOp df{ fk::make_set<float3>(2) };

    // Warmup
    VerticalFusion<float3, float3, 1, IOp>::execute(input, stream, output, df);

    START_FK_BENCHMARK

        VerticalFusion<float3, float3, VARIABLE_DIMENSION, IOp>::execute(input, stream, output, df);

    STOP_FK_BENCHMARK

        return 0;
}

template <int... Idx>
inline int testLatencyHidingHelper(cudaStream_t stream, const std::integer_sequence<int, Idx...>& seq) {
    const bool result = ((testLatencyHiding<variableDimanesionValues[Idx]>(stream) == 0) && ...);
    if (result) {
        return 0;
    } else {
        return -1;
    }
}

int launch() {
    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    const int result = testLatencyHidingHelper(stream, std::make_integer_sequence<int, variableDimanesionValues.size()>{});

    CLOSE_BENCHMARK

    return result;

    return 0;
}
