/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

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

#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/algorithms/algorithms.cuh>
#include <fused_kernel/fused_kernel.cuh>

constexpr char VARIABLE_DIMENSION_NAME[]{ "Batch size" };

constexpr size_t NUM_EXPERIMENTS = 10; // Used 100 in the paper
constexpr size_t FIRST_VALUE = 1;
constexpr size_t INCREMENT = 5;

constexpr std::array<size_t, NUM_EXPERIMENTS> variableDimensionValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

template <size_t BATCH>
bool benchmark_Horizontal_Fusion(const size_t& NUM_ELEMS_X, const size_t& NUM_ELEMS_Y, cudaStream_t stream) {
    constexpr std::string_view FIRST_LABEL{ "Iterated Batch" };
    constexpr std::string_view SECOND_LABEL{ "Fused Batch" };
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    using InputType = uchar;
    using OutputType = float;

    const InputType val_init = 10u;
    const OutputType val_alpha = 1.0f;
    const OutputType val_sub = 1.f;
    const OutputType val_div = 3.2f;
    try {
        const fk::Size cropSize(60, 120);
        fk::Ptr2D<InputType> d_input((int)NUM_ELEMS_Y, (int)NUM_ELEMS_X);
        fk::setTo(val_init, d_input, stream);
        std::array<fk::Ptr2D<OutputType>, BATCH> d_output_cv;
        std::array<fk::Ptr2D<OutputType>, BATCH> h_cvResults;
        std::array<fk::Ptr2D<OutputType>, BATCH> h_cvGSResults;

        fk::Tensor<OutputType> d_tensor_output(cropSize.width, cropSize.height, BATCH);
        fk::Tensor<OutputType> h_tensor_output(cropSize.width, cropSize.height, BATCH, 1, fk::MemType::HostPinned);

        std::array<fk::Ptr2D<InputType>, BATCH> crops;
        for (int crop_i = 0; crop_i < BATCH; crop_i++) {
            crops[crop_i] = d_input.crop(fk::Point(crop_i, crop_i), fk::PtrDims<fk::_2D>{static_cast<uint>(cropSize.width),
                                                                                         static_cast<uint>(cropSize.height), 
                                                                                         static_cast<uint>(d_input.dims().pitch)});
            d_output_cv[crop_i].Alloc(cropSize, 0, fk::MemType::Device);
            h_cvResults[crop_i].Alloc(cropSize, 0, fk::MemType::HostPinned);
        }

        START_FIRST_BENCHMARK
        for (int crop_i = 0; crop_i < BATCH; crop_i++) {
            fk::executeOperations(crops[crop_i], stream,
                fk::SaturateCast<InputType, OutputType>::build(),
                fk::Mul<OutputType>::build(val_alpha),
                fk::Sub<OutputType>::build(val_sub),
                fk::Div<OutputType>::build(val_div),
                fk::PerThreadWrite<fk::_2D, OutputType>::build(d_output_cv[crop_i]));
        }
        STOP_FIRST_START_SECOND_BENCHMARK
        fk::executeOperations(crops, stream,
            fk::SaturateCast<InputType, OutputType>::build(),
            fk::Mul<OutputType>::build(val_alpha),
            fk::Sub<OutputType>::build(val_sub),
            fk::Div<OutputType>::build(val_div),
            fk::TensorWrite<OutputType>::build(d_tensor_output));
        STOP_SECOND_BENCHMARK

        d_tensor_output.download(h_tensor_output, stream);

        // Verify results
        for (int crop_i = 0; crop_i < BATCH; crop_i++) {
            d_output_cv[crop_i].download(h_cvResults[crop_i], stream);
        }

        gpuErrchk(cudaStreamSynchronize(stream));

        for (int crop_i = 0; crop_i < BATCH; crop_i++) {
            fk::Ptr2D<OutputType> cvRes = h_cvResults[crop_i];
            fk::Ptr2D<OutputType> cvGSRes = h_tensor_output.getPlane(crop_i);
            bool passedThisTime = compareAndCheck(cvRes, cvGSRes);
            if (!passedThisTime) { std::cout << "Failed on crop idx=" << crop_i << std::endl; }
            passed &= passedThisTime;
        }
    } catch (const std::exception& e) {
        error_s << e.what();
        passed = false;
        exception = true;
    }

    if (!passed) {
        if (!exception) {
            std::stringstream ss;
            ss << "benchark_Horizontal_Fusion";
            std::cout << ss.str() << " failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
        } else {
            std::stringstream ss;
            ss << "benchark_Horizontal_Fusion";
            std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
        }
    }

    return passed;
}

template <size_t BATCH>
bool benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD(const size_t& NUM_ELEMS_X, const size_t& NUM_ELEMS_Y, cudaStream_t stream) {
    constexpr std::string_view FIRST_LABEL{ "Iterated Batch" };
    constexpr std::string_view SECOND_LABEL{ "Fused Batch" };
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    using InputType = uchar;
    using OutputType = float;

    const InputType val_init = 10u;
    const OutputType val_alpha = 1.0f;
    const OutputType val_sub = 1.f;
    const OutputType val_div = 3.2f;
    try {
        const fk::Size cropSize(60, 120);
        fk::Ptr2D<InputType> d_input((int)NUM_ELEMS_Y, (int)NUM_ELEMS_X);
        fk::setTo(val_init, d_input, stream);
        std::array<fk::Ptr2D<OutputType>, BATCH> d_output_cv;
        std::array<fk::Ptr2D<OutputType>, BATCH> h_cvResults;
        std::array<fk::Ptr2D<OutputType>, BATCH> h_cvGSResults;

        fk::Tensor<OutputType> d_tensor_output(cropSize.width, cropSize.height, BATCH);
        fk::Tensor<OutputType> h_tensor_output(cropSize.width, cropSize.height, BATCH, 1, fk::MemType::HostPinned);

        std::array<fk::Ptr2D<InputType>, BATCH> crops;
        for (int crop_i = 0; crop_i < BATCH; crop_i++) {
            crops[crop_i] = d_input.crop(fk::Point(crop_i, crop_i), fk::PtrDims<fk::_2D>{static_cast<uint>(cropSize.width),
                                                                                         static_cast<uint>(cropSize.height), 
                                                                                         static_cast<uint>(d_input.dims().pitch)});
            d_output_cv[crop_i].Alloc(cropSize, 0, fk::MemType::Device);
            h_cvResults[crop_i].Alloc(cropSize, 0, fk::MemType::HostPinned);
        }

        // Read Ops
        const auto read_array = fk::PerThreadRead<fk::_2D, InputType>::build_batch(crops);
        const auto read = fk::PerThreadRead<fk::_2D, InputType>::build(crops);

        // Compute Ops
        const auto saturate = fk::SaturateCast<InputType, OutputType>::build();
        const auto mul = fk::Mul<OutputType>::build(val_alpha);
        const auto sub = fk::Sub<OutputType>::build(val_sub);
        const auto div = fk::Div<OutputType>::build(val_div);

        // Write Ops
        const auto write_array = fk::PerThreadWrite<fk::_2D, OutputType>::build_batch(d_output_cv);
        const auto write = fk::TensorWrite<OutputType>::build(d_tensor_output);

        START_FIRST_BENCHMARK
        for (int crop_i = 0; crop_i < BATCH; crop_i++) {
            fk::executeOperations(stream, read_array[crop_i], saturate,
                                   mul, sub, div, write_array[crop_i]);
        }
        STOP_FIRST_START_SECOND_BENCHMARK
            fk::executeOperations(stream, read, saturate, mul, sub, div, write);
        STOP_SECOND_BENCHMARK

        d_tensor_output.download(h_tensor_output, stream);

        // Verify results
        for (int crop_i = 0; crop_i < BATCH; crop_i++) {
            d_output_cv[crop_i].download(h_cvResults[crop_i], stream);
        }

        gpuErrchk(cudaStreamSynchronize(stream));

        for (int crop_i = 0; crop_i < BATCH; crop_i++) {
            fk::Ptr2D<OutputType> cvRes = h_cvResults[crop_i];
            fk::Ptr2D<OutputType> cvGSRes = h_tensor_output.getPlane(crop_i);
            bool passedThisTime = compareAndCheck(cvRes, cvGSRes);
            if (!passedThisTime) { std::cout << "Failed on crop idx=" << crop_i << std::endl; }
            passed &= passedThisTime;
        }
    } catch (const std::exception& e) {
        error_s << e.what();
        passed = false;
        exception = true;
    }

    if (!passed) {
        if (!exception) {
            std::stringstream ss;
            ss << "benchark_Horizontal_Fusion_NO_CPU_OVERHEAD";
            std::cout << ss.str() << " failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
        } else {
            std::stringstream ss;
            ss << "benchark_Horizontal_Fusion_NO_CPU_OVERHEAD";
            std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
        }
    }

    return passed;
}

template <size_t... Is>
bool launch_benchmark_Horizontal_Fusion(const size_t& NUM_ELEMS_X, const size_t& NUM_ELEMS_Y, const std::index_sequence<Is...>& seq, cudaStream_t stream) {
    bool passed = true;

    passed &= (benchmark_Horizontal_Fusion<variableDimensionValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, stream) && ...);

    return passed;
}

template <size_t... Is>
bool launch_benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD(const size_t& NUM_ELEMS_X, const size_t& NUM_ELEMS_Y, const std::index_sequence<Is...>& seq, cudaStream_t stream) {
    bool passed = true;

    passed &= (benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD<variableDimensionValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, stream) && ...);

    return passed;
}

int launch() {
    constexpr size_t NUM_ELEMS_X = 3840;
    constexpr size_t NUM_ELEMS_Y = 2160;
    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    warmup = true;
    launch_benchmark_Horizontal_Fusion(NUM_ELEMS_X, NUM_ELEMS_Y, std::make_index_sequence<NUM_EXPERIMENTS>{}, stream);
    launch_benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD(NUM_ELEMS_X, NUM_ELEMS_Y, std::make_index_sequence<NUM_EXPERIMENTS>{}, stream);
    warmup = false;

    launch_benchmark_Horizontal_Fusion(NUM_ELEMS_X, NUM_ELEMS_Y, std::make_index_sequence<NUM_EXPERIMENTS>{}, stream);
    launch_benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD(NUM_ELEMS_X, NUM_ELEMS_Y, std::make_index_sequence<NUM_EXPERIMENTS>{}, stream);

    gpuErrchk(cudaStreamDestroy(stream));

    return 0;
}