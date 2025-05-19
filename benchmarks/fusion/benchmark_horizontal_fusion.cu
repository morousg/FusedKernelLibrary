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

#include <fused_kernel/fused_kernel.cuh>

constexpr char VARIABLE_DIMENSION_NAME[]{ "Batch size" };

constexpr size_t NUM_EXPERIMENTS = 60; // Used 100 in the paper
constexpr size_t FIRST_VALUE = 1;
constexpr size_t INCREMENT = 5;

constexpr std::array<size_t, NUM_EXPERIMENTS> variableDimensionValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

template <size_t BATCH>
bool benchark_Horizontal_Fusion(const size_t& NUM_ELEMS_X, const size_t& NUM_ELEMS_Y, const cudaStream_t& stream) {
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
            crops[crop_i].Alloc(cropSize);
            d_output_cv[crop_i].Alloc(cropSize, 0, fk::MemType::Device);
            h_cvResults[crop_i].Alloc(cropSize, 0, fk::MemType::HostPinned);
        }

        START_FIRST_BENCHMARK
            for (int crop_i = 0; crop_i < BATCH; crop_i++) {
                fk::executeOperations(crops[crop_i], stream,
                    fk::convertTo<InputType, OutputType>(val_alpha),
                    fk::subtract<OutputType>(val_sub),
                    fk::divide<OutputType>(val_div),
                    fk::write<OutputType>(d_output_cv[crop_i]));
            }

        STOP_FIRST_START_SECOND_BENCHMARK
            // cvGPUSpeedup
            // Assuming we use all the batch
            // On Linux it is necessary to pass the BATCH as a template parameter
            // On Windows (VS2022 Community) it is not needed, it is deduced from crops 
            fk::executeOperations(crops, stream,
                fk::convertTo<InputType, OutputType>(val_alpha),
                fk::subtract<OutputType>(val_sub),
                fk::divide<OutputType>(val_div),
                fk::write<OutputType>(d_tensor_output, cropSize));

        STOP_SECOND_BENCHMARK

            d_tensor_output.download(h_tensor_output, cv_stream);

        // Verify results
        for (int crop_i = 0; crop_i < BATCH; crop_i++) {
            d_output_cv[crop_i].download(h_cvResults[crop_i], cv_stream);
        }

        cv_stream.waitForCompletion();

        for (int crop_i = 0; crop_i < BATCH; crop_i++) {
            cv::Mat cvRes = h_cvResults[crop_i];
            cv::Mat cvGSRes = cv::Mat(cropSize.height, cropSize.width, CV_TYPE_O, h_tensor_output.row(crop_i).data);
            bool passedThisTime = compareAndCheck<CV_TYPE_O>(cropSize.width, cropSize.height, cvRes, cvGSRes);
            if (!passedThisTime) { std::cout << "Failed on crop idx=" << crop_i << std::endl; }
            passed &= passedThisTime;
        }
    } catch (const cv::Exception& e) {
        if (e.code != -210) {
            error_s << e.what();
            passed = false;
            exception = true;
        }
    } catch (const std::exception& e) {
        error_s << e.what();
        passed = false;
        exception = true;
    }

    if (!passed) {
        if (!exception) {
            std::stringstream ss;
            ss << "test_batchread_x_write3D<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
            std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
        } else {
            std::stringstream ss;
            ss << "test_batchread_x_write3D<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
            std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
        }
    }

    return passed;
}

template <int CV_TYPE_I, int CV_TYPE_O, size_t... Is>
bool launch_test_batchread_x_write3D_only_HorizontalFusion(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y, std::index_sequence<Is...> seq, cv::cuda::Stream cv_stream, bool enabled) {
    bool passed = true;

    int dummy[] = { (passed &= test_batchread_x_write3D_only_HorizontalFusion<CV_TYPE_I, CV_TYPE_O, batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, enabled), 0)... };
    (void)dummy;

    return passed;
}

int launch() {
    return 0;
}