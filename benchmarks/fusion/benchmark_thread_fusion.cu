/* Copyright 2023-2025 Oscar Amoros Huguet
   Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

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

#include <fused_kernel/core/core.cuh>
#include <iostream>
#include <fused_kernel/fused_kernel.cuh>
#include "tests/nvtx.h"

constexpr size_t NUM_EXPERIMENTS = 5;
constexpr size_t FIRST_VALUE = 1024;
constexpr size_t INCREMENT = 1024;
constexpr std::array<size_t, NUM_EXPERIMENTS> variableDimensionValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;
constexpr char VARIABLE_DIMENSION_NAME[] = "Pixels per side";
constexpr std::string_view FIRST_LABEL = "Normal";
constexpr std::string_view SECOND_LABEL = "ThreadFusion";

template <typename T, size_t RESOLUTION>
bool testThreadFusionSameTypeIO(cudaStream_t& stream) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    constexpr size_t BATCH = RESOLUTION;

    constexpr uint NUM_ELEMS_X = (uint)RESOLUTION;
    constexpr uint NUM_ELEMS_Y = (uint)RESOLUTION;

    T val_init = fk::make_set<T>(2);

    try {
        fk::Ptr2D<T> d_input(NUM_ELEMS_Y, NUM_ELEMS_X);
        fk::setTo(val_init, d_input, stream);
        fk::Ptr2D<T> d_output_cvGS(NUM_ELEMS_Y, NUM_ELEMS_X);
        fk::Ptr2D<T> d_output_cvGS_ThreadFusion(NUM_ELEMS_Y, NUM_ELEMS_X);

        fk::Ptr2D<T> h_cvGSResults(NUM_ELEMS_Y, NUM_ELEMS_X, 0, fk::MemType::HostPinned);
        fk::Ptr2D<T> h_cvGSResults_ThreadFusion(NUM_ELEMS_Y, NUM_ELEMS_X, 0, fk::MemType::HostPinned);

        // In this case it's not OpenCV, it's cvGPUSpeedup without thread fusion
        START_FIRST_BENCHMARK
        // cvGPUSpeedup non fusion version
        const auto read = fk::PerThreadRead<fk::_2D, T>::build(d_input);
        const auto write = fk::PerThreadWrite<fk::_2D, T>::build(d_output_cvGS);
        fk::executeOperations(stream, read, write);
        STOP_FIRST_START_SECOND_BENCHMARK
        // cvGPUSpeedup fusion version
        const auto readTF = fk::PerThreadRead<fk::_2D, T>::build(d_input);
        const auto writeTF = fk::PerThreadWrite<fk::_2D, T>::build(d_output_cvGS_ThreadFusion);
        fk::executeOperations<true>(stream, readTF, writeTF);
        STOP_SECOND_BENCHMARK

        // Verify results
        d_output_cvGS_ThreadFusion.download(h_cvGSResults_ThreadFusion, stream);
        d_output_cvGS.download(h_cvGSResults, stream);

        gpuErrchk(cudaStreamSynchronize(stream));

        passed = compareAndCheck(h_cvGSResults_ThreadFusion, h_cvGSResults);
    } catch (const std::exception& e) {
        error_s << e.what();
        passed = false;
        exception = true;
    }

    if (!passed) {
        if (!exception) {
            std::stringstream ss;
            ss << "testThreadFusionTimes<" << fk::typeToString<T>();
            std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
        } else {
            std::stringstream ss;
            ss << "testThreadFusionTimes<" << fk::typeToString<T>();
            std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
        }
    }
    return passed;
}

template <size_t... IDX>
bool testThreadFusionSameTypeIO_launcher_impl(cudaStream_t stream, const std::integer_sequence<size_t, IDX...>&) {
    bool passed = true;

#define LAUNCH_testThreadFusionSameTypeIO(BASE) \
    passed &= (testThreadFusionSameTypeIO<BASE, variableDimensionValues[IDX]>(stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE, variableDimensionValues[IDX] + 1>(stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## 2, variableDimensionValues[IDX]>(stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## 2, variableDimensionValues[IDX] + 1>(stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## 3, variableDimensionValues[IDX]>(stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## 3, variableDimensionValues[IDX] + 1>(stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## 4, variableDimensionValues[IDX]>(stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## 4, variableDimensionValues[IDX] + 1>(stream) && ...);

    LAUNCH_testThreadFusionSameTypeIO(uchar)
    LAUNCH_testThreadFusionSameTypeIO(char)
    LAUNCH_testThreadFusionSameTypeIO(ushort)
    LAUNCH_testThreadFusionSameTypeIO(short)
    LAUNCH_testThreadFusionSameTypeIO(int)
    LAUNCH_testThreadFusionSameTypeIO(float)
    LAUNCH_testThreadFusionSameTypeIO(double)
#undef LAUNCH_testThreadFusionTimes

        return passed;
}

bool testThreadFusionSameTypeIO_launcher(cudaStream_t stream) {
    return testThreadFusionSameTypeIO_launcher_impl(stream, std::make_index_sequence<variableDimensionValues.size()>());
}

int launch() {
    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));
    bool passed = true;
    {
        PUSH_RANGE_RAII p("testThreadFusionSameTypeIO");
        passed &= testThreadFusionSameTypeIO_launcher(stream);
    }

    return 0;
}