/* Copyright 2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_BENCHMARKS_COMMON_H
#define FK_BENCHMARKS_COMMON_H

#include <array>
#include <chrono>
#include <unordered_map>
#include <iostream>

#include <fused_kernel/core/utils/vlimits.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>

#include <fused_kernel/core/execution_model/parallel_architectures.h>

constexpr int ITERS = 100;
std::unordered_map<std::string, std::stringstream> benchmarkResultsText;
std::unordered_map<std::string, std::ofstream> currentFile;
// Select the path where to write the benchmark files
const std::string path{ "" };

template <size_t START_VALUE, size_t INCREMENT, std::size_t... Is>
constexpr std::array<size_t, sizeof...(Is)> generate_sequence(std::index_sequence<Is...>) {
    return std::array<size_t, sizeof...(Is)>{(START_VALUE + (INCREMENT * Is))...};
}

template <size_t START_VALUE, size_t INCREMENT, size_t NUM_ELEMS>
constexpr std::array<size_t, NUM_ELEMS> arrayIndexSecuence = generate_sequence<START_VALUE, INCREMENT>(std::make_index_sequence<NUM_ELEMS>{});

template <typename T>
inline bool compareAndCheck(const fk::Ptr2D<T>& firstResult, const fk::Ptr2D<T>& secondResult) {
    const bool sameDims = firstResult.dims().width == secondResult.dims().width && firstResult.dims().height == secondResult.dims().height;
    if (!sameDims) {
        std::cout << "Dimensions do not match: " << firstResult.dims().width << "x" << firstResult.dims().height << " vs " << secondResult.dims().width << "x" << secondResult.dims().height << std::endl;
        return false;
    }
    for (uint y = 0; y < firstResult.dims().height; ++y) {
        for (uint x = 0; x < firstResult.dims().width; ++x) {
            if (!fk::Equal<T>::exec(fk::make_tuple(firstResult.at(fk::Point(x, y)), secondResult.at(fk::Point(x, y))))) {
                std::cout << "Mismatch at (" << x << ", " << y << ") " << std::endl;
                return false;
            }
        }
    }
    return true;
}

struct BenchmarkResultsNumbersOne {
    float fkElapsedTimeMax{ fk::minValue<float> };;
    float fkElapsedTimeMin{ fk::maxValue<float> };
    float fkElapsedTimeAcum{ 0.f };
};

struct BenchmarkResultsNumbersTwo {
    float firstElapsedTimeMax{ fk::minValue<float> };
    float firstElapsedTimeMin{ fk::maxValue<float> };
    float firstElapsedTimeAcum{ 0.f };
    float secondElapsedTimeMax{ fk::minValue<float> };
    float secondElapsedTimeMin{ fk::maxValue<float> };
    float secondElapsedTimeAcum{ 0.f };
};

class TimeMarkerInterfaceTwo {
    virtual void startFirst() = 0;
    virtual void stopFirstStartSecond(BenchmarkResultsNumbersTwo& resF, const int& idx) = 0;
    virtual void stopSecond(BenchmarkResultsNumbersTwo& resF, const int& idx) = 0;
    virtual std::array<float, ITERS> getFirstElapsedTime() const = 0;
    virtual std::array<float, ITERS> getSecondElapsedTime() const = 0;
};

template <enum fk::ParArch PA = fk::defaultParArch>
class TimeMarkerTwo;

template <>
class TimeMarkerTwo<fk::ParArch::CPU> final : public TimeMarkerInterfaceTwo {
    std::array<float, ITERS> firstElapsedTime;
    std::array<float, ITERS> secondElapsedTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
public:
    TimeMarkerTwo(fk::Stream stream) {
        firstElapsedTime.fill(0.f);
        secondElapsedTime.fill(0.f);
    }

    ~TimeMarkerTwo() = default;

    void startFirst() final {
        start = std::chrono::high_resolution_clock::now();
    };

    void stopFirstStartSecond(BenchmarkResultsNumbersTwo& resF, const int& idx) final {
        stop = std::chrono::high_resolution_clock::now();
        firstElapsedTime[idx] = std::chrono::duration<float, std::milli>(stop - start).count();
        resF.firstElapsedTimeMax = resF.firstElapsedTimeMax < firstElapsedTime[idx] ? firstElapsedTime[idx] : resF.firstElapsedTimeMax;
        resF.firstElapsedTimeMin = resF.firstElapsedTimeMin > firstElapsedTime[idx] ? firstElapsedTime[idx] : resF.firstElapsedTimeMin;
        resF.firstElapsedTimeAcum += firstElapsedTime[idx];
        start = std::chrono::high_resolution_clock::now();
    }

    void stopSecond(BenchmarkResultsNumbersTwo& resF, const int& idx) final {
        stop = std::chrono::high_resolution_clock::now();
        secondElapsedTime[idx] = std::chrono::duration<float, std::milli>(stop - start).count();
        resF.secondElapsedTimeMax = resF.secondElapsedTimeMax < secondElapsedTime[idx] ? secondElapsedTime[idx] : resF.secondElapsedTimeMax;
        resF.secondElapsedTimeMin = resF.secondElapsedTimeMin > secondElapsedTime[idx] ? secondElapsedTime[idx] : resF.secondElapsedTimeMin;
        resF.secondElapsedTimeAcum += secondElapsedTime[idx];
    };
    std::array<float, ITERS> getFirstElapsedTime() const final {
        return firstElapsedTime;
    };
    std::array<float, ITERS> getSecondElapsedTime() const final {
        return secondElapsedTime;
    };
};

#if defined(__CUDACC__) || defined(__HIP__)
template <>
class TimeMarkerTwo<fk::ParArch::GPU_NVIDIA> final : public TimeMarkerInterfaceTwo {
    cudaEvent_t start, stop;
    cudaStream_t stream;
    std::array<float, ITERS> firstElapsedTime;
    std::array<float, ITERS> secondElapsedTime;
public:
    TimeMarkerTwo(fk::Stream stream_) : stream(stream_) {
        gpuErrchk(cudaEventCreate(&start));
        gpuErrchk(cudaEventCreate(&stop));
        firstElapsedTime.fill(0.f);
        secondElapsedTime.fill(0.f);
    }

    ~TimeMarkerTwo() {
        gpuErrchk(cudaEventDestroy(start));
        gpuErrchk(cudaEventDestroy(stop));
    }

    void startFirst() final {
        gpuErrchk(cudaEventRecord(start, stream));
    };
    void stopFirstStartSecond(BenchmarkResultsNumbersTwo& resF, const int& idx) final {
        gpuErrchk(cudaEventRecord(stop, stream));
        gpuErrchk(cudaEventSynchronize(stop));
        gpuErrchk(cudaEventElapsedTime(&firstElapsedTime[idx], start, stop));
        resF.firstElapsedTimeMax = resF.firstElapsedTimeMax < firstElapsedTime[idx] ? firstElapsedTime[idx] : resF.firstElapsedTimeMax;
        resF.firstElapsedTimeMin = resF.firstElapsedTimeMin > firstElapsedTime[idx] ? firstElapsedTime[idx] : resF.firstElapsedTimeMin;
        resF.firstElapsedTimeAcum += firstElapsedTime[idx];
        gpuErrchk(cudaEventRecord(start, stream));
    }
    void stopSecond(BenchmarkResultsNumbersTwo& resF, const int& idx) final {
        gpuErrchk(cudaEventRecord(stop, stream));
        gpuErrchk(cudaEventSynchronize(stop));
        gpuErrchk(cudaEventElapsedTime(&secondElapsedTime[idx], start, stop));
        resF.secondElapsedTimeMax = resF.secondElapsedTimeMax < secondElapsedTime[idx] ? secondElapsedTime[idx] : resF.secondElapsedTimeMax;
        resF.secondElapsedTimeMin = resF.secondElapsedTimeMin > secondElapsedTime[idx] ? secondElapsedTime[idx] : resF.secondElapsedTimeMin;
        resF.secondElapsedTimeAcum += secondElapsedTime[idx];
    };
    std::array<float, ITERS> getFirstElapsedTime() const final {
        return firstElapsedTime;
    };
    std::array<float, ITERS> getSecondElapsedTime() const final {
        return secondElapsedTime;
    };
};
#endif 

#endif // FK_BENCHMARKS_COMMON_H