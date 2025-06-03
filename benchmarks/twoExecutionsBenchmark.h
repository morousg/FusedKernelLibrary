/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Albert Andaluz Gonzalez
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <unordered_map>
#include <iostream>
#include <sstream>
#include <fstream>
#include <array>
#include <chrono>
#include <fused_kernel/core/execution_model/parallel_architectures.h>

std::unordered_map<std::string, std::stringstream> benchmarkResultsText;
std::unordered_map<std::string, std::ofstream> currentFile;
// Select the path where to write the benchmark files
const std::string path{ "" };

constexpr int ITERS = 100;
bool warmup{false};

struct BenchmarkResultsNumbers {
    float firstElapsedTimeMax{ fk::minValue<float> };
    float firstElapsedTimeMin{ fk::maxValue<float> };
    float firstElapsedTimeAcum{ 0.f };
    float secondElapsedTimeMax{ fk::minValue<float> };
    float secondElapsedTimeMin{ fk::maxValue<float> };
    float secondElapsedTimeAcum{ 0.f };
};

template <size_t ITERATIONS> float computeVariance(const float &mean, const std::array<float, ITERATIONS> &times) {
  float sumOfDiff = 0.f;
  for (int idx = 0; idx <ITERATIONS; ++idx) {
    const float diff = times[idx] - mean;
    sumOfDiff += (diff * diff);
  }
  return sumOfDiff / (ITERATIONS - 1);
}

template <int BATCH, int ITERATIONS, int NUM_BATCH_VALUES, const std::array<size_t, NUM_BATCH_VALUES> &batchValues>
inline void processExecution(const BenchmarkResultsNumbers &resF, const std::string &functionName,
                             const std::string& firstLabel, const std::string& secondLabel,
                             const std::array<float, ITERS> &firstElapsedTime,
                             const std::array<float, ITERS> &secondElapsedTime, const std::string &variableDimension) {
  // Create 2D Table for changing types and changing batch
  const std::string fileName = functionName + std::string(".csv");
  if constexpr (BATCH == batchValues[0]) {
    if (currentFile.find(fileName) == currentFile.end()) {
      currentFile[fileName].open(path + fileName);
    }
    currentFile[fileName] << variableDimension;
    currentFile[fileName] << ", " + firstLabel + " MeanTime";
    currentFile[fileName] << ", " + firstLabel + " TimeVariance";
    currentFile[fileName] << ", " + firstLabel + " MaxTime";
    currentFile[fileName] << ", " + firstLabel + " MinTime";
    currentFile[fileName] << ", " + secondLabel + " MeanTime";
    currentFile[fileName] << ", " + secondLabel + " TimeVariance";
    currentFile[fileName] << ", " + secondLabel + " MaxTime";
    currentFile[fileName] << ", " + secondLabel + " MinTime";
    currentFile[fileName] << ", Mean Speedup";
    currentFile[fileName] << std::endl;
  }

  const bool mustStore = currentFile.find(fileName) != currentFile.end();
  if (mustStore) {
    const float firstMean = resF.firstElapsedTimeAcum / ITERATIONS;
    const float secondMean = resF.secondElapsedTimeAcum / ITERATIONS;
    const float firstVariance = computeVariance(firstMean, firstElapsedTime);
    const float secondVariance = computeVariance(secondMean, secondElapsedTime);
    float meanSpeedup{0.f};
    for (int idx = 0; idx <ITERS; ++idx) {
      meanSpeedup += firstElapsedTime[idx] / secondElapsedTime[idx];
    }
    meanSpeedup /= ITERS;

 
    currentFile[fileName] << BATCH;
    currentFile[fileName] << ", " << firstMean;
    currentFile[fileName] << ", " << computeVariance(firstMean, firstElapsedTime);
    currentFile[fileName] << ", " << resF.firstElapsedTimeMax;
    currentFile[fileName] << ", " << resF.firstElapsedTimeMin;
    currentFile[fileName] << ", " << secondMean;
    currentFile[fileName] << ", " << computeVariance(secondMean, secondElapsedTime);
    currentFile[fileName] << ", " << resF.secondElapsedTimeMax;
    currentFile[fileName] << ", " << resF.secondElapsedTimeMin;
    currentFile[fileName] << ", " << meanSpeedup;
    currentFile[fileName] << std::endl;
  }
}

class TimeMarkerInterface {
    virtual void startFirst() = 0;
    virtual void stopFirstStartSecond(BenchmarkResultsNumbers& resF, const int& idx) = 0;
    virtual void stopSecond(BenchmarkResultsNumbers& resF, const int& idx) = 0;
    virtual std::array<float, ITERS> getFirstElapsedTime() const = 0;
    virtual std::array<float, ITERS> getSecondElapsedTime() const = 0;
};

template <enum fk::ParArch PA = fk::defaultParArch>
class TimeMarker;

template <>
class TimeMarker<fk::ParArch::CPU> final : public TimeMarkerInterface {
    std::array<float, ITERS> firstElapsedTime;
    std::array<float, ITERS> secondElapsedTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
public:
    TimeMarker(fk::Stream stream) {
        firstElapsedTime.fill(0.f);
        secondElapsedTime.fill(0.f);
    }

    ~TimeMarker() = default;

    void startFirst() final {
        start = std::chrono::high_resolution_clock::now();
    };

    void stopFirstStartSecond(BenchmarkResultsNumbers& resF, const int& idx) final {
        stop = std::chrono::high_resolution_clock::now();
        firstElapsedTime[idx] = std::chrono::duration<float, std::milli>(stop - start).count();
        resF.firstElapsedTimeMax = resF.firstElapsedTimeMax < firstElapsedTime[idx] ? firstElapsedTime[idx] : resF.firstElapsedTimeMax;
        resF.firstElapsedTimeMin = resF.firstElapsedTimeMin > firstElapsedTime[idx] ? firstElapsedTime[idx] : resF.firstElapsedTimeMin;
        resF.firstElapsedTimeAcum += firstElapsedTime[idx];
        start = std::chrono::high_resolution_clock::now();
    }

    void stopSecond(BenchmarkResultsNumbers& resF, const int& idx) final {
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
class TimeMarker<fk::ParArch::GPU_NVIDIA> final : public TimeMarkerInterface {
    cudaEvent_t start, stop;
    cudaStream_t stream;
    std::array<float, ITERS> firstElapsedTime;
    std::array<float, ITERS> secondElapsedTime;
public:
    TimeMarker(fk::Stream stream_) : stream(stream_) {
        gpuErrchk(cudaEventCreate(&start));
        gpuErrchk(cudaEventCreate(&stop));
        firstElapsedTime.fill(0.f);
        secondElapsedTime.fill(0.f);
    }

    ~TimeMarker() {
        gpuErrchk(cudaEventDestroy(start));
        gpuErrchk(cudaEventDestroy(stop));
    }

    void startFirst() final {
        gpuErrchk(cudaEventRecord(start, stream));
    };
    void stopFirstStartSecond(BenchmarkResultsNumbers& resF, const int& idx) final {
        gpuErrchk(cudaEventRecord(stop, stream));
        gpuErrchk(cudaEventSynchronize(stop));
        gpuErrchk(cudaEventElapsedTime(&firstElapsedTime[idx], start, stop));
        resF.firstElapsedTimeMax = resF.firstElapsedTimeMax < firstElapsedTime[idx] ? firstElapsedTime[idx] : resF.firstElapsedTimeMax;
        resF.firstElapsedTimeMin = resF.firstElapsedTimeMin > firstElapsedTime[idx] ? firstElapsedTime[idx] : resF.firstElapsedTimeMin;
        resF.firstElapsedTimeAcum += firstElapsedTime[idx];
        gpuErrchk(cudaEventRecord(start, stream));
    }
    void stopSecond(BenchmarkResultsNumbers& resF, const int& idx) final {
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

#define START_FIRST_BENCHMARK(ARCH)                                                                                            \
  std::cout << "Executing " << __func__ << " using " << BATCH << " " << VARIABLE_DIMENSION_NAME << " " << (BATCH - FIRST_VALUE) / INCREMENT \
            << "/" << NUM_EXPERIMENTS << std::endl;                                                                    \
  BenchmarkResultsNumbers resF;                                                                                        \
  TimeMarker<ARCH> marker(stream);                                                                             \
  for (int idx = 0; idx <ITERS; ++idx) {                                                                                    \
    marker.startFirst();

#define STOP_FIRST_START_SECOND_BENCHMARK marker.stopFirstStartSecond(resF, idx);

#define STOP_SECOND_BENCHMARK                                                                                              \
    marker.stopSecond(resF, idx);                                                                                       \
    if (warmup) break;                                                                        \
  }                                                                                                                  \
processExecution<BATCH, ITERS, variableDimensionValues.size(), variableDimensionValues>(                               \
        resF, __func__, std::string(FIRST_LABEL), std::string(SECOND_LABEL), \
        marker.getFirstElapsedTime(), marker.getSecondElapsedTime(), VARIABLE_DIMENSION_NAME);
 
 
#define CLOSE_BENCHMARK                                                                                                \
  for (auto &&[_, file] : currentFile) {                                                                               \
    file.close();                                                                                                      \
  }
 