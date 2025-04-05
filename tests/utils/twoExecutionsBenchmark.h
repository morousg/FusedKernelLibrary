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
#include <string>
#include <fstream>
#include <array>

std::unordered_map<std::string, std::stringstream> benchmarkResultsText;
std::unordered_map<std::string, std::ofstream> currentFile;
// Select the path where to write the benchmark files
const std::string path{ "" };

constexpr int ITERS = 100;

struct BenchmarkResultsNumbers {
  float firstElapsedTimeMax;
  float firstElapsedTimeMin;
  float firstElapsedTimeAcum;
  float secondElapsedTimeMax;
  float secondElapsedTimeMin;
  float secondElapsedTimeAcum;
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
                             const std::string& firstLable, const std::string& secondLable,
                             const std::array<float, ITERS> &firstElapsedTime,
                             const std::array<float, ITERS> &secondElapsedTime, const std::string &variableDimension) {
  // Create 2D Table for changing types and changing batch
  const std::string fileName = functionName + std::string(".csv");
  if constexpr (BATCH == batchValues[0]) {
    if (currentFile.find(fileName) == currentFile.end()) {
      currentFile[fileName].open(path + fileName);
    }
    currentFile[fileName] << variableDimension;
    currentFile[fileName] << ", " + firstLable + " MeanTime";
    currentFile[fileName] << ", " + firstLable + " TimeVariance";
    currentFile[fileName] << ", " + firstLable + " MaxTime";
    currentFile[fileName] << ", " + firstLable + " MinTime";
    currentFile[fileName] << ", " + secondLable + " MeanTime";
    currentFile[fileName] << ", " + secondLable + " TimeVariance";
    currentFile[fileName] << ", " + secondLable + " MaxTime";
    currentFile[fileName] << ", " + secondLable + " MinTime";
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

#define START_FIRST_BENCHMARK                                                                                            \
  std::cout << "Executing " << __func__ << " fusing " << BATCH << " operations. " << (BATCH - FIRST_VALUE) / INCREMENT \
            << "/" << NUM_EXPERIMENTS << std::endl;                                                                    \
  cudaEvent_t start, stop;                                                                                             \
  BenchmarkResultsNumbers resF;                                                                                        \
  resF.firstElapsedTimeMax = fk::minValue<float>;                                                                        \
  resF.firstElapsedTimeMin = fk::maxValue<float>;                                                                        \
  resF.firstElapsedTimeAcum = 0.f;                                                                                       \
  resF.secondElapsedTimeMax = fk::minValue<float>;                                                                         \
  resF.secondElapsedTimeMin = fk::maxValue<float>;                                                                         \
  resF.secondElapsedTimeAcum = 0.f;                                                                                        \
  gpuErrchk(cudaEventCreate(&start));                                                                                  \
  gpuErrchk(cudaEventCreate(&stop));                                                                                   \
  std::array<float, ITERS> firstElapsedTime;                                                                             \
  std::array<float, ITERS> secondElapsedTime;                                                                              \
  for (int idx = 0; idx <ITERS; ++idx) {                                                                                    \
    gpuErrchk(cudaEventRecord(start, compute_stream));

#define STOP_FIRST_START_SECOND_BENCHMARK                                                                                    \
  gpuErrchk(cudaEventRecord(stop, compute_stream));                                                                            \
  gpuErrchk(cudaEventSynchronize(stop));                                                                               \
  gpuErrchk(cudaEventElapsedTime(&firstElapsedTime[idx], start, stop));                                                    \
  resF.firstElapsedTimeMax = resF.firstElapsedTimeMax < firstElapsedTime[idx] ? firstElapsedTime[idx] : resF.firstElapsedTimeMax;    \
  resF.firstElapsedTimeMin = resF.firstElapsedTimeMin > firstElapsedTime[idx] ? firstElapsedTime[idx] : resF.firstElapsedTimeMin;    \
  resF.firstElapsedTimeAcum += firstElapsedTime[idx];                                                                        \
  gpuErrchk(cudaEventRecord(start, compute_stream));

#define STOP_SECOND_BENCHMARK                                                                                              \
  gpuErrchk(cudaEventRecord(stop, compute_stream));                                                                            \
  gpuErrchk(cudaEventSynchronize(stop));                                                                               \
  gpuErrchk(cudaEventElapsedTime(&secondElapsedTime[idx], start, stop));                                                     \
  resF.secondElapsedTimeMax = resF.secondElapsedTimeMax < secondElapsedTime[idx] ? secondElapsedTime[idx] : resF.secondElapsedTimeMax;         \
  resF.secondElapsedTimeMin = resF.secondElapsedTimeMin > secondElapsedTime[idx] ? secondElapsedTime[idx] : resF.secondElapsedTimeMin;         \
  resF.secondElapsedTimeAcum += secondElapsedTime[idx];                                                                          \
  }                                                                                                                  \
processExecution<BATCH, ITERS, batchValues.size(), batchValues>(                               \
      resF, __func__, std::string(FIRST_LABLE), std::string(SECOND_LABLE), firstElapsedTime, secondElapsedTime ,VARIABLE_DIMENSION);
 
 
#define CLOSE_BENCHMARK                                                                                                \
  for (auto &&[_, file] : currentFile) {                                                                               \
    file.close();                                                                                                      \
  }
 