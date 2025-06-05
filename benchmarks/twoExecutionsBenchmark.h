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

#ifndef FK_BENCHMARKS_TWO_EXECUTIONS_H
#define FK_BENCHMARKS_TWO_EXECUTIONS_H

#include <sstream>
#include <fstream>

#include <benchmarks/fkBenchmarksCommon.h>

bool warmup{false};

template <size_t ITERATIONS> float computeVariance(const float &mean, const std::array<float, ITERATIONS> &times) {
  float sumOfDiff = 0.f;
  for (int idx = 0; idx <ITERATIONS; ++idx) {
    const float diff = times[idx] - mean;
    sumOfDiff += (diff * diff);
  }
  return sumOfDiff / (ITERATIONS - 1);
}

template <int BATCH, int ITERATIONS, int NUM_BATCH_VALUES, const std::array<size_t, NUM_BATCH_VALUES> &batchValues>
inline void processExecution(const BenchmarkResultsNumbersTwo& resF, const std::string &functionName,
                             const std::string& firstLabel, const std::string& secondLabel,
                             const std::array<float, ITERS> &firstElapsedTime,
                             const std::array<float, ITERS> &secondElapsedTime, const std::string &variableDimension) {
  // Create 2D Table for changing types and changing batch
  const std::string fileName = functionName + std::string("_") + std::string(fk::toStrView(fk::defaultParArch)) + std::string(".csv");
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

#define START_FIRST_BENCHMARK(ARCH)                                                                                            \
  std::cout << "Executing " << __func__ << " using " << BATCH << " " << VARIABLE_DIMENSION_NAME << " " << (BATCH - FIRST_VALUE) / INCREMENT \
            << "/" << NUM_EXPERIMENTS << std::endl;                                                                    \
  BenchmarkResultsNumbersTwo resF;                                                                                        \
  TimeMarkerTwo<ARCH> marker(stream);                                                                             \
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

#endif // FK_BENCHMARKS_TWO_EXECUTIONS_H