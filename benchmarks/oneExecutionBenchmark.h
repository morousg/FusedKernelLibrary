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

#ifndef FK_BENCHMARKS_ONE_EXECUTION_H
#define FK_BENCHMARKS_ONE_EXECUTION_H

#include <sstream>
#include <fstream>

#include <benchmarks/fkBenchmarksCommon.h>

template <size_t ITERATIONS>
float computeVariance(const float& mean, const std::array<float, ITERATIONS>& times) {
    float sumOfDiff = 0.f;
    for (int i = 0; i < ITERATIONS; i++) {
        const float diff = times[i] - mean;
        sumOfDiff += (diff * diff);
    }
    return sumOfDiff / (ITERATIONS - 1);
}

template <int VARIABLE_DIMENSION, int ITERATIONS, int NUM_BATCH_VALUES, const std::array<size_t, NUM_BATCH_VALUES>& variableDimensionValues>
inline void processExecution(const BenchmarkResultsNumbersOne& resF,
                             const std::string& functionName,
                             const std::array<float, ITERS>& fkElapsedTime,
                             const std::string& variableDimension) {
    // Create 2D Table for changing types and changing batch
    const std::string fileName = functionName + std::string("_") + std::string(fk::toStrView(fk::defaultParArch)) + std::string(".csv");
    if constexpr (VARIABLE_DIMENSION == variableDimensionValues[0]) {
        if (currentFile.find(fileName) == currentFile.end()) {
            currentFile[fileName].open(path + fileName);
        }
        currentFile[fileName] << variableDimension;
        currentFile[fileName] << ", MeanTime";
        currentFile[fileName] << ", TimeVariance";
        currentFile[fileName] << ", MaxTime";
        currentFile[fileName] << ", MinTime";
        currentFile[fileName] << std::endl;
    }

    const bool mustStore = currentFile.find(fileName) != currentFile.end();
    if (mustStore) {
        const float fkMean = resF.fkElapsedTimeAcum / ITERATIONS;
        const float fkVariance = computeVariance(fkMean, fkElapsedTime);

        currentFile[fileName] << VARIABLE_DIMENSION;
        currentFile[fileName] << ", " << fkMean;
        currentFile[fileName] << ", " << computeVariance(fkMean, fkElapsedTime);
        currentFile[fileName] << ", " << resF.fkElapsedTimeMax;
        currentFile[fileName] << ", " << resF.fkElapsedTimeMin;
        currentFile[fileName] << std::endl;
    }
}
  
#define START_FK_BENCHMARK \
std::cout << "Executing " << __func__ << " fusing " << VARIABLE_DIMENSION << " operations. " << std::endl; \
BenchmarkResultsNumbersOne resF; \
TimeMarkerOne<> marker(stream); \
for (int i = 0; i < ITERS; i++) { \
    marker.start();
 
#define STOP_FK_BENCHMARK \
    marker.stop(resF, i); \
} \
processExecution<VARIABLE_DIMENSION, ITERS, variableDimensionValues.size(), variableDimensionValues>(resF, __func__, marker.getElapsedTime(), VARIABLE_DIMENSION_NAME);
 
#define CLOSE_BENCHMARK \
for (auto&& [_, file] : currentFile) { \
    file.close(); \
}

#endif // FK_BENCHMARKS_ONE_EXECUTION_H
 