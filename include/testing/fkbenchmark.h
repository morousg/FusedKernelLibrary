std::unordered_map<std::string, std::stringstream> benchmarkResultsText;
std::unordered_map<std::string, std::ofstream> currentFile;
// Select the path where to write the benchmark files
const std::string path{ "" };

constexpr int ITERS = 100;

struct BenchmarkResultsNumbers {
    float cvGSelapsedTimeMax;
    float cvGSelapsedTimeMin;
    float cvGSelapsedTimeAcum;
};

template <size_t ITERATIONS>
float computeVariance(const float& mean, const std::array<float, ITERATIONS>& times) {
    float sumOfDiff = 0.f;
    for (int i = 0; i < ITERATIONS; i++) {
        const float diff = times[i] - mean;
        sumOfDiff += (diff * diff);
    }
    return sumOfDiff / (ITERATIONS - 1);
}

template <int VARIABLE_DIMENSION, int ITERATIONS, int NUM_BATCH_VALUES, const std::array<size_t, NUM_BATCH_VALUES>& variableDimanesionValues>
inline void processExecution(const BenchmarkResultsNumbers& resF,
                             const std::string& functionName,
                             const std::array<float, ITERS>& cvGSelapsedTime,
                             const std::string& variableDimension) {
    // Create 2D Table for changing types and changing batch
    const std::string fileName = functionName + std::string(".csv");
    if constexpr (VARIABLE_DIMENSION == variableDimanesionValues[0]) {
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
        const float cvgsMean = resF.cvGSelapsedTimeAcum / ITERATIONS;
        const float cvgsVariance = computeVariance(cvgsMean, cvGSelapsedTime);

        currentFile[fileName] << VARIABLE_DIMENSION;
        currentFile[fileName] << ", " << cvgsMean;
        currentFile[fileName] << ", " << computeVariance(cvgsMean, cvGSelapsedTime);
        currentFile[fileName] << ", " << resF.cvGSelapsedTimeMax;
        currentFile[fileName] << ", " << resF.cvGSelapsedTimeMin;
        currentFile[fileName] << std::endl;
    }
}
  
#define START_FK_BENCHMARK \
std::cout << "Executing " << __func__ << " fusing " << VARIABLE_DIMENSION << " operations. " << std::endl; \
cudaEvent_t start, stop; \
BenchmarkResultsNumbers resF; \
resF.cvGSelapsedTimeMax = fk::minValue<float>; \
resF.cvGSelapsedTimeMin = fk::maxValue<float>; \
resF.cvGSelapsedTimeAcum = 0.f; \
gpuErrchk(cudaEventCreate(&start)); \
gpuErrchk(cudaEventCreate(&stop)); \
std::array<float, ITERS> cvGSelapsedTime; \
for (int i = 0; i < ITERS; i++) { \
gpuErrchk(cudaEventRecord(start, stream));
 
#define STOP_FK_BENCHMARK \
gpuErrchk(cudaEventRecord(stop, stream)); \
gpuErrchk(cudaEventSynchronize(stop)); \
gpuErrchk(cudaEventElapsedTime(&cvGSelapsedTime[i], start, stop)); \
resF.cvGSelapsedTimeMax = resF.cvGSelapsedTimeMax < cvGSelapsedTime[i] ? cvGSelapsedTime[i] : resF.cvGSelapsedTimeMax; \
resF.cvGSelapsedTimeMin = resF.cvGSelapsedTimeMin > cvGSelapsedTime[i] ? cvGSelapsedTime[i] : resF.cvGSelapsedTimeMin; \
resF.cvGSelapsedTimeAcum += cvGSelapsedTime[i]; \
} \
processExecution<VARIABLE_DIMENSION, ITERS, variableDimanesionValues.size(), variableDimanesionValues>(resF, __func__, cvGSelapsedTime, VARIABLE_DIMENSION_NAME);
 
#define CLOSE_BENCHMARK \
for (auto&& [_, file] : currentFile) { \
    file.close(); \
}
 
 