#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// Naive CUDA transpose.
// One thread reads one value from A and writes one transposed value to B.
__global__ void transposeNaive(const float* A, float* B, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        B[col * rows + row] = A[row * cols + col];
    }
}

// Tiled CUDA transpose.
// The tile uses shared memory. The second dimension is 33 to reduce bank conflicts.
__global__ void transposeTiled(const float* A, float* B, int rows, int cols, int tileSize) {
    __shared__ float tile[32][33];

    int inputCol = blockIdx.x * tileSize + threadIdx.x;
    int inputRow = blockIdx.y * tileSize + threadIdx.y;

    // Some blocks use 32 x 8 threads. The loop lets those 8 rows cover a 32-row tile.
    for (int j = 0; j < tileSize; j += blockDim.y) {
        if (inputRow + j < rows && inputCol < cols) {
            tile[threadIdx.y + j][threadIdx.x] = A[(inputRow + j) * cols + inputCol];
        }
    }

    __syncthreads();

    int outputCol = blockIdx.y * tileSize + threadIdx.x;
    int outputRow = blockIdx.x * tileSize + threadIdx.y;

    for (int j = 0; j < tileSize; j += blockDim.y) {
        if (outputRow + j < cols && outputCol < rows) {
            B[(outputRow + j) * rows + outputCol] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

bool checkCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cout << message << ": " << cudaGetErrorString(result) << "\n";
        return false;
    }

    return true;
}

void fillRandom(std::vector<float>& A) {
    std::mt19937 randomGenerator(1234);
    std::uniform_real_distribution<float> randomValue(-1.0f, 1.0f);

    for (int i = 0; i < static_cast<int>(A.size()); i++) {
        A[i] = randomValue(randomGenerator);
    }
}

double transposeCpu(const std::vector<float>& A, std::vector<float>& B, int rows, int cols) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            B[col * rows + row] = A[row * cols + col];
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;
    return elapsed.count();
}

bool compareResults(const std::vector<float>& B, const float* gpuB, int N, const char* name) {
    for (int i = 0; i < N; i++) {
        if (std::fabs(B[i] - gpuB[i]) > 1.0e-5f) {
            std::cout << name << " wrong result at index " << i << "\n";
            return false;
        }
    }

    return true;
}

bool checkCpuTranspose(const std::vector<float>& A, const std::vector<float>& B, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float expected = A[row * cols + col];
            float actual = B[col * rows + row];

            if (std::fabs(expected - actual) > 1.0e-5f) {
                std::cout << "CPU wrong result at row " << row << ", col " << col << "\n";
                return false;
            }
        }
    }

    return true;
}

void writeCsvRow(std::ofstream& csv,
                 const char* implementation,
                 int rows,
                 int cols,
                 int blockX,
                 int blockY,
                 double timeMs,
                 double bandwidthGbPerSec,
                 double speedupVsCpu,
                 bool correct) {
    csv << implementation << "," << rows << "," << cols << "," << blockX << "," << blockY << "," << timeMs << ","
        << bandwidthGbPerSec << "," << speedupVsCpu << "," << (correct ? "true" : "false") << "\n";
}

bool runBenchmark(int rows, int cols, int blockX, int blockY, bool runCpu, double& cpuTimeForSize, std::ofstream& csv) {
    int N = rows * cols;
    size_t bytes = static_cast<size_t>(N) * sizeof(float);
    double bytesMoved = static_cast<double>(N) * sizeof(float) * 2.0;

    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> gpuB(N);

    fillRandom(A);

    double cpuTimeMs = 0.0;
    bool cpuCorrect = true;

    if (runCpu) {
        cpuTimeMs = transposeCpu(A, B, rows, cols);

        cpuCorrect = checkCpuTranspose(A, B, rows, cols);
        cpuTimeForSize = cpuTimeMs;

        double cpuBandwidthGbPerSec = bytesMoved / (cpuTimeMs / 1000.0) / 1.0e9;
        writeCsvRow(csv, "cpu", rows, cols, 0, 0, cpuTimeMs, cpuBandwidthGbPerSec, 1.0, cpuCorrect);

        std::cout << "CPU baseline\n";
        std::cout << "Matrix size: " << rows << " x " << cols << "\n";
        std::cout << "CPU time: " << cpuTimeMs << " ms\n";
        std::cout << "CPU correct: " << (cpuCorrect ? "true" : "false") << "\n\n";
    } else {
        transposeCpu(A, B, rows, cols);
    }

    float* deviceA = nullptr;
    float* deviceB = nullptr;

    if (!checkCuda(cudaMalloc(&deviceA, bytes), "cudaMalloc deviceA failed")) {
        return false;
    }

    if (!checkCuda(cudaMalloc(&deviceB, bytes), "cudaMalloc deviceB failed")) {
        cudaFree(deviceA);
        return false;
    }

    if (!checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "Copy A to GPU failed")) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        return false;
    }

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    dim3 threadsPerBlock(blockX, blockY);
    dim3 blocks((cols + blockX - 1) / blockX, (rows + blockY - 1) / blockY);

    // Warm up the kernel once so startup overhead does not dominate the timing.
    transposeNaive<<<blocks, threadsPerBlock>>>(deviceA, deviceB, rows, cols);
    cudaDeviceSynchronize();

    cudaEventRecord(startEvent);
    transposeNaive<<<blocks, threadsPerBlock>>>(deviceA, deviceB, rows, cols);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float naiveTimeMs = 0.0f;
    cudaEventElapsedTime(&naiveTimeMs, startEvent, stopEvent);

    if (!checkCuda(cudaGetLastError(), "Naive CUDA kernel failed")) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return false;
    }

    if (!checkCuda(cudaMemcpy(gpuB.data(), deviceB, bytes, cudaMemcpyDeviceToHost), "Copy naive B to CPU failed")) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return false;
    }

    bool naiveCorrect = compareResults(B, gpuB.data(), N, "Naive GPU");
    double naiveBandwidthGbPerSec = bytesMoved / (naiveTimeMs / 1000.0) / 1.0e9;
    double naiveSpeedup = cpuTimeForSize / naiveTimeMs;
    writeCsvRow(csv,
                "naive_gpu",
                rows,
                cols,
                blockX,
                blockY,
                naiveTimeMs,
                naiveBandwidthGbPerSec,
                naiveSpeedup,
                naiveCorrect);

    int tileSize = blockX;
    dim3 tiledBlocks((cols + tileSize - 1) / tileSize, (rows + tileSize - 1) / tileSize);

    // Warm up the tiled kernel once before timing.
    transposeTiled<<<tiledBlocks, threadsPerBlock>>>(deviceA, deviceB, rows, cols, tileSize);
    cudaDeviceSynchronize();

    cudaEventRecord(startEvent);
    transposeTiled<<<tiledBlocks, threadsPerBlock>>>(deviceA, deviceB, rows, cols, tileSize);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float tiledTimeMs = 0.0f;
    cudaEventElapsedTime(&tiledTimeMs, startEvent, stopEvent);

    if (!checkCuda(cudaGetLastError(), "Tiled CUDA kernel failed")) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return false;
    }

    if (!checkCuda(cudaMemcpy(gpuB.data(), deviceB, bytes, cudaMemcpyDeviceToHost), "Copy tiled B to CPU failed")) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return false;
    }

    bool tiledCorrect = compareResults(B, gpuB.data(), N, "Tiled GPU");
    double tiledBandwidthGbPerSec = bytesMoved / (tiledTimeMs / 1000.0) / 1.0e9;
    double tiledSpeedup = cpuTimeForSize / tiledTimeMs;
    writeCsvRow(csv,
                "tiled_gpu",
                rows,
                cols,
                blockX,
                blockY,
                tiledTimeMs,
                tiledBandwidthGbPerSec,
                tiledSpeedup,
                tiledCorrect);

    float* managedA = nullptr;
    float* managedB = nullptr;

    bool managedCorrect = true;
    float managedTimeMs = 0.0f;
    if (checkCuda(cudaMallocManaged(&managedA, bytes), "cudaMallocManaged managedA failed") &&
        checkCuda(cudaMallocManaged(&managedB, bytes), "cudaMallocManaged managedB failed")) {
        for (int i = 0; i < N; i++) {
            managedA[i] = A[i];
            managedB[i] = 0.0f;
        }

        int deviceId = 0;
        cudaGetDevice(&deviceId);

        int concurrentManagedAccess = 0;
        cudaDeviceGetAttribute(&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, deviceId);

        if (concurrentManagedAccess != 0) {
            cudaMemLocation gpuLocation;
            gpuLocation.type = cudaMemLocationTypeDevice;
            gpuLocation.id = deviceId;

            cudaMemPrefetchAsync(managedA, bytes, gpuLocation, 0, 0);
            cudaMemPrefetchAsync(managedB, bytes, gpuLocation, 0, 0);
        }

        cudaEventRecord(startEvent);
        transposeTiled<<<tiledBlocks, threadsPerBlock>>>(managedA, managedB, rows, cols, tileSize);
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);

        cudaEventElapsedTime(&managedTimeMs, startEvent, stopEvent);

        if (!checkCuda(cudaGetLastError(), "Unified Memory tiled kernel failed")) {
            managedCorrect = false;
        }

        cudaDeviceSynchronize();
        managedCorrect = managedCorrect && compareResults(B, managedB, N, "Unified Memory GPU");

        double managedBandwidthGbPerSec = bytesMoved / (managedTimeMs / 1000.0) / 1.0e9;
        double managedSpeedup = cpuTimeForSize / managedTimeMs;
        writeCsvRow(csv,
                    "managed_tiled_gpu",
                    rows,
                    cols,
                    blockX,
                    blockY,
                    managedTimeMs,
                    managedBandwidthGbPerSec,
                    managedSpeedup,
                    managedCorrect);
    } else {
        managedCorrect = false;
        writeCsvRow(csv, "managed_tiled_gpu", rows, cols, blockX, blockY, 0.0, 0.0, 0.0, false);
    }

    cudaFree(managedA);
    cudaFree(managedB);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    std::cout << "Matrix " << rows << " x " << cols << ", block " << blockX << " x " << blockY << "\n";
    std::cout << "Naive GPU time: " << naiveTimeMs << " ms, speedup vs CPU: " << naiveSpeedup
              << "x, correct: " << (naiveCorrect ? "true" : "false")
              << "\n";
    std::cout << "Tiled GPU time: " << tiledTimeMs << " ms, speedup vs CPU: " << tiledSpeedup
              << "x, correct: " << (tiledCorrect ? "true" : "false")
              << "\n";
    std::cout << "Unified Memory tiled time: " << managedTimeMs
              << " ms, speedup vs CPU: " << (managedTimeMs > 0.0f ? cpuTimeForSize / managedTimeMs : 0.0)
              << "x, correct: " << (managedCorrect ? "true" : "false") << "\n\n";

    return cpuCorrect && naiveCorrect && tiledCorrect && managedCorrect;
}

int main() {
    int sizes[3][2] = {
        {2048, 1024},
        {4096, 2048},
        {8192, 4096},
    };

    int blockSizes[3][2] = {
        {16, 16},
        {32, 8},
        {32, 32},
    };

    std::ofstream csv("results/timings.csv");
    csv << "implementation,rows,cols,block_x,block_y,time_ms,bandwidth_gb_s,speedup_vs_cpu,correct\n";

    bool allCorrect = true;

    for (int sizeIndex = 0; sizeIndex < 3; sizeIndex++) {
        double cpuTimeForSize = 0.0;

        for (int blockIndex = 0; blockIndex < 3; blockIndex++) {
            int rows = sizes[sizeIndex][0];
            int cols = sizes[sizeIndex][1];
            int blockX = blockSizes[blockIndex][0];
            int blockY = blockSizes[blockIndex][1];

            bool runCpu = blockIndex == 0;
            bool correct = runBenchmark(rows, cols, blockX, blockY, runCpu, cpuTimeForSize, csv);
            allCorrect = allCorrect && correct;
        }
    }

    std::cout << "Benchmark finished\n";
    std::cout << "All correctness checks passed: " << (allCorrect ? "true" : "false") << "\n";
    std::cout << "Results saved to results\\timings.csv\n";

    return allCorrect ? 0 : 1;
}
