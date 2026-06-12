#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

const int MAX_FILTER_SIZE = 7;
const int MAX_FILTER_VALUES = MAX_FILTER_SIZE * MAX_FILTER_SIZE;
const int TIMING_RUNS = 5;

__constant__ float constFilter[MAX_FILTER_VALUES];
__constant__ float constSepFilter[MAX_FILTER_SIZE];

struct BenchmarkRow
{
    std::string name;
    int rows;
    int cols;
    int filterSize;
    double timeMs;
    float error;
};

std::vector<BenchmarkRow> benchmarkRows;

__global__ void warmUpKernel()
{
}

void checkCuda(cudaError_t error, const char* message)
{
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error at " << message << ": "
                  << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

void makeImage(std::vector<float>& A, int rows, int cols)
{
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            int index = row * cols + col;
            A[index] = static_cast<float>((row * 3 + col * 7) % 256) / 255.0f;
        }
    }
}

void makeAverageFilter(std::vector<float>& filter, int filterSize)
{
    filter.assign(filterSize * filterSize, 1.0f / (filterSize * filterSize));
}

void makeSeparableFilter(std::vector<float>& filter1D, int filterSize)
{
    filter1D.assign(filterSize, 0.0f);

    if (filterSize == 3)
    {
        filter1D[0] = 1.0f / 4.0f;
        filter1D[1] = 2.0f / 4.0f;
        filter1D[2] = 1.0f / 4.0f;
    }
    else if (filterSize == 5)
    {
        float values[5] = {1, 4, 6, 4, 1};
        for (int i = 0; i < 5; i++)
        {
            filter1D[i] = values[i] / 16.0f;
        }
    }
    else
    {
        float values[7] = {1, 6, 15, 20, 15, 6, 1};
        for (int i = 0; i < 7; i++)
        {
            filter1D[i] = values[i] / 64.0f;
        }
    }
}

void makeOuterFilter(std::vector<float>& filter2D,
                     const std::vector<float>& filter1D,
                     int filterSize)
{
    filter2D.assign(filterSize * filterSize, 0.0f);

    for (int row = 0; row < filterSize; row++)
    {
        for (int col = 0; col < filterSize; col++)
        {
            filter2D[row * filterSize + col] = filter1D[row] * filter1D[col];
        }
    }
}

void convolutionCPU(const std::vector<float>& A,
                    std::vector<float>& B,
                    const std::vector<float>& filter,
                    int rows,
                    int cols,
                    int filterSize)
{
    int radius = filterSize / 2;

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            float sum = 0.0f;

            for (int fy = 0; fy < filterSize; fy++)
            {
                for (int fx = 0; fx < filterSize; fx++)
                {
                    int imageRow = row + fy - radius;
                    int imageCol = col + fx - radius;

                    if (imageRow >= 0 && imageRow < rows &&
                        imageCol >= 0 && imageCol < cols)
                    {
                        int imageIndex = imageRow * cols + imageCol;
                        int filterIndex = fy * filterSize + fx;
                        sum += A[imageIndex] * filter[filterIndex];
                    }
                }
            }

            B[row * cols + col] = sum;
        }
    }
}

__global__ void convolutionNaiveKernel(const float* A,
                                       float* B,
                                       const float* filter,
                                       int rows,
                                       int cols,
                                       int filterSize)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = filterSize / 2;

    if (row >= rows || col >= cols)
    {
        return;
    }

    float sum = 0.0f;

    for (int fy = 0; fy < filterSize; fy++)
    {
        for (int fx = 0; fx < filterSize; fx++)
        {
            int imageRow = row + fy - radius;
            int imageCol = col + fx - radius;

            if (imageRow >= 0 && imageRow < rows &&
                imageCol >= 0 && imageCol < cols)
            {
                int imageIndex = imageRow * cols + imageCol;
                int filterIndex = fy * filterSize + fx;
                sum += A[imageIndex] * filter[filterIndex];
            }
        }
    }

    B[row * cols + col] = sum;
}

__global__ void convolutionConstantKernel(const float* A,
                                          float* B,
                                          int rows,
                                          int cols,
                                          int filterSize)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = filterSize / 2;

    if (row >= rows || col >= cols)
    {
        return;
    }

    float sum = 0.0f;

    for (int fy = 0; fy < filterSize; fy++)
    {
        for (int fx = 0; fx < filterSize; fx++)
        {
            int imageRow = row + fy - radius;
            int imageCol = col + fx - radius;

            if (imageRow >= 0 && imageRow < rows &&
                imageCol >= 0 && imageCol < cols)
            {
                int imageIndex = imageRow * cols + imageCol;
                int filterIndex = fy * filterSize + fx;
                sum += A[imageIndex] * constFilter[filterIndex];
            }
        }
    }

    B[row * cols + col] = sum;
}

__global__ void convolutionSharedKernel(const float* A,
                                        float* B,
                                        const float* filter,
                                        int rows,
                                        int cols,
                                        int filterSize)
{
    extern __shared__ float tile[];

    int radius = filterSize / 2;
    int sharedCols = blockDim.x + 2 * radius;
    int sharedRows = blockDim.y + 2 * radius;

    int outputCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outputRow = blockIdx.y * blockDim.y + threadIdx.y;

    // The tile includes extra border pixels, called the halo.
    for (int y = threadIdx.y; y < sharedRows; y += blockDim.y)
    {
        for (int x = threadIdx.x; x < sharedCols; x += blockDim.x)
        {
            int imageRow = blockIdx.y * blockDim.y + y - radius;
            int imageCol = blockIdx.x * blockDim.x + x - radius;
            int sharedIndex = y * sharedCols + x;

            if (imageRow >= 0 && imageRow < rows &&
                imageCol >= 0 && imageCol < cols)
            {
                tile[sharedIndex] = A[imageRow * cols + imageCol];
            }
            else
            {
                tile[sharedIndex] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (outputRow >= rows || outputCol >= cols)
    {
        return;
    }

    float sum = 0.0f;

    for (int fy = 0; fy < filterSize; fy++)
    {
        for (int fx = 0; fx < filterSize; fx++)
        {
            int sharedRow = threadIdx.y + fy;
            int sharedCol = threadIdx.x + fx;
            int sharedIndex = sharedRow * sharedCols + sharedCol;
            int filterIndex = fy * filterSize + fx;
            sum += tile[sharedIndex] * filter[filterIndex];
        }
    }

    B[outputRow * cols + outputCol] = sum;
}

__global__ void separableHorizontalKernel(const float* A,
                                          float* temp,
                                          int rows,
                                          int cols,
                                          int filterSize)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = filterSize / 2;

    if (row >= rows || col >= cols)
    {
        return;
    }

    float sum = 0.0f;

    for (int i = 0; i < filterSize; i++)
    {
        int imageCol = col + i - radius;

        if (imageCol >= 0 && imageCol < cols)
        {
            sum += A[row * cols + imageCol] * constSepFilter[i];
        }
    }

    temp[row * cols + col] = sum;
}

__global__ void separableVerticalKernel(const float* temp,
                                        float* B,
                                        int rows,
                                        int cols,
                                        int filterSize)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = filterSize / 2;

    if (row >= rows || col >= cols)
    {
        return;
    }

    float sum = 0.0f;

    for (int i = 0; i < filterSize; i++)
    {
        int imageRow = row + i - radius;

        if (imageRow >= 0 && imageRow < rows)
        {
            sum += temp[imageRow * cols + col] * constSepFilter[i];
        }
    }

    B[row * cols + col] = sum;
}

float maxError(const std::vector<float>& cpu, const std::vector<float>& gpu)
{
    float error = 0.0f;

    for (size_t i = 0; i < cpu.size(); i++)
    {
        float current = std::fabs(cpu[i] - gpu[i]);
        if (current > error)
        {
            error = current;
        }
    }

    return error;
}

double runCpu(const std::vector<float>& A,
              std::vector<float>& B,
              const std::vector<float>& filter,
              int rows,
              int cols,
              int filterSize)
{
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TIMING_RUNS; i++)
    {
        convolutionCPU(A, B, filter, rows, cols, filterSize);
    }
    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> time = stop - start;
    return time.count() / TIMING_RUNS;
}

float runNaiveGpu(const std::vector<float>& A,
                  std::vector<float>& B,
                  const std::vector<float>& filter,
                  int rows,
                  int cols,
                  int filterSize)
{
    int N = rows * cols;
    float* dA = nullptr;
    float* dB = nullptr;
    float* dFilter = nullptr;

    checkCuda(cudaMalloc(&dA, N * sizeof(float)), "cudaMalloc dA");
    checkCuda(cudaMalloc(&dB, N * sizeof(float)), "cudaMalloc dB");
    checkCuda(cudaMalloc(&dFilter, filter.size() * sizeof(float)), "cudaMalloc dFilter");

    checkCuda(cudaMemcpy(dA, A.data(), N * sizeof(float), cudaMemcpyHostToDevice), "copy image to GPU");
    checkCuda(cudaMemcpy(dFilter, filter.data(), filter.size() * sizeof(float), cudaMemcpyHostToDevice), "copy filter to GPU");

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start;
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    convolutionNaiveKernel<<<blocks, threadsPerBlock>>>(dA, dB, dFilter, rows, cols, filterSize);
    checkCuda(cudaDeviceSynchronize(), "warm up naive");

    checkCuda(cudaEventRecord(start), "record start");
    for (int i = 0; i < TIMING_RUNS; i++)
    {
        convolutionNaiveKernel<<<blocks, threadsPerBlock>>>(dA, dB, dFilter, rows, cols, filterSize);
    }
    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop), "sync stop");
    checkCuda(cudaGetLastError(), "naive kernel");

    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed naive");
    milliseconds = milliseconds / TIMING_RUNS;

    checkCuda(cudaMemcpy(B.data(), dB, N * sizeof(float), cudaMemcpyDeviceToHost), "copy result to CPU");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dFilter);

    return milliseconds;
}

float runSharedGpu(const std::vector<float>& A,
                   std::vector<float>& B,
                   const std::vector<float>& filter,
                   int rows,
                   int cols,
                   int filterSize,
                   int tileX,
                   int tileY)
{
    int N = rows * cols;
    int radius = filterSize / 2;
    float* dA = nullptr;
    float* dB = nullptr;
    float* dFilter = nullptr;

    checkCuda(cudaMalloc(&dA, N * sizeof(float)), "cudaMalloc dA");
    checkCuda(cudaMalloc(&dB, N * sizeof(float)), "cudaMalloc dB");
    checkCuda(cudaMalloc(&dFilter, filter.size() * sizeof(float)), "cudaMalloc dFilter");

    checkCuda(cudaMemcpy(dA, A.data(), N * sizeof(float), cudaMemcpyHostToDevice), "copy image to GPU");
    checkCuda(cudaMemcpy(dFilter, filter.data(), filter.size() * sizeof(float), cudaMemcpyHostToDevice), "copy filter to GPU");

    dim3 threadsPerBlock(tileX, tileY);
    dim3 blocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    int sharedCols = threadsPerBlock.x + 2 * radius;
    int sharedRows = threadsPerBlock.y + 2 * radius;
    int sharedBytes = sharedCols * sharedRows * sizeof(float);

    cudaEvent_t start;
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    convolutionSharedKernel<<<blocks, threadsPerBlock, sharedBytes>>>(dA, dB, dFilter, rows, cols, filterSize);
    checkCuda(cudaDeviceSynchronize(), "warm up shared");

    checkCuda(cudaEventRecord(start), "record start");
    for (int i = 0; i < TIMING_RUNS; i++)
    {
        convolutionSharedKernel<<<blocks, threadsPerBlock, sharedBytes>>>(dA, dB, dFilter, rows, cols, filterSize);
    }
    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop), "sync stop");
    checkCuda(cudaGetLastError(), "shared kernel");

    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed shared");
    milliseconds = milliseconds / TIMING_RUNS;

    checkCuda(cudaMemcpy(B.data(), dB, N * sizeof(float), cudaMemcpyDeviceToHost), "copy result to CPU");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dFilter);

    return milliseconds;
}

float runConstantGpu(const std::vector<float>& A,
                     std::vector<float>& B,
                     const std::vector<float>& filter,
                     int rows,
                     int cols,
                     int filterSize)
{
    int N = rows * cols;
    float* dA = nullptr;
    float* dB = nullptr;

    checkCuda(cudaMalloc(&dA, N * sizeof(float)), "cudaMalloc dA");
    checkCuda(cudaMalloc(&dB, N * sizeof(float)), "cudaMalloc dB");

    checkCuda(cudaMemcpy(dA, A.data(), N * sizeof(float), cudaMemcpyHostToDevice), "copy image to GPU");
    checkCuda(cudaMemcpyToSymbol(constFilter, filter.data(), filter.size() * sizeof(float)), "copy filter to constant memory");

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start;
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    convolutionConstantKernel<<<blocks, threadsPerBlock>>>(dA, dB, rows, cols, filterSize);
    checkCuda(cudaDeviceSynchronize(), "warm up constant");

    checkCuda(cudaEventRecord(start), "record start");
    for (int i = 0; i < TIMING_RUNS; i++)
    {
        convolutionConstantKernel<<<blocks, threadsPerBlock>>>(dA, dB, rows, cols, filterSize);
    }
    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop), "sync stop");
    checkCuda(cudaGetLastError(), "constant kernel");

    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed constant");
    milliseconds = milliseconds / TIMING_RUNS;

    checkCuda(cudaMemcpy(B.data(), dB, N * sizeof(float), cudaMemcpyDeviceToHost), "copy result to CPU");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);

    return milliseconds;
}

float runSeparableGpu(const std::vector<float>& A,
                      std::vector<float>& B,
                      const std::vector<float>& filter1D,
                      int rows,
                      int cols,
                      int filterSize)
{
    int N = rows * cols;
    float* dA = nullptr;
    float* dTemp = nullptr;
    float* dB = nullptr;

    checkCuda(cudaMalloc(&dA, N * sizeof(float)), "cudaMalloc dA");
    checkCuda(cudaMalloc(&dTemp, N * sizeof(float)), "cudaMalloc dTemp");
    checkCuda(cudaMalloc(&dB, N * sizeof(float)), "cudaMalloc dB");

    checkCuda(cudaMemcpy(dA, A.data(), N * sizeof(float), cudaMemcpyHostToDevice), "copy image to GPU");
    checkCuda(cudaMemcpyToSymbol(constSepFilter, filter1D.data(), filter1D.size() * sizeof(float)), "copy separable filter");

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start;
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    separableHorizontalKernel<<<blocks, threadsPerBlock>>>(dA, dTemp, rows, cols, filterSize);
    separableVerticalKernel<<<blocks, threadsPerBlock>>>(dTemp, dB, rows, cols, filterSize);
    checkCuda(cudaDeviceSynchronize(), "warm up separable");

    checkCuda(cudaEventRecord(start), "record start");
    for (int i = 0; i < TIMING_RUNS; i++)
    {
        separableHorizontalKernel<<<blocks, threadsPerBlock>>>(dA, dTemp, rows, cols, filterSize);
        separableVerticalKernel<<<blocks, threadsPerBlock>>>(dTemp, dB, rows, cols, filterSize);
    }
    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop), "sync stop");
    checkCuda(cudaGetLastError(), "separable kernels");

    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed separable");
    milliseconds = milliseconds / TIMING_RUNS;

    checkCuda(cudaMemcpy(B.data(), dB, N * sizeof(float), cudaMemcpyDeviceToHost), "copy result to CPU");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dTemp);
    cudaFree(dB);

    return milliseconds;
}

void writeResult(std::ofstream& file,
                 const std::string& name,
                 int rows,
                 int cols,
                 int filterSize,
                 double timeMs,
                 float error)
{
    std::cout << std::left << std::setw(18) << name
              << " size=" << rows << "x" << cols
              << " filter=" << filterSize << "x" << filterSize
              << " avgTime=" << std::fixed << std::setprecision(3) << timeMs << " ms"
              << " maxError=" << std::setprecision(6) << error << std::endl;

    file << name << ","
         << rows << ","
         << cols << ","
         << filterSize << ","
         << std::fixed << std::setprecision(6) << timeMs << ","
         << error << "\n";

    BenchmarkRow row;
    row.name = name;
    row.rows = rows;
    row.cols = cols;
    row.filterSize = filterSize;
    row.timeMs = timeMs;
    row.error = error;
    benchmarkRows.push_back(row);
}

double findTime(const std::string& name, int rows, int cols, int filterSize)
{
    for (size_t i = 0; i < benchmarkRows.size(); i++)
    {
        if (benchmarkRows[i].name == name &&
            benchmarkRows[i].rows == rows &&
            benchmarkRows[i].cols == cols &&
            benchmarkRows[i].filterSize == filterSize)
        {
            return benchmarkRows[i].timeMs;
        }
    }

    return 0.0;
}

void runDirectBenchmark(std::ofstream& file, int rows, int cols, int filterSize)
{
    int N = rows * cols;
    std::vector<float> A(N);
    std::vector<float> cpu(N);
    std::vector<float> gpu(N);
    std::vector<float> filter;

    makeImage(A, rows, cols);
    makeAverageFilter(filter, filterSize);

    std::cout << "\nDirect convolution benchmark" << std::endl;

    double cpuTime = runCpu(A, cpu, filter, rows, cols, filterSize);
    writeResult(file, "CPU", rows, cols, filterSize, cpuTime, 0.0f);

    float naiveTime = runNaiveGpu(A, gpu, filter, rows, cols, filterSize);
    writeResult(file, "Naive GPU", rows, cols, filterSize, naiveTime, maxError(cpu, gpu));

    float shared8Time = runSharedGpu(A, gpu, filter, rows, cols, filterSize, 8, 8);
    writeResult(file, "Shared GPU 8x8", rows, cols, filterSize, shared8Time, maxError(cpu, gpu));

    float shared16Time = runSharedGpu(A, gpu, filter, rows, cols, filterSize, 16, 16);
    writeResult(file, "Shared GPU 16x16", rows, cols, filterSize, shared16Time, maxError(cpu, gpu));

    float shared32x8Time = runSharedGpu(A, gpu, filter, rows, cols, filterSize, 32, 8);
    writeResult(file, "Shared GPU 32x8", rows, cols, filterSize, shared32x8Time, maxError(cpu, gpu));

    float constantTime = runConstantGpu(A, gpu, filter, rows, cols, filterSize);
    writeResult(file, "Constant GPU", rows, cols, filterSize, constantTime, maxError(cpu, gpu));
}

void runSeparableBenchmark(std::ofstream& file, int rows, int cols, int filterSize)
{
    int N = rows * cols;
    std::vector<float> A(N);
    std::vector<float> cpu(N);
    std::vector<float> gpu(N);
    std::vector<float> filter1D;
    std::vector<float> filter2D;

    makeImage(A, rows, cols);
    makeSeparableFilter(filter1D, filterSize);
    makeOuterFilter(filter2D, filter1D, filterSize);

    std::cout << "\nSeparable convolution benchmark" << std::endl;

    double cpuTime = runCpu(A, cpu, filter2D, rows, cols, filterSize);
    writeResult(file, "CPU sep check", rows, cols, filterSize, cpuTime, 0.0f);

    float separableTime = runSeparableGpu(A, gpu, filter1D, rows, cols, filterSize);
    writeResult(file, "Separable GPU", rows, cols, filterSize, separableTime, maxError(cpu, gpu));
}

int main()
{
    std::ofstream file("results/timings.csv");
    file << "implementation,rows,cols,filter_size,avg_time_ms,max_error\n";

    std::cout << "Image convolution CUDA assignment" << std::endl;
    std::cout << "Boundary handling: zero padding" << std::endl;
    std::cout << "Correctness: each GPU result is compared with CPU result" << std::endl;
    std::cout << "Timing: average of " << TIMING_RUNS << " timed runs" << std::endl;

    // This starts CUDA before the first timed kernel.
    // Without this, the first GPU timing can include one-time startup cost.
    checkCuda(cudaFree(0), "CUDA warm up");
    warmUpKernel<<<1, 1>>>();
    checkCuda(cudaDeviceSynchronize(), "warm up kernel");

    int sizes[3] = {512, 1024, 2048};

    std::cout << "\nSmall benchmark set: 3x3 filter" << std::endl;
    for (int i = 0; i < 3; i++)
    {
        int rows = sizes[i];
        int cols = sizes[i];
        runDirectBenchmark(file, rows, cols, 3);
    }

    std::cout << "\nSeparable filter version: 1024x1024 image" << std::endl;
    runSeparableBenchmark(file, 1024, 1024, 3);

    std::cout << "\nExtended benchmarks: 1024x1024 image with 3x3, 5x5, 7x7 filters" << std::endl;
    for (int filterSize = 3; filterSize <= 7; filterSize += 2)
    {
        runDirectBenchmark(file, 1024, 1024, filterSize);
        runSeparableBenchmark(file, 1024, 1024, filterSize);
    }

    file.close();

    std::cout << "\nSaved results to results/timings.csv" << std::endl;
    return 0;
}
