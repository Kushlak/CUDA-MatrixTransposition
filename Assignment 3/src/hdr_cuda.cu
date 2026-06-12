#include "hdr.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>

namespace {

constexpr int kBlockSize = 16;

struct DeviceBuffers {
    uchar3* input = nullptr;
    uchar3* output = nullptr;
    size_t bytes = 0;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    ~DeviceBuffers()
    {
        if (input != nullptr) {
            cudaFree(input);
        }
        if (output != nullptr) {
            cudaFree(output);
        }
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
    }
};

__device__ float clamp01Device(float value)
{
    return fminf(1.0f, fmaxf(0.0f, value));
}

__device__ float applyToneOperator(float luminance, HDRParams params)
{
    const float x = fmaxf(0.0f, luminance * params.exposure);

    if (params.toneOperator == ToneOperator::EXPOSURE) {
        return 1.0f - expf(-x);
    }
    if (params.toneOperator == ToneOperator::ACES) {
        return clamp01Device((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f));
    }

    return x / (1.0f + x);
}

__global__ void hdrGlobalKernel(const uchar3* input, uchar3* output, int width, int height, HDRParams params)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = y * width + x;
    const uchar3 pixel = input[index];

    const float b = pixel.x / 255.0f;
    const float g = pixel.y / 255.0f;
    const float r = pixel.z / 255.0f;

    const float luminance = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    const float mapped = powf(clamp01Device(applyToneOperator(luminance, params)), 1.0f / fmaxf(params.gamma, 0.001f));
    const float scale = mapped / fmaxf(luminance, 1e-4f);

    float outR = r * scale;
    float outG = g * scale;
    float outB = b * scale;

    const float gray = mapped;
    outR = gray + params.saturation * (outR - gray);
    outG = gray + params.saturation * (outG - gray);
    outB = gray + params.saturation * (outB - gray);

    output[index] = make_uchar3(
        static_cast<unsigned char>(clamp01Device(outB) * 255.0f),
        static_cast<unsigned char>(clamp01Device(outG) * 255.0f),
        static_cast<unsigned char>(clamp01Device(outR) * 255.0f));
}

__global__ void hdrLocalKernel(const uchar3* input, uchar3* output, int width, int height, HDRParams params)
{
    __shared__ float tile[kBlockSize][kBlockSize];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float luminance = 0.0f;
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;

    if (x < width && y < height) {
        const uchar3 pixel = input[y * width + x];
        b = pixel.x / 255.0f;
        g = pixel.y / 255.0f;
        r = pixel.z / 255.0f;
        luminance = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }

    tile[ty][tx] = luminance;
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    float localSum = 0.0f;
    int localCount = 0;

    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            const int nx = tx + ox;
            const int ny = ty + oy;
            if (nx >= 0 && nx < blockDim.x && ny >= 0 && ny < blockDim.y) {
                localSum += tile[ny][nx];
                ++localCount;
            }
        }
    }

    const float localAverage = localSum / fmaxf(static_cast<float>(localCount), 1.0f);
    const float adapted = luminance / (1.0f + localAverage);
    const float mapped = powf(clamp01Device(applyToneOperator(adapted, params)), 1.0f / fmaxf(params.gamma, 0.001f));
    const float scale = mapped / fmaxf(luminance, 1e-4f);

    float outR = r * scale;
    float outG = g * scale;
    float outB = b * scale;

    const float gray = mapped;
    outR = gray + params.saturation * (outR - gray);
    outG = gray + params.saturation * (outG - gray);
    outB = gray + params.saturation * (outB - gray);

    output[y * width + x] = make_uchar3(
        static_cast<unsigned char>(clamp01Device(outB) * 255.0f),
        static_cast<unsigned char>(clamp01Device(outG) * 255.0f),
        static_cast<unsigned char>(clamp01Device(outR) * 255.0f));
}

bool checkCuda(cudaError_t status, const char* message)
{
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr << message << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool ensureBuffers(DeviceBuffers& buffers, size_t bytes)
{
    if (buffers.bytes == bytes && buffers.input != nullptr && buffers.output != nullptr) {
        return true;
    }

    if (buffers.input != nullptr) {
        cudaFree(buffers.input);
        buffers.input = nullptr;
    }
    if (buffers.output != nullptr) {
        cudaFree(buffers.output);
        buffers.output = nullptr;
    }

    if (!checkCuda(cudaMalloc(&buffers.input, bytes), "cudaMalloc input failed")) {
        buffers.bytes = 0;
        return false;
    }
    if (!checkCuda(cudaMalloc(&buffers.output, bytes), "cudaMalloc output failed")) {
        cudaFree(buffers.input);
        buffers.input = nullptr;
        buffers.bytes = 0;
        return false;
    }

    if (buffers.start == nullptr && !checkCuda(cudaEventCreate(&buffers.start), "cudaEventCreate start failed")) {
        return false;
    }
    if (buffers.stop == nullptr && !checkCuda(cudaEventCreate(&buffers.stop), "cudaEventCreate stop failed")) {
        return false;
    }

    buffers.bytes = bytes;
    return true;
}

} // namespace

bool applyHDRToneMappingGPU(const cv::Mat& input, cv::Mat& output, const HDRParams& params, float* kernelMs)
{
    static DeviceBuffers buffers;

    const cv::Mat continuousInput = input.isContinuous() ? input : input.clone();
    output.create(continuousInput.size(), continuousInput.type());

    if (!output.isContinuous()) {
        output = output.clone();
    }

    const int width = continuousInput.cols;
    const int height = continuousInput.rows;
    const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uchar3);

    if (!ensureBuffers(buffers, bytes)) {
        return false;
    }

    if (!checkCuda(cudaMemcpy(buffers.input, continuousInput.ptr<uchar3>(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy to device failed")) {
        return false;
    }

    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaEventRecord(buffers.start);
    if (params.toneOperator == ToneOperator::LOCAL) {
        hdrLocalKernel<<<grid, block>>>(buffers.input, buffers.output, width, height, params);
    } else {
        hdrGlobalKernel<<<grid, block>>>(buffers.input, buffers.output, width, height, params);
    }
    cudaEventRecord(buffers.stop);
    cudaEventSynchronize(buffers.stop);

    if (!checkCuda(cudaGetLastError(), "HDR kernel launch failed")) {
        return false;
    }

    if (kernelMs != nullptr) {
        cudaEventElapsedTime(kernelMs, buffers.start, buffers.stop);
    }

    if (!checkCuda(cudaMemcpy(output.ptr<uchar3>(), buffers.output, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy to host failed")) {
        return false;
    }

    return true;
}
