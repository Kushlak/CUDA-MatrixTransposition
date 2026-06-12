#include "kernels.h"
#include <cuda_runtime.h>
#include <plog/Log.h>
#include <vector>
#include <algorithm>
#include "../utils/filter_utils.h"

namespace cuda_filter
{

#define CHECK_CUDA_ERROR(call)                                                          \
    {                                                                                   \
        cudaError_t err = call;                                                         \
        if (err != cudaSuccess)                                                         \
        {                                                                               \
            PLOG_ERROR << "CUDA error in " << #call << ": " << cudaGetErrorString(err); \
            return;                                                                     \
        }                                                                               \
    }

    __global__ void convolutionKernel(const unsigned char *input, unsigned char *output,
                                      const float *kernel, int width, int height,
                                      int channels, int kernelSize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int radius = kernelSize / 2;

        for (int c = 0; c < channels; c++)
        {
            float sum = 0.0f;

            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);

                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];
                    float pixelValue = input[(iy * width + ix) * channels + c];

                    sum += pixelValue * kernelValue;
                }
            }

            output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
        }
    }

    __global__ void convolutionBandKernel(const unsigned char *input, unsigned char *output,
                                          const float *kernel, int width, int height,
                                          int channels, int kernelSize,
                                          int yStart, int yEnd)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = yStart + blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= yEnd || y >= height)
            return;

        int radius = kernelSize / 2;

        for (int c = 0; c < channels; c++)
        {
            float sum = 0.0f;

            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);

                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];
                    float pixelValue = input[(iy * width + ix) * channels + c];

                    sum += pixelValue * kernelValue;
                }
            }

            output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
        }
    }

    __global__ void wipeKernel(const unsigned char *left, const unsigned char *right,
                               unsigned char *output, int width, int height,
                               int channels, float progress)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        bool useRight = x < (int)(width * progress);
        int i = (y * width + x) * channels;

        for (int c = 0; c < channels; c++)
            output[i + c] = useRight ? right[i + c] : left[i + c];
    }

    struct PipelineDeviceMemory
    {
        unsigned char *ping = nullptr;
        unsigned char *pong = nullptr;
        unsigned char *extra = nullptr;
        float *kernelA = nullptr;
        float *kernelB = nullptr;
        size_t imageBytes = 0;
        size_t kernelBytes = 0;
    };

    static PipelineDeviceMemory g_pipelineMem;

    static void ensurePipelineMemory(size_t imageBytes, size_t kernelBytes)
    {
        if (g_pipelineMem.imageBytes < imageBytes)
        {
            cudaFree(g_pipelineMem.ping);
            cudaFree(g_pipelineMem.pong);
            cudaFree(g_pipelineMem.extra);
            cudaMalloc(&g_pipelineMem.ping, imageBytes);
            cudaMalloc(&g_pipelineMem.pong, imageBytes);
            cudaMalloc(&g_pipelineMem.extra, imageBytes);
            g_pipelineMem.imageBytes = imageBytes;
        }

        if (g_pipelineMem.kernelBytes < kernelBytes)
        {
            cudaFree(g_pipelineMem.kernelA);
            cudaFree(g_pipelineMem.kernelB);
            cudaMalloc(&g_pipelineMem.kernelA, kernelBytes);
            cudaMalloc(&g_pipelineMem.kernelB, kernelBytes);
            g_pipelineMem.kernelBytes = kernelBytes;
        }
    }

    static void uploadKernel(const std::string &name, int kernelSize, float intensity,
                             float *deviceKernel, cudaStream_t stream)
    {
        FilterType type = FilterUtils::stringToFilterType(name);
        if (type == FilterType::HDR_TONEMAPPING)
            type = FilterType::IDENTITY;

        cv::Mat kernel = FilterUtils::createFilterKernel(type, kernelSize, intensity);
        cudaMemcpyAsync(deviceKernel, kernel.ptr<float>(),
                        kernelSize * kernelSize * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }

    static void runConvolutionStage(const unsigned char *input, unsigned char *output,
                                    float *deviceKernel, int width, int height,
                                    int channels, int kernelSize, cudaStream_t stream)
    {
        dim3 blockDim(16, 16);
        dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));
        convolutionKernel<<<gridDim, blockDim, 0, stream>>>(input, output, deviceKernel,
                                                           width, height, channels,
                                                           kernelSize);
    }

    static void runConvolutionBandStage(const unsigned char *input, unsigned char *output,
                                        float *deviceKernel, int width, int height,
                                        int channels, int kernelSize,
                                        int yStart, int yEnd, cudaStream_t stream)
    {
        dim3 blockDim(16, 16);
        dim3 gridDim(cuda::divUp(width, blockDim.x),
                     cuda::divUp(yEnd - yStart, blockDim.y));
        convolutionBandKernel<<<gridDim, blockDim, 0, stream>>>(input, output,
                                                               deviceKernel,
                                                               width, height,
                                                               channels,
                                                               kernelSize,
                                                               yStart, yEnd);
    }

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        output.create(input.size(), input.type());

        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;

        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;
        float *d_kernel = nullptr;

        size_t imageSize = width * height * channels * sizeof(unsigned char);
        size_t kernelSize_bytes = kernelSize * kernelSize * sizeof(float);

        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
            for (int j = 0; j < kernelSize; j++)
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);

        CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernelSize_bytes));

        CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernelSize_bytes, cudaMemcpyHostToDevice));

        dim3 blockDim(16, 16);
        dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));

        convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, width, height, channels, kernelSize);

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);
        delete[] h_kernel;
    }

    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        output.create(input.size(), input.type());

        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;
        int radius = kernelSize / 2;

        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
            for (int j = 0; j < kernelSize; j++)
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    float sum = 0.0f;

                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            int ix = std::min(std::max(x + kx, 0), width - 1);
                            int iy = std::min(std::max(y + ky, 0), height - 1);

                            float kernelValue = h_kernel[(ky + radius) * kernelSize + (kx + radius)];
                            float pixelValue = input.at<cv::Vec3b>(iy, ix)[c];

                            sum += pixelValue * kernelValue;
                        }
                    }

                    output.at<cv::Vec3b>(y, x)[c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
                }
            }
        }

        delete[] h_kernel;
    }


#define HDR_TILE_W 16
#define HDR_TILE_H 16
#define HDR_HALO   4
#define HDR_SH_W   (HDR_TILE_W + 2 * HDR_HALO)
#define HDR_SH_H   (HDR_TILE_H + 2 * HDR_HALO)

// OpenCV is BGR: channel 0=B, 1=G, 2=R — Rec. 709 coefficients applied accordingly
__host__ __device__ static inline float hdr_bgr2lum(float b, float g, float r)
{
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

__host__ __device__ static inline float hdr_clamp01(float v)
{
    return (v >= 0.0f) ? (v <= 1.0f ? v : 1.0f) : 0.0f;
}

__host__ __device__ static inline unsigned char hdr_toUchar(float v)
{
    return static_cast<unsigned char>(hdr_clamp01(v) * 255.0f + 0.5f);
}

__host__ __device__ static inline void hdr_satGamma(float &c0, float &c1, float &c2,
                                                     float sat, float invGamma)
{
    float lum = hdr_bgr2lum(c0, c1, c2);
    c0 = lum + sat * (c0 - lum);
    c1 = lum + sat * (c1 - lum);
    c2 = lum + sat * (c2 - lum);
    c0 = powf(c0 < 0.0f ? 0.0f : c0, invGamma);
    c1 = powf(c1 < 0.0f ? 0.0f : c1, invGamma);
    c2 = powf(c2 < 0.0f ? 0.0f : c2, invGamma);
}

// Narkowicz 2015 ACES RRT+ODT approximation
__host__ __device__ static inline float hdr_acesFilmic(float x)
{
    return hdr_clamp01((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f));
}

// Reinhard et al. 2002 extended operator with white-point
__global__ static void k_hdr_reinhardGlobal(const unsigned char *__restrict__ in,
                                              unsigned char *__restrict__ out,
                                              int W, int H,
                                              float exposure, float white2,
                                              float sat, float invGamma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i = (y * W + x) * 3;
    float c0 = in[i+0] / 255.0f * exposure;
    float c1 = in[i+1] / 255.0f * exposure;
    float c2 = in[i+2] / 255.0f * exposure;

    float L     = hdr_bgr2lum(c0, c1, c2);
    float L_out = L * (1.0f + L / white2) / (1.0f + L);
    float scale = (L > 1e-6f) ? (L_out / L) : 0.0f;
    c0 *= scale; c1 *= scale; c2 *= scale;

    hdr_satGamma(c0, c1, c2, sat, invGamma);
    out[i+0] = hdr_toUchar(c0);
    out[i+1] = hdr_toUchar(c1);
    out[i+2] = hdr_toUchar(c2);
}

__global__ static void k_hdr_aces(const unsigned char *__restrict__ in,
                                   unsigned char *__restrict__ out,
                                   int W, int H,
                                   float exposure, float sat, float invGamma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i = (y * W + x) * 3;
    float c0 = hdr_acesFilmic(in[i+0] / 255.0f * exposure);
    float c1 = hdr_acesFilmic(in[i+1] / 255.0f * exposure);
    float c2 = hdr_acesFilmic(in[i+2] / 255.0f * exposure);

    hdr_satGamma(c0, c1, c2, sat, invGamma);
    out[i+0] = hdr_toUchar(c0);
    out[i+1] = hdr_toUchar(c1);
    out[i+2] = hdr_toUchar(c2);
}

__global__ static void k_hdr_toFloat(const unsigned char *__restrict__ in,
                                      float *__restrict__ out,
                                      int W, int H, float exposure)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i = (y * W + x) * 3;
    out[i+0] = in[i+0] / 255.0f * exposure;
    out[i+1] = in[i+1] / 255.0f * exposure;
    out[i+2] = in[i+2] / 255.0f * exposure;
}

__global__ static void k_hdr_localReinhard(const float *__restrict__ in,
                                            unsigned char *__restrict__ out,
                                            int W, int H,
                                            float sat, float invGamma)
{
    __shared__ float shLum[HDR_SH_H][HDR_SH_W];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * HDR_TILE_W;
    int by = blockIdx.y * HDR_TILE_H;

    for (int dy = ty; dy < HDR_SH_H; dy += HDR_TILE_H)
        for (int dx = tx; dx < HDR_SH_W; dx += HDR_TILE_W)
        {
            int gx = min(max(bx + dx - HDR_HALO, 0), W - 1);
            int gy = min(max(by + dy - HDR_HALO, 0), H - 1);
            int gi = (gy * W + gx) * 3;
            shLum[dy][dx] = hdr_bgr2lum(in[gi+0], in[gi+1], in[gi+2]);
        }
    __syncthreads();

    int x = bx + tx, y = by + ty;
    if (x >= W || y >= H) return;

    float sumL = 0.0f;
    #pragma unroll
    for (int dy = 0; dy < 2 * HDR_HALO + 1; ++dy)
        #pragma unroll
        for (int dx = 0; dx < 2 * HDR_HALO + 1; ++dx)
            sumL += shLum[ty + dy][tx + dx];

    float L_local = sumL / ((float)((2 * HDR_HALO + 1) * (2 * HDR_HALO + 1)));

    int gi = (y * W + x) * 3;
    float c0 = in[gi+0], c1 = in[gi+1], c2 = in[gi+2];
    float scale = 1.0f / (1.0f + L_local);
    c0 *= scale; c1 *= scale; c2 *= scale;

    hdr_satGamma(c0, c1, c2, sat, invGamma);
    out[gi+0] = hdr_toUchar(c0);
    out[gi+1] = hdr_toUchar(c1);
    out[gi+2] = hdr_toUchar(c2);
}

void applyHDRFilterGPU(const cv::Mat &input, cv::Mat &output, const HdrOptions &opts)
{
    if (input.empty() || input.channels() != 3)
    {
        PLOG_ERROR << "HDR filter requires a non-empty 3-channel image";
        return;
    }

    output.create(input.size(), input.type());

    int W = input.cols, H = input.rows;
    size_t imgBytes   = (size_t)W * H * 3;
    size_t floatBytes = imgBytes * sizeof(float);

    unsigned char *d_in = nullptr, *d_out = nullptr;
    float         *d_float = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_in,  imgBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, imgBytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, input.data, imgBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(cuda::divUp(W, 16), cuda::divUp(H, 16));

    float invGamma = 1.0f / opts.gamma;
    float white2   = opts.whitePoint * opts.whitePoint;

    if (opts.algorithm == "aces")
    {
        k_hdr_aces<<<grid, block>>>(d_in, d_out, W, H,
                                    opts.exposure, opts.saturation, invGamma);
    }
    else if (opts.algorithm == "local")
    {
        CHECK_CUDA_ERROR(cudaMalloc(&d_float, floatBytes));
        k_hdr_toFloat<<<grid, block>>>(d_in, d_float, W, H, opts.exposure);
        k_hdr_localReinhard<<<grid, block>>>(d_float, d_out, W, H,
                                             opts.saturation, invGamma);
        cudaFree(d_float);
    }
    else
    {
        k_hdr_reinhardGlobal<<<grid, block>>>(d_in, d_out, W, H,
                                              opts.exposure, white2,
                                              opts.saturation, invGamma);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_out, imgBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
}

void applyHDRFilterCPU(const cv::Mat &input, cv::Mat &output, const HdrOptions &opts)
{
    if (input.empty() || input.channels() != 3)
    {
        PLOG_ERROR << "HDR filter requires a non-empty 3-channel image";
        return;
    }

    output.create(input.size(), input.type());

    int W = input.cols, H = input.rows;
    float invGamma = 1.0f / opts.gamma;
    float white2   = opts.whitePoint * opts.whitePoint;

    const unsigned char *in  = input.data;
    unsigned char       *out = output.data;

    if (opts.algorithm == "aces")
    {
        for (int y = 0; y < H; ++y)
        for (int x = 0; x < W;  ++x)
        {
            int i = (y * W + x) * 3;
            float c0 = hdr_acesFilmic(in[i+0] / 255.0f * opts.exposure);
            float c1 = hdr_acesFilmic(in[i+1] / 255.0f * opts.exposure);
            float c2 = hdr_acesFilmic(in[i+2] / 255.0f * opts.exposure);
            hdr_satGamma(c0, c1, c2, opts.saturation, invGamma);
            out[i+0] = hdr_toUchar(c0);
            out[i+1] = hdr_toUchar(c1);
            out[i+2] = hdr_toUchar(c2);
        }
    }
    else if (opts.algorithm == "local")
    {
        std::vector<float> fbuf((size_t)W * H * 3);
        for (int j = 0; j < W * H * 3; ++j)
            fbuf[j] = in[j] / 255.0f * opts.exposure;

        constexpr int WIN = (2 * HDR_HALO + 1) * (2 * HDR_HALO + 1);

        for (int y = 0; y < H; ++y)
        for (int x = 0; x < W;  ++x)
        {
            float sumL = 0.0f;
            for (int dy = -HDR_HALO; dy <= HDR_HALO; ++dy)
            for (int dx = -HDR_HALO; dx <= HDR_HALO; ++dx)
            {
                int ix = std::min(std::max(x + dx, 0), W - 1);
                int iy = std::min(std::max(y + dy, 0), H - 1);
                int ii = (iy * W + ix) * 3;
                sumL += hdr_bgr2lum(fbuf[ii+0], fbuf[ii+1], fbuf[ii+2]);
            }
            float L_local = sumL / WIN;
            int i = (y * W + x) * 3;
            float c0 = fbuf[i+0], c1 = fbuf[i+1], c2 = fbuf[i+2];
            float scale = 1.0f / (1.0f + L_local);
            c0 *= scale; c1 *= scale; c2 *= scale;
            hdr_satGamma(c0, c1, c2, opts.saturation, invGamma);
            out[i+0] = hdr_toUchar(c0);
            out[i+1] = hdr_toUchar(c1);
            out[i+2] = hdr_toUchar(c2);
        }
    }
    else
    {
        for (int y = 0; y < H; ++y)
        for (int x = 0; x < W;  ++x)
        {
            int i = (y * W + x) * 3;
            float c0 = in[i+0] / 255.0f * opts.exposure;
            float c1 = in[i+1] / 255.0f * opts.exposure;
            float c2 = in[i+2] / 255.0f * opts.exposure;
            float L     = hdr_bgr2lum(c0, c1, c2);
            float L_out = L * (1.0f + L / white2) / (1.0f + L);
            float scale = (L > 1e-6f) ? (L_out / L) : 0.0f;
            c0 *= scale; c1 *= scale; c2 *= scale;
            hdr_satGamma(c0, c1, c2, opts.saturation, invGamma);
            out[i+0] = hdr_toUchar(c0);
            out[i+1] = hdr_toUchar(c1);
            out[i+2] = hdr_toUchar(c2);
        }
    }
}

void applyPipelineGPU(const cv::Mat &input, cv::Mat &output,
                      const std::vector<std::string> &filters,
                      int kernelSize, float intensity,
                      bool multiStream, PipelinePerf &perf)
{
    if (input.empty())
    {
        PLOG_ERROR << "Pipeline input image is empty";
        return;
    }

    output.create(input.size(), input.type());
    perf = PipelinePerf{};
    perf.stageMs.resize(filters.size(), 0.0f);
    if (kernelSize % 2 == 0)
        kernelSize++;

    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    size_t imageBytes = (size_t)width * height * channels;
    size_t kernelBytes = (size_t)kernelSize * kernelSize * sizeof(float);
    ensurePipelineMemory(imageBytes, kernelBytes);

    const int streamCount = multiStream ? 4 : 1;
    std::vector<cudaStream_t> streams(streamCount, nullptr);
    if (multiStream)
    {
        for (int i = 0; i < streamCount; ++i)
            cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t totalStart, totalEnd, start, end, kernelReady;
    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalEnd);
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&kernelReady);

    cudaEventRecord(totalStart, streams[0]);
    cudaEventRecord(start, streams[0]);
    cudaMemcpyAsync(g_pipelineMem.ping, input.data, imageBytes,
                    cudaMemcpyHostToDevice, streams[0]);
    cudaEventRecord(end, streams[0]);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&perf.uploadMs, start, end);

    unsigned char *src = g_pipelineMem.ping;
    unsigned char *dst = g_pipelineMem.pong;

    for (size_t i = 0; i < filters.size(); ++i)
    {
        cudaEventRecord(start, streams[0]);
        uploadKernel(filters[i], kernelSize, intensity, g_pipelineMem.kernelA, streams[0]);

        if (multiStream)
        {
            cudaEventRecord(kernelReady, streams[0]);
            for (int s = 0; s < streamCount; ++s)
            {
                cudaStreamWaitEvent(streams[s], kernelReady, 0);
                int yStart = s * height / streamCount;
                int yEnd = (s + 1) * height / streamCount;
                runConvolutionBandStage(src, dst, g_pipelineMem.kernelA, width, height,
                                        channels, kernelSize, yStart, yEnd, streams[s]);
            }

            for (int s = 0; s < streamCount; ++s)
                cudaStreamSynchronize(streams[s]);
        }
        else
        {
            runConvolutionStage(src, dst, g_pipelineMem.kernelA, width, height,
                                channels, kernelSize, streams[0]);
        }

        cudaEventRecord(end, streams[0]);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&perf.stageMs[i], start, end);
        std::swap(src, dst);
    }

    cudaEventRecord(start, streams[0]);
    cudaMemcpyAsync(output.data, src, imageBytes, cudaMemcpyDeviceToHost, streams[0]);
    cudaEventRecord(end, streams[0]);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&perf.downloadMs, start, end);

    cudaEventRecord(totalEnd, streams[0]);
    cudaEventSynchronize(totalEnd);
    cudaEventElapsedTime(&perf.totalMs, totalStart, totalEnd);

    cudaEventDestroy(totalStart);
    cudaEventDestroy(totalEnd);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaEventDestroy(kernelReady);
    if (multiStream)
    {
        for (int i = 0; i < streamCount; ++i)
            cudaStreamDestroy(streams[i]);
    }
}

void applyWipeTransitionGPU(const cv::Mat &input, cv::Mat &output,
                            const std::string &fromFilter,
                            const std::string &toFilter,
                            int kernelSize, float intensity,
                            float progress, bool multiStream,
                            PipelinePerf &perf)
{
    if (input.empty())
    {
        PLOG_ERROR << "Transition input image is empty";
        return;
    }

    output.create(input.size(), input.type());
    perf = PipelinePerf{};
    perf.stageMs.resize(3, 0.0f);
    if (kernelSize % 2 == 0)
        kernelSize++;

    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    size_t imageBytes = (size_t)width * height * channels;
    size_t kernelBytes = (size_t)kernelSize * kernelSize * sizeof(float);
    ensurePipelineMemory(imageBytes, kernelBytes);

    progress = std::min(std::max(progress, 0.0f), 1.0f);

    cudaStream_t fromStream = nullptr;
    cudaStream_t toStream = nullptr;
    cudaStream_t mixStream = nullptr;
    if (multiStream)
    {
        cudaStreamCreate(&fromStream);
        cudaStreamCreate(&toStream);
        cudaStreamCreate(&mixStream);
    }

    cudaEvent_t totalStart, totalEnd, start, end, uploadDone;
    cudaEvent_t fromStart, fromDone, toStart, toDone;
    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalEnd);
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&uploadDone);
    cudaEventCreate(&fromStart);
    cudaEventCreate(&fromDone);
    cudaEventCreate(&toStart);
    cudaEventCreate(&toDone);

    cudaEventRecord(totalStart, mixStream);
    cudaEventRecord(start, mixStream);
    cudaMemcpyAsync(g_pipelineMem.ping, input.data, imageBytes,
                    cudaMemcpyHostToDevice, mixStream);
    cudaEventRecord(uploadDone, mixStream);
    cudaEventRecord(end, mixStream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&perf.uploadMs, start, end);

    cudaStreamWaitEvent(fromStream, uploadDone, 0);
    cudaStreamWaitEvent(toStream, uploadDone, 0);

    cudaEventRecord(fromStart, fromStream);
    uploadKernel(fromFilter, kernelSize, intensity, g_pipelineMem.kernelA, fromStream);
    runConvolutionStage(g_pipelineMem.ping, g_pipelineMem.pong, g_pipelineMem.kernelA,
                        width, height, channels, kernelSize, fromStream);
    cudaEventRecord(fromDone, fromStream);

    cudaEventRecord(toStart, toStream);
    uploadKernel(toFilter, kernelSize, intensity, g_pipelineMem.kernelB, toStream);
    runConvolutionStage(g_pipelineMem.ping, g_pipelineMem.extra, g_pipelineMem.kernelB,
                        width, height, channels, kernelSize, toStream);
    cudaEventRecord(toDone, toStream);

    cudaEventSynchronize(fromDone);
    cudaEventElapsedTime(&perf.stageMs[0], fromStart, fromDone);
    cudaEventSynchronize(toDone);
    cudaEventElapsedTime(&perf.stageMs[1], toStart, toDone);

    cudaStreamWaitEvent(mixStream, fromDone, 0);
    cudaStreamWaitEvent(mixStream, toDone, 0);

    cudaEventRecord(start, mixStream);
    dim3 blockDim(16, 16);
    dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));
    wipeKernel<<<gridDim, blockDim, 0, mixStream>>>(g_pipelineMem.pong,
                                                    g_pipelineMem.extra,
                                                    g_pipelineMem.ping,
                                                    width, height, channels,
                                                    progress);
    cudaEventRecord(end, mixStream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&perf.stageMs[2], start, end);

    cudaEventRecord(start, mixStream);
    cudaMemcpyAsync(output.data, g_pipelineMem.ping, imageBytes,
                    cudaMemcpyDeviceToHost, mixStream);
    cudaEventRecord(end, mixStream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&perf.downloadMs, start, end);

    cudaEventRecord(totalEnd, mixStream);
    cudaEventSynchronize(totalEnd);
    cudaEventElapsedTime(&perf.totalMs, totalStart, totalEnd);

    cudaEventDestroy(totalStart);
    cudaEventDestroy(totalEnd);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaEventDestroy(uploadDone);
    cudaEventDestroy(fromStart);
    cudaEventDestroy(fromDone);
    cudaEventDestroy(toStart);
    cudaEventDestroy(toDone);

    if (multiStream)
    {
        cudaStreamDestroy(fromStream);
        cudaStreamDestroy(toStream);
        cudaStreamDestroy(mixStream);
    }
}

} // namespace cuda_filter
