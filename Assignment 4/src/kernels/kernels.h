#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace cuda_filter
{
    struct PipelinePerf
    {
        float uploadMs = 0.0f;
        float downloadMs = 0.0f;
        float totalMs = 0.0f;
        std::vector<float> stageMs;
    };

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);

    struct HdrOptions
    {
        float exposure;
        float gamma;
        float saturation;
        float whitePoint;
        std::string algorithm;  // "reinhard", "aces", "local"
    };

    void applyHDRFilterGPU(const cv::Mat &input, cv::Mat &output, const HdrOptions &opts);
    void applyHDRFilterCPU(const cv::Mat &input, cv::Mat &output, const HdrOptions &opts);

    void applyPipelineGPU(const cv::Mat &input, cv::Mat &output,
                          const std::vector<std::string> &filters,
                          int kernelSize, float intensity,
                          bool multiStream, PipelinePerf &perf);

    void applyWipeTransitionGPU(const cv::Mat &input, cv::Mat &output,
                                const std::string &fromFilter,
                                const std::string &toFilter,
                                int kernelSize, float intensity,
                                float progress, bool multiStream,
                                PipelinePerf &perf);

    namespace cuda
    {
// CUDA-specific type declarations and helper functions
#ifdef __CUDACC__
        // These will only be visible to CUDA compiler
        __host__ __device__ inline int divUp(int a, int b)
        {
            return (a + b - 1) / b;
        }
#endif
    }

} // namespace cuda_filter
