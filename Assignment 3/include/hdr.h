#pragma once

#include "filter_utils.h"

#include <opencv2/core.hpp>

void applyHDRToneMappingCPU(const cv::Mat& input, cv::Mat& output, const HDRParams& params);

bool applyHDRToneMappingGPU(const cv::Mat& input, cv::Mat& output, const HDRParams& params, float* kernelMs);
