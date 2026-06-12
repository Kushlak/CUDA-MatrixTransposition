#include "hdr.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr int kLocalRadius = 1;
constexpr int kBlockSize = 16;

float clamp01(float value)
{
    return std::max(0.0f, std::min(1.0f, value));
}

float toneMap(float luminance, const HDRParams& params)
{
    const float x = std::max(0.0f, luminance * params.exposure);

    switch (params.toneOperator) {
    case ToneOperator::REINHARD:
        return x / (1.0f + x);
    case ToneOperator::EXPOSURE:
        return 1.0f - std::exp(-x);
    case ToneOperator::ACES:
        return clamp01((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f));
    case ToneOperator::LOCAL:
        return x / (1.0f + x);
    }

    return x / (1.0f + x);
}

} // namespace

void applyHDRToneMappingLocalCPU(const cv::Mat& input, cv::Mat& output, const HDRParams& params);

void applyHDRToneMappingCPU(const cv::Mat& input, cv::Mat& output, const HDRParams& params)
{
    if (params.toneOperator == ToneOperator::LOCAL) {
        applyHDRToneMappingLocalCPU(input, output, params);
        return;
    }

    output.create(input.size(), input.type());

    const float invGamma = 1.0f / std::max(params.gamma, 0.001f);
    const float epsilon = 1e-4f;

    for (int y = 0; y < input.rows; ++y) {
        const cv::Vec3b* srcRow = input.ptr<cv::Vec3b>(y);
        cv::Vec3b* dstRow = output.ptr<cv::Vec3b>(y);

        for (int x = 0; x < input.cols; ++x) {
            const float b = srcRow[x][0] / 255.0f;
            const float g = srcRow[x][1] / 255.0f;
            const float r = srcRow[x][2] / 255.0f;

            const float luminance = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            const float mapped = std::pow(clamp01(toneMap(luminance, params)), invGamma);
            const float scale = mapped / std::max(luminance, epsilon);

            float outR = r * scale;
            float outG = g * scale;
            float outB = b * scale;

            const float gray = mapped;
            outR = gray + params.saturation * (outR - gray);
            outG = gray + params.saturation * (outG - gray);
            outB = gray + params.saturation * (outB - gray);

            dstRow[x][0] = static_cast<unsigned char>(clamp01(outB) * 255.0f);
            dstRow[x][1] = static_cast<unsigned char>(clamp01(outG) * 255.0f);
            dstRow[x][2] = static_cast<unsigned char>(clamp01(outR) * 255.0f);
        }
    }
}

void applyHDRToneMappingLocalCPU(const cv::Mat& input, cv::Mat& output, const HDRParams& params)
{
    output.create(input.size(), input.type());

    const float invGamma = 1.0f / std::max(params.gamma, 0.001f);
    const float epsilon = 1e-4f;

    for (int y = 0; y < input.rows; ++y) {
        cv::Vec3b* dstRow = output.ptr<cv::Vec3b>(y);

        for (int x = 0; x < input.cols; ++x) {
            const cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);
            const float b = pixel[0] / 255.0f;
            const float g = pixel[1] / 255.0f;
            const float r = pixel[2] / 255.0f;
            const float luminance = 0.2126f * r + 0.7152f * g + 0.0722f * b;

            float localSum = 0.0f;
            int localCount = 0;
            const int blockX = (x / kBlockSize) * kBlockSize;
            const int blockY = (y / kBlockSize) * kBlockSize;
            const int threadX = x - blockX;
            const int threadY = y - blockY;

            for (int oy = -kLocalRadius; oy <= kLocalRadius; ++oy) {
                for (int ox = -kLocalRadius; ox <= kLocalRadius; ++ox) {
                    const int neighborThreadX = threadX + ox;
                    const int neighborThreadY = threadY + oy;
                    if (neighborThreadX >= 0 && neighborThreadX < kBlockSize &&
                        neighborThreadY >= 0 && neighborThreadY < kBlockSize) {
                        const int nx = blockX + neighborThreadX;
                        const int ny = blockY + neighborThreadY;
                        if (ny >= 0 && ny < input.rows && nx >= 0 && nx < input.cols) {
                            const cv::Vec3b neighbor = input.at<cv::Vec3b>(ny, nx);
                            const float nb = neighbor[0] / 255.0f;
                            const float ng = neighbor[1] / 255.0f;
                            const float nr = neighbor[2] / 255.0f;
                            localSum += 0.2126f * nr + 0.7152f * ng + 0.0722f * nb;
                        }
                        ++localCount;
                    }
                }
            }

            const float localAverage = localSum / static_cast<float>(std::max(localCount, 1));
            const float adapted = luminance / (1.0f + localAverage);
            const float mapped = std::pow(clamp01(toneMap(adapted, params)), invGamma);
            const float scale = mapped / std::max(luminance, epsilon);

            float outR = r * scale;
            float outG = g * scale;
            float outB = b * scale;

            const float gray = mapped;
            outR = gray + params.saturation * (outR - gray);
            outG = gray + params.saturation * (outG - gray);
            outB = gray + params.saturation * (outB - gray);

            dstRow[x][0] = static_cast<unsigned char>(clamp01(outB) * 255.0f);
            dstRow[x][1] = static_cast<unsigned char>(clamp01(outG) * 255.0f);
            dstRow[x][2] = static_cast<unsigned char>(clamp01(outR) * 255.0f);
        }
    }
}
