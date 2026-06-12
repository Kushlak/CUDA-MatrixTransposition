#include "filter_utils.h"
#include "hdr.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

struct AppOptions {
    FilterType filter = FilterType::HDR_TONEMAPPING;
    HDRParams hdr;
    bool useCpu = false;
    bool compare = false;
    bool showHelp = false;
    int cameraIndex = 0;
};

bool readFloatArg(int argc, char** argv, int& i, float& value)
{
    if (i + 1 >= argc) {
        return false;
    }
    value = std::stof(argv[++i]);
    return true;
}

bool readIntArg(int argc, char** argv, int& i, int& value)
{
    if (i + 1 >= argc) {
        return false;
    }
    value = std::stoi(argv[++i]);
    return true;
}

bool parseArgs(int argc, char** argv, AppOptions& options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            options.showHelp = true;
            return true;
        }
        if (arg == "--filter" && i + 1 < argc) {
            options.filter = stringToFilterType(argv[++i]);
        } else if (arg == "--exposure") {
            if (!readFloatArg(argc, argv, i, options.hdr.exposure)) {
                return false;
            }
        } else if (arg == "--gamma") {
            if (!readFloatArg(argc, argv, i, options.hdr.gamma)) {
                return false;
            }
        } else if (arg == "--saturation") {
            if (!readFloatArg(argc, argv, i, options.hdr.saturation)) {
                return false;
            }
        } else if (arg == "--tone-operator" && i + 1 < argc) {
            options.hdr.toneOperator = stringToToneOperator(argv[++i]);
        } else if (arg == "--cpu") {
            options.useCpu = true;
        } else if (arg == "--compare") {
            options.compare = true;
        } else if (arg == "--camera") {
            if (!readIntArg(argc, argv, i, options.cameraIndex)) {
                return false;
            }
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << '\n';
            printUsage(argv[0]);
            return false;
        }
    }

    if (options.filter == FilterType::UNKNOWN) {
        std::cerr << "Unknown filter name.\n";
        return false;
    }

    return true;
}

void applyGrayscale(const cv::Mat& input, cv::Mat& output)
{
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, output, cv::COLOR_GRAY2BGR);
}

int maxPixelDiff(const cv::Mat& a, const cv::Mat& b)
{
    if (a.empty() || b.empty() || a.size() != b.size() || a.type() != b.type()) {
        return -1;
    }

    int maxDiff = 0;
    for (int y = 0; y < a.rows; ++y) {
        const cv::Vec3b* rowA = a.ptr<cv::Vec3b>(y);
        const cv::Vec3b* rowB = b.ptr<cv::Vec3b>(y);
        for (int x = 0; x < a.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                const int diff = std::abs(static_cast<int>(rowA[x][c]) - static_cast<int>(rowB[x][c]));
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }
    }
    return maxDiff;
}

} // namespace

int main(int argc, char** argv)
{
    AppOptions options;
    if (!parseArgs(argc, argv, options)) {
        return 1;
    }
    if (options.showHelp) {
        printUsage(argv[0]);
        return 0;
    }

    cv::VideoCapture camera(options.cameraIndex);
    if (!camera.isOpened()) {
        std::cerr << "Could not open webcam index " << options.cameraIndex << ".\n";
        return 1;
    }

    std::cout << "Filter: "
              << (options.filter == FilterType::HDR_TONEMAPPING ? "hdr_tonemapping" : "other")
              << ", mode: " << (options.compare ? "CPU/GPU compare" : (options.useCpu ? "CPU" : "GPU"))
              << ", tone operator: " << toneOperatorToString(options.hdr.toneOperator)
              << '\n';
    std::cout << "Press ESC to exit.\n";

    cv::Mat frame;
    cv::Mat output;
    cv::Mat cpuOutput;
    double totalFrameMs = 0.0;
    double totalCpuReferenceMs = 0.0;
    double totalGpuTotalMs = 0.0;
    double totalGpuKernelMs = 0.0;
    int maxObservedDiff = 0;
    int measuredFrames = 0;

    while (camera.read(frame)) {
        const auto frameStart = std::chrono::high_resolution_clock::now();
        float gpuKernelMs = 0.0f;

        if (options.filter == FilterType::HDR_TONEMAPPING) {
            if (options.compare) {
                const auto cpuStart = std::chrono::high_resolution_clock::now();
                applyHDRToneMappingCPU(frame, cpuOutput, options.hdr);
                const auto cpuStop = std::chrono::high_resolution_clock::now();
                totalCpuReferenceMs += std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();

                const auto gpuStart = std::chrono::high_resolution_clock::now();
                if (!applyHDRToneMappingGPU(frame, output, options.hdr, &gpuKernelMs)) {
                    std::cerr << "GPU HDR failed, showing CPU output.\n";
                    output = cpuOutput;
                }
                const auto gpuStop = std::chrono::high_resolution_clock::now();
                totalGpuTotalMs += std::chrono::duration<double, std::milli>(gpuStop - gpuStart).count();
                totalGpuKernelMs += gpuKernelMs;

                const int frameDiff = maxPixelDiff(cpuOutput, output);
                if (frameDiff > maxObservedDiff) {
                    maxObservedDiff = frameDiff;
                }
            } else if (options.useCpu) {
                applyHDRToneMappingCPU(frame, output, options.hdr);
            } else if (!applyHDRToneMappingGPU(frame, output, options.hdr, &gpuKernelMs)) {
                std::cerr << "GPU HDR failed, falling back to CPU.\n";
                applyHDRToneMappingCPU(frame, output, options.hdr);
            }
        } else if (options.filter == FilterType::GRAYSCALE) {
            applyGrayscale(frame, output);
        } else {
            output = frame;
        }

        const auto frameStop = std::chrono::high_resolution_clock::now();
        const double frameMs = std::chrono::duration<double, std::milli>(frameStop - frameStart).count();

        totalFrameMs += frameMs;
        if (!options.compare) {
            totalGpuKernelMs += gpuKernelMs;
        }
        ++measuredFrames;

        if (measuredFrames % 60 == 0) {
            const double avgFrameMs = totalFrameMs / measuredFrames;
            const double avgKernelMs = totalGpuKernelMs / measuredFrames;
            std::cout << "Average frame time: " << avgFrameMs << " ms";
            if (options.compare) {
                std::cout << ", CPU reference: " << (totalCpuReferenceMs / measuredFrames) << " ms"
                          << ", GPU total: " << (totalGpuTotalMs / measuredFrames) << " ms"
                          << ", GPU kernel: " << avgKernelMs << " ms"
                          << ", max pixel diff: " << maxObservedDiff;
            } else if (!options.useCpu) {
                std::cout << ", average GPU kernel: " << avgKernelMs << " ms";
            }
            std::cout << ", FPS: " << (1000.0 / avgFrameMs) << '\n';
        }

        cv::imshow("CUDA HDR Tone Mapping", output);
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    return 0;
}
