#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>
#include "input_args_parser/input_args_parser.h"
#include "utils/input_handler.h"
#include "kernels/kernels.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace
{
std::vector<std::string> splitPipeline(const std::string &text, const std::string &fallback)
{
    std::vector<std::string> filters;
    std::stringstream ss(text.empty() ? fallback : text);
    std::string item;

    while (std::getline(ss, item, ','))
    {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }), item.end());
        if (!item.empty())
            filters.push_back(item);
    }

    if (filters.empty())
        filters.push_back("blur");
    return filters;
}

std::string joinPipeline(const std::vector<std::string> &filters)
{
    std::string result;
    for (size_t i = 0; i < filters.size(); ++i)
    {
        if (i > 0)
            result += " -> ";
        result += filters[i];
    }
    return result;
}

cv::Mat makeTimingView(const cuda_filter::PipelinePerf &perf, int width)
{
    cv::Mat chart(120, width, CV_8UC3, cv::Scalar(35, 35, 35));
    std::vector<float> values;
    std::vector<std::string> names;

    values.push_back(perf.uploadMs);
    names.push_back("upload");
    for (size_t i = 0; i < perf.stageMs.size(); ++i)
    {
        values.push_back(perf.stageMs[i]);
        names.push_back("stage" + std::to_string(i + 1));
    }
    values.push_back(perf.downloadMs);
    names.push_back("download");

    float maxValue = std::max(1.0f, perf.totalMs);
    int barWidth = std::max(20, width / (int)values.size());

    for (size_t i = 0; i < values.size(); ++i)
    {
        int h = (int)(80.0f * values[i] / maxValue);
        int x = (int)i * barWidth + 6;
        cv::Scalar color = (i == 0 || i + 1 == values.size())
                               ? cv::Scalar(80, 160, 230)
                               : cv::Scalar(80, 210, 120);
        cv::rectangle(chart, cv::Rect(x, 95 - h, barWidth - 12, h), color, cv::FILLED);
        cv::putText(chart, names[i], cv::Point(x, 112), cv::FONT_HERSHEY_SIMPLEX,
                    0.35, cv::Scalar(230, 230, 230), 1);
    }

    cv::putText(chart, "total: " + std::to_string(perf.totalMs).substr(0, 5) + " ms",
                cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(255, 255, 0), 1);
    return chart;
}

cv::Mat makeBenchmarkFrame(int width, int height, int shift)
{
    cv::Mat frame(height, width, CV_8UC3);
    for (int y = 0; y < frame.rows; ++y)
    {
        for (int x = 0; x < frame.cols; ++x)
        {
            unsigned char b = (unsigned char)((x + shift) % 256);
            unsigned char g = (unsigned char)((y + shift * 2) % 256);
            unsigned char r = (unsigned char)(((x + y) / 2 + shift * 3) % 256);
            frame.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
        }
    }
    return frame;
}

struct BenchmarkResult
{
    std::string name;
    float uploadMs = 0.0f;
    float computeMs = 0.0f;
    float downloadMs = 0.0f;
    float totalMs = 0.0f;
};

float sumStages(const std::vector<float> &values)
{
    float total = 0.0f;
    for (float value : values)
        total += value;
    return total;
}

BenchmarkResult runPipelineBench(const std::string &name,
                                 const std::vector<std::string> &pipeline,
                                 bool multiStream,
                                 int frames,
                                 int kernelSize,
                                 float intensity)
{
    BenchmarkResult result;
    result.name = name;

    cv::Mat output;
    cuda_filter::PipelinePerf perf;

    // A few warm-up frames remove first-use CUDA setup from the average.
    for (int i = 0; i < 10; ++i)
    {
        cv::Mat frame = makeBenchmarkFrame(640, 480, i);
        cuda_filter::applyPipelineGPU(frame, output, pipeline, kernelSize,
                                      intensity, multiStream, perf);
    }

    for (int i = 0; i < frames; ++i)
    {
        cv::Mat frame = makeBenchmarkFrame(640, 480, i);
        cuda_filter::applyPipelineGPU(frame, output, pipeline, kernelSize,
                                      intensity, multiStream, perf);
        result.uploadMs += perf.uploadMs;
        result.computeMs += sumStages(perf.stageMs);
        result.downloadMs += perf.downloadMs;
        result.totalMs += perf.totalMs;
    }

    result.uploadMs /= frames;
    result.computeMs /= frames;
    result.downloadMs /= frames;
    result.totalMs /= frames;
    return result;
}

BenchmarkResult runTransitionBench(const std::string &name,
                                   bool multiStream,
                                   int frames,
                                   int kernelSize,
                                   float intensity)
{
    BenchmarkResult result;
    result.name = name;

    cv::Mat output;
    cuda_filter::PipelinePerf perf;

    for (int i = 0; i < 10; ++i)
    {
        cv::Mat frame = makeBenchmarkFrame(640, 480, i);
        cuda_filter::applyWipeTransitionGPU(frame, output, "blur", "emboss",
                                            kernelSize, intensity, 0.5f,
                                            multiStream, perf);
    }

    for (int i = 0; i < frames; ++i)
    {
        cv::Mat frame = makeBenchmarkFrame(640, 480, i);
        cuda_filter::applyWipeTransitionGPU(frame, output, "blur", "emboss",
                                            kernelSize, intensity, 0.5f,
                                            multiStream, perf);
        result.uploadMs += perf.uploadMs;
        result.computeMs += sumStages(perf.stageMs);
        result.downloadMs += perf.downloadMs;
        result.totalMs += perf.totalMs;
    }

    result.uploadMs /= frames;
    result.computeMs /= frames;
    result.downloadMs /= frames;
    result.totalMs /= frames;
    return result;
}

void writeBenchmarkChart(const std::vector<BenchmarkResult> &results,
                         const std::string &path)
{
    cv::Mat chart(320, 760, CV_8UC3, cv::Scalar(245, 245, 245));
    float maxValue = 1.0f;
    for (const auto &result : results)
        maxValue = std::max(maxValue, result.totalMs);

    int x = 45;
    const int baseY = 250;
    const int barW = 80;
    for (const auto &result : results)
    {
        int barH = (int)(190.0f * result.totalMs / maxValue);
        cv::rectangle(chart, cv::Rect(x, baseY - barH, barW, barH),
                      cv::Scalar(70, 150, 220), cv::FILLED);
        cv::putText(chart, std::to_string(result.totalMs).substr(0, 5) + " ms",
                    cv::Point(x - 5, baseY - barH - 8), cv::FONT_HERSHEY_SIMPLEX,
                    0.45, cv::Scalar(30, 30, 30), 1);
        cv::putText(chart, result.name, cv::Point(x - 10, baseY + 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(30, 30, 30), 1);
        x += 140;
    }

    cv::putText(chart, "Pipeline benchmark, average frame time",
                cv::Point(35, 35), cv::FONT_HERSHEY_SIMPLEX, 0.75,
                cv::Scalar(20, 20, 20), 2);
    cv::putText(chart, "640x480 synthetic input",
                cv::Point(35, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(80, 80, 80), 1);
    cv::imwrite(path, chart);
}

int runBenchmarkMode(const cuda_filter::FilterOptions &options)
{
    const int frames = std::max(1, options.benchmarkFrames);
    std::vector<BenchmarkResult> results;

    results.push_back(runPipelineBench("blur single", {"blur"}, false, frames,
                                       options.kernelSize, options.intensity));
    results.push_back(runPipelineBench("3 filters single", {"blur", "sharpen", "edge"},
                                       false, frames, options.kernelSize,
                                       options.intensity));
    results.push_back(runPipelineBench("3 filters multi", {"blur", "sharpen", "edge"},
                                       true, frames, options.kernelSize,
                                       options.intensity));
    results.push_back(runTransitionBench("wipe single", false, frames,
                                         options.kernelSize, options.intensity));
    results.push_back(runTransitionBench("wipe multi", true, frames,
                                         options.kernelSize, options.intensity));

    std::ofstream csv(options.benchmarkCsv);
    csv << "case,upload_ms,compute_ms,download_ms,total_ms\n";
    csv << std::fixed << std::setprecision(4);
    for (const auto &result : results)
    {
        csv << result.name << ','
            << result.uploadMs << ','
            << result.computeMs << ','
            << result.downloadMs << ','
            << result.totalMs << '\n';
    }

    writeBenchmarkChart(results, options.benchmarkChart);

    PLOG_INFO << "Benchmark CSV: " << options.benchmarkCsv;
    PLOG_INFO << "Benchmark chart: " << options.benchmarkChart;
    return 0;
}
}

int main(int argc, char **argv)
{
    plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::info, &consoleAppender);

    cuda_filter::InputArgsParser parser(argc, argv);
    cuda_filter::FilterOptions options = parser.parseArgs();

    if (options.benchmark)
        return runBenchmarkMode(options);

    cuda_filter::InputHandler inputHandler(options);
    if (!inputHandler.isOpened())
    {
        PLOG_ERROR << "Failed to initialize input source";
        return -1;
    }

    std::vector<std::string> pipeline = splitPipeline(options.pipeline, options.filterType);
    bool multiStream = options.multiStream;
    bool transitionActive = options.transition;
    double transitionStart = (double)cv::getTickCount();

    PLOG_INFO << "Pipeline: " << joinPipeline(pipeline);
    PLOG_INFO << "Keys: 1 blur, 2 sharpen, 3 edge, 4 emboss, r remove, m streams, t transition, ESC exit";

    cv::Mat frame, filtered, shown;
    cuda_filter::PipelinePerf perf;

    while (true)
    {
        if (!inputHandler.readFrame(frame))
        {
            PLOG_ERROR << "Failed to read frame";
            break;
        }

        if (transitionActive)
        {
            double now = (double)cv::getTickCount();
            float seconds = (float)((now - transitionStart) / cv::getTickFrequency());
            float progress = seconds / std::max(0.1f, options.transitionTime);

            cuda_filter::applyWipeTransitionGPU(frame, filtered,
                                                options.transitionFrom,
                                                options.transitionTo,
                                                options.kernelSize,
                                                options.intensity,
                                                progress,
                                                multiStream,
                                                perf);

            if (progress >= 1.0f)
            {
                transitionActive = false;
                pipeline.clear();
                pipeline.push_back(options.transitionTo);
            }
        }
        else
        {
            cuda_filter::applyPipelineGPU(frame, filtered, pipeline,
                                          options.kernelSize,
                                          options.intensity,
                                          multiStream,
                                          perf);
        }

        std::string mode = multiStream ? "multi-stream" : "single-stream";
        cv::putText(filtered, "Pipeline: " + joinPipeline(pipeline),
                    cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    cv::Scalar(255, 255, 0), 2);
        cv::putText(filtered, mode,
                    cv::Point(10, 56), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    cv::Scalar(255, 255, 0), 2);

        cv::Mat chart = makeTimingView(perf, filtered.cols);
        cv::vconcat(filtered, chart, shown);

        if (options.preview)
        {
            cv::Mat left;
            cv::vconcat(frame, cv::Mat::zeros(chart.size(), chart.type()), left);
            cv::hconcat(left, shown, shown);
        }

        inputHandler.displayFrame(shown);

        int key = cv::waitKey(1);
        if (key == 27)
            break;
        if (key == '1')
            pipeline.push_back("blur");
        if (key == '2')
            pipeline.push_back("sharpen");
        if (key == '3')
            pipeline.push_back("edge");
        if (key == '4')
            pipeline.push_back("emboss");
        if (key == 'r' && pipeline.size() > 1)
            pipeline.pop_back();
        if (key == 'm')
            multiStream = !multiStream;
        if (key == 't')
        {
            transitionActive = true;
            transitionStart = (double)cv::getTickCount();
        }
    }

    PLOG_INFO << "Application terminated";
    return 0;
}
