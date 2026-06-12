#include "filter_utils.h"

#include <algorithm>
#include <cctype>
#include <iostream>

namespace {

std::string normalize(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

} // namespace

FilterType stringToFilterType(const std::string& value)
{
    const std::string key = normalize(value);

    if (key == "none") {
        return FilterType::NONE;
    }
    if (key == "gray" || key == "grayscale") {
        return FilterType::GRAYSCALE;
    }
    if (key == "hdr" || key == "hdr_tonemapping" || key == "tone_mapping") {
        return FilterType::HDR_TONEMAPPING;
    }

    return FilterType::UNKNOWN;
}

ToneOperator stringToToneOperator(const std::string& value)
{
    const std::string key = normalize(value);

    if (key == "reinhard") {
        return ToneOperator::REINHARD;
    }
    if (key == "exposure") {
        return ToneOperator::EXPOSURE;
    }
    if (key == "aces") {
        return ToneOperator::ACES;
    }
    if (key == "local") {
        return ToneOperator::LOCAL;
    }

    return ToneOperator::REINHARD;
}

std::string toneOperatorToString(ToneOperator op)
{
    switch (op) {
    case ToneOperator::REINHARD:
        return "reinhard";
    case ToneOperator::EXPOSURE:
        return "exposure";
    case ToneOperator::ACES:
        return "aces";
    case ToneOperator::LOCAL:
        return "local";
    }
    return "reinhard";
}

void printUsage(const char* programName)
{
    std::cout
        << "Usage:\n"
        << "  " << programName << " --filter hdr_tonemapping [options]\n\n"
        << "Options:\n"
        << "  --filter <name>             none | grayscale | hdr_tonemapping\n"
        << "  --exposure <float>          Exposure strength, default 1.0\n"
        << "  --gamma <float>             Gamma correction, default 2.2\n"
        << "  --saturation <float>        Color saturation, default 1.0\n"
        << "  --tone-operator <string>    reinhard | exposure | aces | local\n"
        << "  --cpu                       Run HDR tone mapping on CPU\n"
        << "  --compare                   Run CPU and GPU HDR each frame and print both timings\n"
        << "  --camera <index>            Webcam index, default 0\n"
        << "  --help                      Show this help\n\n"
        << "Example:\n"
        << "  " << programName << " --filter hdr_tonemapping --exposure 1.2 --gamma 2.2 --saturation 1.1 --tone-operator reinhard\n";
}
