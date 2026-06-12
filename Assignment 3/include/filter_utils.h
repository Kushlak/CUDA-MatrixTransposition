#pragma once

#include <string>

enum class FilterType {
    NONE,
    GRAYSCALE,
    HDR_TONEMAPPING,
    UNKNOWN
};

enum class ToneOperator {
    REINHARD,
    EXPOSURE,
    ACES,
    LOCAL
};

struct HDRParams {
    float exposure = 1.0f;
    float gamma = 2.2f;
    float saturation = 1.0f;
    ToneOperator toneOperator = ToneOperator::REINHARD;
};

FilterType stringToFilterType(const std::string& value);
ToneOperator stringToToneOperator(const std::string& value);
std::string toneOperatorToString(ToneOperator op);
void printUsage(const char* programName);
