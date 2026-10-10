#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "types.hpp"

struct Person {
    std::string name;                   // "person_1", "person_2", ...
    std::vector<cv::Mat> samples;       // L2-normalized 1x128 CV_32F embeddings
    cv::Mat centroid;                   // L2-normalized 1x128 — the match target
    std::vector<std::string> images;    // relative paths of source jpgs
};

struct MatchResult {
    PersonId personId = -1;
    std::string name;
    double cosine = 0.0;
};
