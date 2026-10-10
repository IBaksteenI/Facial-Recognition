#pragma once

#include <opencv2/core.hpp>

#include <vector>

// Stateless helpers for L2-normalized 1x128 CV_32F embeddings.
namespace embed {

    // Flattens to 1xN CV_32F and L2-normalizes, empty input yields an empty Mat.
    cv::Mat normalized(const cv::Mat& emb);

    // Mean of the samples, L2-normalized, empty samples yield an empty Mat.
    cv::Mat centroid(const std::vector<cv::Mat>& samples);

}  // namespace embed
