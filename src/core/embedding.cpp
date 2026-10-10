#include "embedding.hpp"

cv::Mat embed::normalized(const cv::Mat& emb) {
    if (emb.empty()) {
        return {};
    }

    cv::Mat f = emb.reshape(1, 1);  // 1 x N
    cv::Mat out;
    f.convertTo(out, CV_32F);

    const double n = cv::norm(out);
    if (n > 1e-12) {
        out /= n;
    }
    return out;
}

cv::Mat embed::centroid(const std::vector<cv::Mat>& samples) {
    if (samples.empty()) {
        return {};
    }

    cv::Mat sum = cv::Mat::zeros(1, samples[0].cols, CV_32F);
    for (const auto& s : samples) {
        sum += s;
    }
    return normalized(sum);
}
