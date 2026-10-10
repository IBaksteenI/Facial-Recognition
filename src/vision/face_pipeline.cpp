#include "face_pipeline.hpp"

#include "config.hpp"

#include <chrono>

namespace {
    double elapsedMs(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0)
            .count();
    }
}  // namespace

bool FacePipeline::init() {
    detector_ = cv::FaceDetectorYN::create(
        cfg::kYuNetPath, "", cv::Size(640, 480),
        cfg::kScoreThreshold, 0.3f, 5000);

    if (!detector_) {
        return false;
    }

    recognizer_ = cv::FaceRecognizerSF::create(cfg::kSFacePath, "");
    if (!recognizer_) {
        return false;
    }
    return true;
}

bool FacePipeline::detect(const cv::Mat& frame) {
    const auto t0 = std::chrono::steady_clock::now();

    // Detect on a downscaled copy (YuNet cost scales with pixels), then map
    // boxes/landmarks back to source frame coordinates.
    double s = 1.0;
    const cv::Mat* input = &frame;
    if (frame.cols > cfg::kDetectWidth) {
        s = static_cast<double>(cfg::kDetectWidth) / frame.cols;
        cv::resize(frame, detectInput_, cv::Size(), s, s, cv::INTER_AREA);
        input = &detectInput_;
    }
    const double invS = 1.0 / s;

    detector_->setInputSize(input->size());
    detector_->detect(*input, rawFaces_);
    lastDetectMs_ = elapsedMs(t0);

    dets_.clear();
    if (rawFaces_.empty()) {
        return true;
    }

    const double minSize = cfg::kMinFaceSizeFrac * frame.rows;
    for (int r = 0; r < rawFaces_.rows; ++r) {

        float* row = rawFaces_.ptr<float>(r);
        if (invS != 1.0) {
            for (int c = 0; c < 14; ++c) {  // coords + landmarks, not the score
                row[c] *= static_cast<float>(invS);
            }
        }

        cv::Rect box{static_cast<int>(row[0]), static_cast<int>(row[1]),
                     static_cast<int>(row[2]), static_cast<int>(row[3])};

        if (box.height < minSize) {
            continue;
        }

        dets_.push_back({box, row[14], r});
    }
    return true;
}

cv::Mat FacePipeline::embed(const cv::Mat& frame, size_t detIndex) {
    if (detIndex >= dets_.size()) {
        return {};
    }

    const auto t0 = std::chrono::steady_clock::now();

    cv::Mat aligned;
    recognizer_->alignCrop(frame, rawFaces_.row(dets_[detIndex].row), aligned);
    cv::Mat feature;
    recognizer_->feature(aligned, feature);

    lastEmbedMs_ = elapsedMs(t0);
    return feature;
}
