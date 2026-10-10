#pragma once

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <vector>

struct FaceDetect {
    cv::Rect box;    // pixels in the source frame
    float score = 0;
    int row = -1;    // row in the raw detector output
};

// YuNet detection + SFace alignment/embedding.
class FacePipeline {
    public:
        [[nodiscard]] bool init();

        // Detect faces in the frame fills the internal detection list.
        // Returns false if the underlying detector failed.
        [[nodiscard]] bool detect(const cv::Mat& frame);
        const std::vector<FaceDetect>& detections() const { return dets_; }

        // Align + embed detection i of the last detect() call. Returns 1x128 CV_32F, empty on failure.
        cv::Mat embed(const cv::Mat& frame, size_t detIndex);

        double lastDetectMs() const { return lastDetectMs_; }
        double lastEmbedMs() const { return lastEmbedMs_; }

    private:
        cv::Ptr<cv::FaceDetectorYN> detector_;
        cv::Ptr<cv::FaceRecognizerSF> recognizer_;
        cv::Mat rawFaces_;                 // Nx15 detector output (source frame coords)
        cv::Mat detectInput_;              // downscaled frame used for detection
        std::vector<FaceDetect> dets_;
        double lastDetectMs_ = 0;
        double lastEmbedMs_ = 0;
};
