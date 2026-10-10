#pragma once

#include <opencv2/videoio.hpp>

#include <string>
#include <vector>

#include "config.hpp"

class Camera {
    public:
        // Opens index 0, probing cfg::kCameraPrefs in order.
        // Prints the chosen mode (or the driver fallback) to stdout.
        [[nodiscard]] bool open(int index = 0);
        [[nodiscard]] bool read(cv::Mat& frame);
        void release();

        cv::Size size() const { return size_; }
        int fps() const { return fps_; }

    private:
        bool tryPref(const cfg::CameraPref& pref);

        cv::VideoCapture cap_;
        cv::Size size_;
        int fps_ = 0;
};
