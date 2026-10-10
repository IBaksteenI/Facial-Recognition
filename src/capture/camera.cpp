#include "camera.hpp"
#include <iostream>

bool Camera::tryPref(const cfg::CameraPref& pref) {
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, pref.width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, pref.height);
    cap_.set(cv::CAP_PROP_FPS, pref.fps);

    const int w = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    const int h = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double f = cap_.get(cv::CAP_PROP_FPS);

    if (w != pref.width || h != pref.height) {
        return false;
    }

    size_ = {w, h};
    fps_ = f > 1.0 ? static_cast<int>(f + 0.5) : pref.fps;
    return true;
}

bool Camera::open(int index) {
    cap_.open(index);
    if (!cap_.isOpened()) {
        std::cerr << "Cannot open camera " << index << "\n";
        return false;
    }

    bool exact = false;
    for (const auto& pref : cfg::kCameraPrefs) {
        if (tryPref(pref)) {
            std::cout << "Camera mode: " << size_.width << "x" << size_.height
                      << "@" << fps_ << "fps (requested)\n";
            exact = true;
            break;
        }
    }

    if (!exact) {
        size_ = {static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
                 static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT))};
        const double f = cap_.get(cv::CAP_PROP_FPS);
        fps_ = f > 1.0 ? static_cast<int>(f + 0.5) : 0;
        std::cout << "Camera mode: driver default " << size_.width << "x" << size_.height;
        if (fps_ > 0) std::cout << "@" << fps_ << "fps";
        std::cout << "\n";
    }

    // Grab one frame to confirm the real sensor size (drivers may ignore requests).
    cv::Mat probe;
    if (cap_.read(probe) && !probe.empty() && probe.size() != size_) {
        size_ = probe.size();
        std::cout << "Camera mode: actual frame size " << size_.width << "x" << size_.height << "\n";
    }
    return true;
}

bool Camera::read(cv::Mat& frame) {
    if (!cap_.read(frame) || frame.empty()) {
        return false;
    }
    if (frame.size() != size_) {
        size_ = frame.size();
    }
    return true;
}

void Camera::release() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}
