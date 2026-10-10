#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

#include "tracker.hpp"

struct HudStats {
    double fps = 0;
    double detectMs = 0;
    double embedMs = 0;
    double matchMs = 0;
    std::int32_t embeds = 0;
    std::int32_t people = 0;
    std::int32_t pendingCaptures = 0;
};

namespace overlay {

    // Draws face boxes/labels and the stats HUD onto the frame.
    void draw(cv::Mat& frame, const std::vector<Track*>& tracks, const HudStats& stats);

}  // namespace overlay
