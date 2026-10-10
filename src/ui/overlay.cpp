#include "overlay.hpp"

#include "config.hpp"

#include <opencv2/imgproc.hpp>

#include <cstdio>

namespace overlay {

    namespace {

        void shadowText(cv::Mat& img, const std::string& text, cv::Point org,
                        cv::Scalar color, double scale = 0.5) {

            cv::putText(img, text, org + cv::Point(1, 1), cv::FONT_HERSHEY_SIMPLEX,
                        scale, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);

            cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, 2,
                        cv::LINE_AA);
        }

    }  // namespace

    void draw(cv::Mat& frame, const std::vector<Track*>& tracks, const HudStats& stats) {
        for (const Track* t : tracks) {
            const cv::Scalar color = t->locked ? cv::Scalar(0, 200, 0)
                                               : cv::Scalar(0, 0, 220);

            cv::rectangle(frame, t->box, color, 2);

            char label[96];
            if (t->locked) {
                std::snprintf(label, sizeof(label), "#%d %s %.2f", t->id,
                              t->label.c_str(), t->lastCosine);
            }
            else {
                std::snprintf(label, sizeof(label), "#%d ? %.2f", t->id,
                              t->lastCosine);
            }
            const int y = std::max(14, t->box.y - 8);
            shadowText(frame, label, {t->box.x, y}, color);
        }

        char line[160];
        std::snprintf(line, sizeof(line), "FPS %.1f | detect %.1fms | embed %.1fms (%d) | match %.2fms",
                      stats.fps, stats.detectMs, stats.embedMs, stats.embeds,
                      stats.matchMs);

        shadowText(frame, line, {8, 20}, cv::Scalar(255, 255, 255), 0.55);

        std::snprintf(line, sizeof(line), "People: %d", stats.people);
        std::string hud2 = line;
        if (stats.pendingCaptures > 0) {
            char cap[64];
            std::snprintf(cap, sizeof(cap), " | enrolling %d/%d", stats.pendingCaptures,
                          cfg::kUnknownCaptureCount);

            hud2 += cap;
        }
        shadowText(frame, hud2, {8, 42}, cv::Scalar(255, 255, 255), 0.55);
    }

}  // namespace overlay
