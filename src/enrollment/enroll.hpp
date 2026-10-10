#pragma once

#include "face_database.hpp"
#include "tracker.hpp"

// Captures samples from an unknown track and promotes it to a new person.
class Enroller {
    public:
        // Capture a padded face crop + embedding (cooldown limited).
        // Returns true when the capture buffer is full and promotion ran.
        bool maybeCapture(Track& t, const cv::Mat& frame, const cv::Mat& emb,
                          double nowSeconds, FaceDatabase& db);

    private:
        bool promote(Track& t, FaceDatabase& db);
};
