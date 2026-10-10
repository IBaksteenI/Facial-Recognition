#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "face_database.hpp"
#include "face_pipeline.hpp"
#include "types.hpp"

struct Track {
    TrackId id = -1;
    cv::Rect box;               // last seen, pixels
    cv::Point2f center;         // frame fraction coordinates
    std::int32_t misses = 0;
    std::int32_t age = 0;
    std::int32_t lockScore = 0;
    std::int32_t holdFrames = 0;
    bool locked = false;
    PersonId personId = -1;     // valid when locked
    std::string label = "Unknown";
    double lastCosine = 0.0;
    cv::Mat lastEmb;
    std::int32_t framesSinceEmbed = 0;

    // Enrollment state (pending unknown captures).
    std::vector<std::string> captureFiles;   // paths under unknown_faces/
    std::vector<cv::Mat> captureEmbeds;
    double lastCaptureTs = -1e18;
    std::int32_t captureCount = 0;
};

// Center distance track association + lock/hold/forget hysteresis.
class Tracker {
    public:
        // Ages all tracks, associates detections (creating new tracks), prunes stale ones,
        // and fills the det->track mapping (parallel to the detections vector).
        void update(const std::vector<FaceDetect>& dets, cv::Size frameSize);

        Track* trackForDet(size_t i) {
            return i < detToTrack_.size() && detToTrack_[i] >= 0
                       ? &tracks_.at(detToTrack_[i])
                       : nullptr;
        }

        // Applies one match observation (hysteresis update). emb may be empty.
        void applyMatch(Track& t, const cv::Mat& emb, const std::optional<MatchResult>& m);

        // Tracks seen this frame (misses == 0).
        std::vector<Track*> activeTracks();

        // Removes a track's pending capture files (call when the track becomes known).
        void discardCaptures(Track& t) { cleanupCaptures(t); }

        std::unordered_map<TrackId, Track>& tracks() { return tracks_; }

    private:
        void cleanupCaptures(Track& t);

        std::unordered_map<TrackId, Track> tracks_;
        std::vector<TrackId> detToTrack_;
        TrackId nextId_ = 1;
};
