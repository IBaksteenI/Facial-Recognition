#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cfg {

    struct CameraPref {
        std::int32_t width;
        std::int32_t height;
        std::int32_t fps;
    };

    // Tried in order, first fully supported mode wins. Falls back to driver default.
    inline const std::vector<CameraPref> kCameraPrefs = {
        {1280, 720, 30},
        {1920, 1080, 30},
        {640, 480, 30},
    };

    // Spatial thresholds as fractions of frame height (resolution independent).
    inline constexpr double kMinFaceSizeFrac = 0.06;
    inline constexpr double kMaxCenterDistFrac = 0.12;

    // SFace matching (OpenCV documented operating points).
    inline constexpr double kCosineLoose = 0.363;   // accept threshold
    inline constexpr double kCosineStrict = 0.45;   // lock threshold

    // Track hysteresis (frame counts).
    inline constexpr std::int32_t kLockFrames = 5;
    inline constexpr std::int32_t kHoldFrames = 12;
    inline constexpr std::int32_t kForgetMisses = 20;

    // Embedding schedule (detection runs every frame regardless).
    inline constexpr std::int32_t kEmbedEveryUnknown = 3;
    inline constexpr std::int32_t kVerifyEvery = 8;

    // Enrollment.
    inline constexpr std::int32_t kUnknownCaptureCount = 10;
    inline constexpr double kCaptureCooldownS = 2.0;
    inline constexpr double kCropPadFrac = 0.25;    // pad relative to face size

    // Detector.
    inline constexpr float kScoreThreshold = 0.6f;
    // Frames wider than this are downscaled for detection only (boxes scaled back up).
    // YuNet cost scales with pixels 640px is plenty for faces >= kMinFaceSizeFrac.
    inline constexpr std::int32_t kDetectWidth = 640;

    // Paths (relative to working directory).
    inline constexpr const char* kYuNetPath = "models/yunet.onnx";
    inline constexpr const char* kSFacePath = "models/sface.onnx";
    inline constexpr const char* kKnownFacesDir = "known_faces";
    inline constexpr const char* kUnknownFacesDir = "unknown_faces";
    inline constexpr const char* kEmbedCachePath = "data/embed_cache.yml";

    // Stats print interval (seconds).
    inline constexpr double kStatsIntervalS = 2.0;

    // Print every match observation (cosine + lock state) for threshold tuning.
    inline constexpr bool kVerboseMatches = false;

}  // namespace cfg
