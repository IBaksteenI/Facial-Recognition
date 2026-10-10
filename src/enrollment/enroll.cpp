#include "enroll.hpp"

#include "config.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace {

    std::string timestampName() {
        const auto now = std::chrono::system_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch())
                            .count() %
                        1000;
        const std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
        localtime_r(&t, &tm);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
        return std::string(buf) + "_" + std::to_string(ms);
    }

}  // namespace

bool Enroller::promote(Track& t, FaceDatabase& db) {
    const PersonId id = db.createPerson(t.captureEmbeds);
    if (id < 0) {
        return false;
    }

    const fs::path dir = fs::path(cfg::kKnownFacesDir) / ("person_" + std::to_string(id));
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        std::cerr << "Cannot create " << dir << ": " << ec.message() << "\n";
        return false;
    }

    int moved = 0;
    for (size_t i = 0; i < t.captureFiles.size(); ++i) {
        const fs::path src = t.captureFiles[i];
        fs::path target = dir / src.filename();
        if (fs::exists(target)) {
            target = dir / (src.stem().string() + "_" + std::to_string(i) +
                            src.extension().string());
        }
        fs::rename(src, target, ec);
        if (ec) {
            fs::copy_file(src, target, fs::copy_options::overwrite_existing, ec);
            fs::remove(src, ec);
        }
        if (fs::exists(target)) {
            db.attachImage(id, target.lexically_relative(".").generic_string(),
                           t.captureEmbeds[i]);
            ++moved;
        }
    }

    // Lock the track immediately so the label flips without waiting for hysteresis.
    const size_t sampleCount = t.captureEmbeds.size();
    t.locked = true;
    t.personId = id;
    t.label = "person_" + std::to_string(id);
    t.lockScore = cfg::kLockFrames;
    t.holdFrames = cfg::kHoldFrames;
    t.captureFiles.clear();
    t.captureEmbeds.clear();
    t.captureCount = 0;

    std::cout << "New person created: " << t.label << " (" << moved
              << " images, " << sampleCount << " samples)\n";
    return true;
}

bool Enroller::maybeCapture(Track& t, const cv::Mat& frame, const cv::Mat& emb,
                            double nowSeconds, FaceDatabase& db) {

    if (nowSeconds - t.lastCaptureTs < cfg::kCaptureCooldownS) {
        return false;
    }

    const int pad = static_cast<int>(
        std::max(t.box.width, t.box.height) * cfg::kCropPadFrac);
    cv::Rect r(t.box.x - pad, t.box.y - pad, t.box.width + 2 * pad,
               t.box.height + 2 * pad);
    r &= cv::Rect(0, 0, frame.cols, frame.rows);

    if (r.width <= 0 || r.height <= 0) {
        return false;
    }

    const cv::Mat crop = frame(r);
    if (crop.empty()) {
        return false;
    }

    std::error_code ec;
    fs::create_directories(cfg::kUnknownFacesDir, ec);
    const std::string name = "unknown_track" + std::to_string(t.id) + "_" +
                             timestampName() + ".jpg";
    const fs::path path = fs::path(cfg::kUnknownFacesDir) / name;
    if (!cv::imwrite(path.string(), crop,
                     {cv::IMWRITE_JPEG_QUALITY, 95})) {
        return false;
    }

    t.captureFiles.push_back(path.string());
    t.captureEmbeds.push_back(emb);
    t.captureCount++;
    t.lastCaptureTs = nowSeconds;

    std::cout << "Unknown #" << t.id << ": " << t.captureCount << "/"
              << cfg::kUnknownCaptureCount << "\n";

    if (t.captureCount >= cfg::kUnknownCaptureCount) {
        return promote(t, db);
    }
    return false;
}
