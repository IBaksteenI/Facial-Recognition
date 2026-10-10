#include "tracker.hpp"

#include "config.hpp"

#include <filesystem>
#include <limits>

namespace {

    cv::Point2f centerOf(const cv::Rect& box, cv::Size frame) {
        return {static_cast<float>((box.x + box.width * 0.5) / std::max(1, frame.width)),
                static_cast<float>((box.y + box.height * 0.5) / std::max(1, frame.height))};
    }

}  // namespace

void Tracker::cleanupCaptures(Track& t) {
    for (const auto& f : t.captureFiles) {
        std::error_code ec;
        std::filesystem::remove(f, ec);
    }

    t.captureFiles.clear();
    t.captureEmbeds.clear();
    t.captureCount = 0;
}

void Tracker::update(const std::vector<FaceDetect>& dets, cv::Size frameSize) {
    const float maxDist = static_cast<float>(cfg::kMaxCenterDistFrac);

    // Age every track prune stale ones.
    std::vector<TrackId> stale;
    for (auto& [id, t] : tracks_) {
        t.misses++;
        t.framesSinceEmbed++;

        if (t.holdFrames > 0) {
            t.holdFrames--;
        }

        if (t.misses > cfg::kForgetMisses) {
            stale.push_back(id);
        }
    }

    for (const TrackId id : stale) {
        cleanupCaptures(tracks_.at(id));
        tracks_.erase(id);
    }

    // Greedy nearest center association.
    detToTrack_.assign(dets.size(), -1);
    std::vector<bool> trackTaken(tracks_.size(), false);
    std::vector<TrackId> trackIds;
    trackIds.reserve(tracks_.size());

    for (const auto& [id, t] : tracks_) {
        trackIds.push_back(id);
    }

    for (size_t i = 0; i < dets.size(); ++i) {
        const cv::Point2f c = centerOf(dets[i].box, frameSize);
        int bestTrack = -1;
        float bestDist = maxDist;

        for (size_t k = 0; k < trackIds.size(); ++k) {
            if (trackTaken[k]) {
                continue;
            }

            const cv::Point2f tc = tracks_.at(trackIds[k]).center;
            const float d = static_cast<float>(cv::norm(c - tc));
            if (d < bestDist) {
                bestDist = d;
                bestTrack = static_cast<int>(k);
            }
        }

        if (bestTrack < 0) {
            continue;
        }

        trackTaken[bestTrack] = true;
        Track& t = tracks_.at(trackIds[bestTrack]);
        detToTrack_[i] = trackIds[bestTrack];
        t.box = dets[i].box;
        t.center = c;
        t.misses = 0;
        t.age++;
    }

    // New tracks for unassociated detections.
    for (size_t i = 0; i < dets.size(); ++i) {
        if (detToTrack_[i] >= 0) {
            continue;
        }

        Track t;
        t.id = nextId_++;
        t.box = dets[i].box;
        t.center = centerOf(dets[i].box, frameSize);
        t.misses = 0;
        t.age = 1;

        t.framesSinceEmbed = std::numeric_limits<std::int32_t>::max();  // force embed on first sight
        const TrackId id = t.id;
        tracks_.emplace(id, std::move(t));
        detToTrack_[i] = id;
    }
}

void Tracker::applyMatch(Track& t, const cv::Mat& emb, const std::optional<MatchResult>& m) {
    if (!emb.empty()) {
        t.lastEmb = emb;
    }
    t.lastCosine = m ? m->cosine : 0.0;

    const bool has = m.has_value();
    const bool strict = has && m->cosine >= cfg::kCosineStrict;
    const bool loose = has && m->cosine >= cfg::kCosineLoose;
    const bool samePerson = has && t.locked && t.personId == m->personId;

    auto decay = [&] {
        if (t.lockScore > 0) {
            t.lockScore--;
        }
        if (t.lockScore == 0 && t.holdFrames == 0 && t.locked) {
            t.locked = false;
            t.personId = -1;
            t.label = "Unknown";
        }
    };

    if (strict) {
        if (samePerson) {
            t.holdFrames = cfg::kHoldFrames;
            return;
        }
        if (t.locked && t.personId != m->personId) {
            // Contradicting evidence: unlock so the new identity can lock in immediately.
            t.locked = false;
            t.personId = -1;
            t.label = "Unknown";
            t.lockScore = cfg::kLockFrames - 1;
            return;
        }
        t.lockScore = std::min(cfg::kLockFrames, t.lockScore + 1);
        if (t.lockScore >= cfg::kLockFrames) {
            t.locked = true;
            t.personId = m->personId;
            t.label = m->name;
            t.holdFrames = cfg::kHoldFrames;
        }
        return;
    }

    if (loose) {
        if (samePerson) {
            t.holdFrames = cfg::kHoldFrames;
            return;
        }
        if (!t.locked) {
            return;  // ambiguous observation before locking: neither up nor down
        }
    }
    decay();
}

std::vector<Track*> Tracker::activeTracks() {
    std::vector<Track*> out;
    for (auto& [id, t] : tracks_) {
        if (t.misses == 0) {
            out.push_back(&t);
        }
    }
    return out;
}
