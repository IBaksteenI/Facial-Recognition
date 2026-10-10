#include "face_database.hpp"

#include "config.hpp"
#include "embedding.hpp"
#include "gallery.hpp"

#include <iostream>

void FaceDatabase::loadFromDisk(FacePipeline& pipe) {
    store_.clear();
    cache_.load(cfg::kEmbedCachePath);

    gallery::ScanResult scan = gallery::scan(pipe, cache_);
    for (auto& [id, person] : scan.people) {
        store_.insert(id, std::move(person));
    }
    index_.rebuild(store_.people());

    if (scan.cacheDirty) {
        saveCache();
    }

    std::cout << "Loaded " << scan.totalImages << " images for " << store_.size()
              << " people (" << scan.embedded << " newly embedded)\n";
}

void FaceDatabase::saveCache() const {
    cache_.save(cfg::kEmbedCachePath);
}

std::optional<MatchResult> FaceDatabase::match(const cv::Mat& emb) const {
    return index_.match(emb, store_.people());
}

double FaceDatabase::verify(PersonId personId, const cv::Mat& emb) const {
    const Person* p = store_.find(personId);
    if (p == nullptr) {
        return -1.0;
    }
    return FaceIndex::verify(emb, p->centroid);
}

PersonId FaceDatabase::createPerson(const std::vector<cv::Mat>& samples) {
    const PersonId id = store_.createPerson(samples);
    index_.rebuild(store_.people());
    return id;
}

void FaceDatabase::addSample(PersonId personId, const cv::Mat& emb,
                             const std::string& imagePath) {

    store_.addSample(personId, emb, imagePath);
    index_.rebuild(store_.people());
}

void FaceDatabase::attachImage(PersonId personId, const std::string& relPath,
                               const cv::Mat& emb) {

    if (store_.find(personId) == nullptr) {
        return;
    }
    store_.addImage(personId, relPath);
    cache_.put(relPath, EmbedCache::fileMtime(relPath), embed::normalized(emb));
}
