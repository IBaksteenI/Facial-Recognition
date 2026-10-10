#include "face_index.hpp"

#include "config.hpp"
#include "embedding.hpp"

void FaceIndex::rebuild(const std::unordered_map<PersonId, Person>& people) {
    idxCentroids_.clear();
    idxOwner_.clear();
    idxCentroids_.reserve(people.size());
    idxOwner_.reserve(people.size());

    for (const auto& [id, person] : people) {
        if (person.centroid.empty()) {
            continue;
        }
        idxCentroids_.push_back(person.centroid);
        idxOwner_.push_back(id);
    }
}

std::optional<MatchResult> FaceIndex::match(
    const cv::Mat& emb,
    const std::unordered_map<PersonId, Person>& people) const {
    const cv::Mat q = embed::normalized(emb);

    if (q.empty() || idxCentroids_.empty()) {
        return std::nullopt;
    }

    double best = -1.0;
    PersonId bestId = -1;
    for (size_t i = 0; i < idxCentroids_.size(); ++i) {
        const double c = q.dot(idxCentroids_[i]);
        if (c > best) {
            best = c;
            bestId = idxOwner_[i];
        }
    }

    if (bestId < 0 || best < cfg::kCosineLoose) {
        return std::nullopt;
    }

    const auto it = people.find(bestId);
    return MatchResult{bestId, it != people.end() ? it->second.name : "", best};
}

double FaceIndex::verify(const cv::Mat& emb, const cv::Mat& centroid) {
    if (centroid.empty()) {
        return -1.0;
    }
    const cv::Mat q = embed::normalized(emb);
    return q.empty() ? -1.0 : q.dot(centroid);
}
