#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "embed_cache.hpp"
#include "face_index.hpp"
#include "person.hpp"
#include "person_store.hpp"
#include "types.hpp"

class FacePipeline;

// Coordinates the person store, the match index, and the embedding cache.
class FaceDatabase {
    public:
        // Scans known_faces/, embedding gallery images (reusing data/embed_cache.yml when fresh).
        void loadFromDisk(FacePipeline& pipe);
        void saveCache() const;

        // Best centroid cosine nullopt when below the accept threshold.
        [[nodiscard]] std::optional<MatchResult> match(const cv::Mat& emb) const;
        // Cosine against one person's centroid (for future 1:1 verification).
        [[nodiscard]] double verify(PersonId personId, const cv::Mat& emb) const;

        [[nodiscard]] PersonId createPerson(const std::vector<cv::Mat>& samples);
        void addSample(PersonId personId, const cv::Mat& emb, const std::string& imagePath);
        // Register an image file (already moved into place) for a person + cache its embedding.
        void attachImage(PersonId personId, const std::string& relPath, const cv::Mat& emb);

        [[nodiscard]] std::int32_t size() const { return store_.size(); }
        [[nodiscard]] const std::unordered_map<PersonId, Person>& people() const { return store_.people(); }

    private:
        PersonStore store_;
        FaceIndex index_;
        EmbedCache cache_;
};
