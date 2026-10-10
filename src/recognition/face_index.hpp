#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <unordered_map>
#include <vector>

#include "person.hpp"

// Flat centroid index for cosine matching.
class FaceIndex {
    public:
        // Rebuilds the parallel centroid/owner arrays from the current people map.
        void rebuild(const std::unordered_map<PersonId, Person>& people);

        // Best centroid cosine nullopt when empty or below the accept threshold.
        std::optional<MatchResult> match(
            const cv::Mat& emb,
            const std::unordered_map<PersonId, Person>& people) const;

        // Cosine against one person's centroid.
        static double verify(const cv::Mat& emb, const cv::Mat& centroid);

    private:
        std::vector<cv::Mat> idxCentroids_;
        std::vector<PersonId> idxOwner_;
};
