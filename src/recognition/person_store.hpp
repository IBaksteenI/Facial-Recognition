#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <unordered_map>
#include <vector>

#include "person.hpp"
#include "types.hpp"

// personId -> Person map, plus id allocation.
class PersonStore {
    public:
        void clear();

        // Inserts under an explicit id (used when loading from disk); bumps nextId_.
        void insert(PersonId id, Person person);

        [[nodiscard]] PersonId createPerson(const std::vector<cv::Mat>& samples);

        void addSample(PersonId personId, const cv::Mat& emb, const std::string& imagePath);
        void addImage(PersonId personId, const std::string& relPath);

        // Null when personId is unknown.
        const Person* find(PersonId personId) const;

        std::int32_t size() const { return static_cast<std::int32_t>(people_.size()); }
        const std::unordered_map<PersonId, Person>& people() const { return people_; }

    private:
        std::unordered_map<PersonId, Person> people_;
        PersonId nextId_ = 1;
};
