#include "person_store.hpp"

#include "embedding.hpp"

void PersonStore::clear() {
    people_.clear();
    nextId_ = 1;
}

void PersonStore::insert(PersonId id, Person person) {
    people_[id] = std::move(person);
    if (id >= nextId_) {
        nextId_ = static_cast<PersonId>(id + 1);
    }
}

PersonId PersonStore::createPerson(const std::vector<cv::Mat>& samples) {
    if (samples.empty()) {
        return -1;
    }

    const PersonId id = nextId_++;
    Person p;
    p.name = "person_" + std::to_string(id);
    for (const auto& s : samples) {
        const cv::Mat n = embed::normalized(s);
        if (!n.empty()) {
            p.samples.push_back(n);
        }
    }

    p.centroid = embed::centroid(p.samples);
    people_[id] = std::move(p);
    return id;
}

void PersonStore::addSample(PersonId personId, const cv::Mat& emb,
                            const std::string& imagePath) {

    const auto it = people_.find(personId);
    if (it == people_.end()) {
        return;
    }

    const cv::Mat n = embed::normalized(emb);
    if (!n.empty()) {
        it->second.samples.push_back(n);
    }

    it->second.images.push_back(imagePath);
    it->second.centroid = embed::centroid(it->second.samples);
}

void PersonStore::addImage(PersonId personId, const std::string& relPath) {
    const auto it = people_.find(personId);
    if (it == people_.end()) {
        return;
    }
    it->second.images.push_back(relPath);
}

const Person* PersonStore::find(PersonId personId) const {
    const auto it = people_.find(personId);
    return it == people_.end() ? nullptr : &it->second;
}
