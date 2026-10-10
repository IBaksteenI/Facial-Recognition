#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

// mtime-keyed cache of gallery-image embeddings, persisted as YAML.
class EmbedCache {
    public:
        struct Entry {
            std::int64_t mtime = 0;
            cv::Mat emb;
        };

        void load(const std::string& path);
        void save(const std::string& path) const;

        // Cached embedding for relPath if its recorded mtime matches else nullopt.
        std::optional<cv::Mat> get(const std::string& relPath, std::int64_t mtime) const;

        // Stores an embedding returns true if this changed the cache.
        bool put(const std::string& relPath, std::int64_t mtime, const cv::Mat& emb);

        static std::int64_t fileMtime(const std::filesystem::path& p);

    private:
        std::unordered_map<std::string, Entry> cache_;
};
