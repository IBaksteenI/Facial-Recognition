#include "embed_cache.hpp"

#include <iostream>

void EmbedCache::load(const std::string& path) {
    cache_.clear();
    if (!std::filesystem::exists(path)) {
        return;
    }

    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return;
    }

    const cv::FileNode entries = fs["entries"];
    for (const auto& node : entries) {
        Entry e;
        std::string p;
        node["path"] >> p;
        const cv::FileNode mtNode = node["mtime"];

        if (mtNode.isString()) {
            std::string s;
            mtNode >> s;
            e.mtime = std::stoll(s);
        }
        else if (mtNode.isReal()) {
            e.mtime = static_cast<std::int64_t>(static_cast<double>(mtNode));
        }

        node["emb"] >> e.emb;
        if (!p.empty() && !e.emb.empty()) {
            cache_[p] = std::move(e);
        }
    }
}

void EmbedCache::save(const std::string& path) const {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    cv::FileStorage fs(path, cv::FileStorage::WRITE);

    if (!fs.isOpened()) {
        std::cerr << "Cannot write " << path << "\n";
        return;
    }

    fs << "entries" << "[";
    for (const auto& [p, e] : cache_) {
        // mtime as string: tick counts exceed double precision (2^53).
        fs << "{" << "path" << p << "mtime" << std::to_string(e.mtime) << "emb"
           << e.emb << "}";
    }
    fs << "]";
}

std::optional<cv::Mat> EmbedCache::get(const std::string& relPath,
                                       std::int64_t mtime) const {

    const auto it = cache_.find(relPath);
    if (it == cache_.end() || it->second.mtime != mtime) {
        return std::nullopt;
    }
    return it->second.emb;
}

bool EmbedCache::put(const std::string& relPath, std::int64_t mtime,
                     const cv::Mat& emb) {

    const auto it = cache_.find(relPath);
    if (it != cache_.end() && it->second.mtime == mtime) {
        return false;
    }
    cache_[relPath] = {mtime, emb};
    return true;
}

std::int64_t EmbedCache::fileMtime(const std::filesystem::path& p) {
    std::error_code ec;
    const auto t = std::filesystem::last_write_time(p, ec);
    if (ec) {
        return 0;
    }
    return static_cast<std::int64_t>(t.time_since_epoch().count());
}
