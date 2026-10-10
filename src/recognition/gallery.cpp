#include "gallery.hpp"

#include "config.hpp"
#include "embedding.hpp"
#include "face_pipeline.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <iostream>
#include <regex>
#include <string>

namespace fs = std::filesystem;

namespace {

    bool isImageExt(const std::string& ext) {
        return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
    }

    // Highest scoring face in the image, normalized. Empty on failure / no face.
    cv::Mat embedBestFace(FacePipeline& pipe, const fs::path& imagePath,
                          const std::string& rel) {

        cv::Mat image = cv::imread(imagePath.string());
        if (image.empty()) {
            return {};
        }

        if (!pipe.detect(image) || pipe.detections().empty()) {
            std::cout << "  no face in " << rel << ", skipped\n";
            return {};
        }

        size_t best = 0;
        for (size_t i = 1; i < pipe.detections().size(); ++i) {
            if (pipe.detections()[i].score > pipe.detections()[best].score) {
                best = i;
            }
        }
        return pipe.embed(image, best);
    }

}  // namespace

namespace gallery {

    ScanResult scan(FacePipeline& pipe, EmbedCache& cache) {
        ScanResult result;
        const fs::path base(cfg::kKnownFacesDir);
        const std::regex personRe(R"(person_(\d+))");

        if (!fs::exists(base)) {
            return result;
        }

        for (const auto& entry : fs::directory_iterator(base)) {

            if (!entry.is_directory()) {
                continue;
            }

            const std::string dirName = entry.path().filename().string();
            std::smatch m;
            if (!std::regex_match(dirName, m, personRe)) {
                continue;
            }
            const PersonId id = static_cast<PersonId>(std::stoi(m[1]));

            Person p;
            p.name = dirName;

            for (const auto& img : fs::directory_iterator(entry.path())) {

                if (!img.is_regular_file()) {
                    continue;
                }

                const std::string ext = img.path().extension().string();
                if (!isImageExt(ext)) {
                    continue;
                }
                ++result.totalImages;

                const std::string rel =
                    img.path().lexically_relative(".").generic_string();
                const std::int64_t mt = EmbedCache::fileMtime(img.path());

                cv::Mat emb;
                if (const auto cached = cache.get(rel, mt)) {
                    emb = *cached;
                }
                else {
                    emb = embedBestFace(pipe, img.path(), rel);

                    if (emb.empty()) {
                        continue;
                    }

                    emb = embed::normalized(emb);
                    cache.put(rel, mt, emb);
                    result.cacheDirty = true;
                    ++result.embedded;
                }

                const cv::Mat n = embed::normalized(emb);
                if (!n.empty()) {
                    p.samples.push_back(n);
                    p.images.push_back(rel);
                }
            }

            if (!p.samples.empty()) {
                p.centroid = embed::centroid(p.samples);
                result.people.emplace_back(id, std::move(p));
            }
        }

        return result;
    }

}  // namespace gallery
