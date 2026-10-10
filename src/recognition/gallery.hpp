#pragma once

#include <utility>
#include <vector>

#include "embed_cache.hpp"
#include "person.hpp"
#include "types.hpp"

class FacePipeline;

// Discovery + embedding of the known_faces/ gallery.
namespace gallery {

    struct ScanResult {
        std::vector<std::pair<PersonId, Person>> people;
        std::int32_t totalImages = 0;
        std::int32_t embedded = 0;
        bool cacheDirty = false;
    };

    // Walks known_faces/person_N/, returning each person with embedded samples.
    // Reuses cache entries whose mtime matches newly embedded entries are put in
    // `cache` and flagged via cacheDirty.
    ScanResult scan(FacePipeline& pipe, EmbedCache& cache);

}  // namespace gallery
