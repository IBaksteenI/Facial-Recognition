#pragma once

#include <cstdint>

// Stable, fixed-width id types for the whole project.
// Both are signed because -1 is used as an "unassigned / invalid" sentinel.
using PersonId = std::int32_t;
using TrackId = std::int32_t;
