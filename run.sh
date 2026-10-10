#!/usr/bin/env bash
# Build (if needed) and start the facial recognition app from the project root.
set -e
cd "$(dirname "$0")"

cmake -B cmake-build-debug -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build cmake-build-debug

exec ./cmake-build-debug/facial_recognition "$@"
