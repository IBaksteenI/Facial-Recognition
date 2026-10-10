# Facial Recognition

A real time face recognition system written in C++26 with OpenCV. It detects faces from a webcam with YuNet, tracks them over time, recognizes known identities with SFace embeddings and cosine similarity, and automatically enrolls unknown faces as new people.

## Features

* Webcam face detection with YuNet (downscaled to 640px for speed, boxes scaled back)
* Multi face tracking with center distance association and lock, hold, forget hysteresis
* Recognition with SFace 1x128 embeddings matched against per person centroids
* Automatic enrollment of unknown faces (captures crops, promotes to a new person)
* Persistent embedding cache so gallery images are only embedded once
* Live overlay with boxes, labels, and a stats HUD
* Resolution independent thresholds (fractions of frame height)

## How it works in detail

The system runs one continuous loop while the camera is open. On startup it:

1. Loads the two ONNX models (`models/yunet.onnx` for detection, `models/sface.onnx` for embeddings)
2. Scans the `known_faces/` gallery, embedding every image with the cached results in `data/embed_cache.yml`
3. Opens the webcam, probing for the preferred camera modes in order

### The per frame pipeline

Each frame flows through these stages:

`capture > detect > track > embed > match > lock/decay > enroll > overlay > display`

1. Capture. `Camera::read` grabs the next frame from the camera (`src/capture/camera.cpp`). Preferred modes are tried in order and the first exactly supported one wins, otherwise the driver default is used.

2. Detect. `FacePipeline::detect` (`src/vision/face_pipeline.cpp`) runs YuNet on the frame. If the frame is wider than 640px it is downscaled first and the detected boxes and landmarks are scaled back to source coordinates. Detections smaller than 6% of the frame height are discarded.

3. Track. `Tracker::update` (`src/tracking/tracker.cpp`) associates each detection to a track using greedy nearest center distance (threshold 12% of frame width). Each track stores a box, its center, an id, and match state. Tracks age every frame and are forgotten after 20 consecutive misses.

4. Embed. Every 3 frames for an unknown track (every 8 for a locked one) the face is aligned and embedded by SFace into a 1x128 L2 normalized vector. Detection still runs every frame; only embedding is throttled to save compute.

5. Match. `FaceIndex::match` (`src/recognition/face_index.cpp`) computes the cosine between the new embedding and every known person's centroid. The best match is returned only if it clears the loose threshold of 0.363.

6. Lock or decay. `Tracker::applyMatch` implements the hysteresis:
   * A match above the strict threshold (0.45) increments the track's lock score, capped at 5.
   * When the lock score reaches 5 the track is locked to that identity and held for 12 frames.
   * A lock is immediately released if a different identity scores strict, or decays one point per weak frame and unlocks at zero.
   * Loose matches (0.363 to 0.45) refresh the hold of the current identity but never push an unlocked track up or down.

7. Enroll. If a track is still unknown and its best match stays below the loose threshold, `Enroller::maybeCapture` (`src/enrollment/enroll.cpp`) saves a padded crop to `unknown_faces/` and stores the embedding, throttled by a 2 second cooldown. After 10 captures the track is promoted: a new `known_faces/person_N/` directory is created, the crops are moved into it, and the track locks to the new person immediately.

8. Overlay and display. `overlay::draw` (`src/ui/overlay.cpp`) draws a green box with the person name for locked tracks, a red box with a question mark for unknown ones, plus an FPS, detect, embed, and match timing HUD. The frame is shown and the loop waits 1ms.

9. Exit. The loop stops on `q`, `Q`, ESC, closing the window, a failed frame read, or a failed detection. The embedding cache is saved to disk before exiting.

### File map

```
root
  CMakeLists.txt                    build config, links OpenCV
  run.sh                            build and run helper (Unix)
  models/                           yunet.onnx and sface.onnx
  known_faces/person_N/             gallery images, one folder per identity
  unknown_faces/                    staging area for pending enrollment crops
  data/embed_cache.yml              mtime keyed embedding cache
  src/
    main.cpp                        main loop, stats window, quit handling
    core/config.hpp                 all tunables and paths
    core/types.hpp                  PersonId and TrackId types
    core/embedding.cpp              L2 normalization and centroid helper
    capture/camera.cpp              webcam open and mode probing
    vision/face_pipeline.cpp        YuNet detect + SFace embed
    tracking/tracker.cpp            track association and lock hysteresis
    recognition/face_database.cpp   coordinates store, index, cache
    recognition/face_index.cpp      cosine match against centroids
    recognition/person_store.cpp    person map and id allocation
    recognition/gallery.cpp         scans and embeds the gallery
    recognition/embed_cache.cpp     YAML embedding cache
    enrollment/enroll.cpp           unknown capture and promotion
    ui/overlay.cpp                  drawing boxes, labels, HUD
```

### Match thresholds

All configured in `src/core/config.hpp`.

| Setting              | Value | Meaning                                                     |
|----------------------|-------|-------------------------------------------------------------|
| kCosineLoose         | 0.363 | accept threshold for a match                                |
| kCosineStrict        | 0.45  | threshold to count toward a lock                            |
| kLockFrames          | 5     | strict matches to lock                                      |
| kHoldFrames          | 12    | frames a lock is held after confirmation                    |
| kForgetMisses        | 20    | missed frames before a track is dropped                     |
| kEmbedEveryUnknown   | 3     | frames between embeddings for unknown tracks                |
| kVerifyEvery         | 8     | frames between embeddings for locked tracks                 |
| kUnknownCaptureCount | 10    | crops before enrollment                                     |
| kCaptureCooldownS    | 2.0   | seconds between enrollment captures                         |
| kCropPadFrac         | 0.25  | crop padding as a fraction of face size                     |
| kMinFaceSizeFrac     | 0.06  | smallest accepted face as a fraction of frame height        |
| kMaxCenterDistFrac   | 0.12  | max track association distance as a fraction of frame width |
| kDetectWidth         | 640   | frames are downscaled to this width for detection           |
| kScoreThreshold      | 0.6   | YuNet detection score threshold                             |
| kStatsIntervalS      | 2.0   | how often stats are printed and the HUD updated             |

## Requirements

* CMake 4.3 or newer
* A C++26 capable compiler (GCC 14, Clang 18, MSVC 2022 17.13 or newer)
* Ninja build system
* OpenCV with the core, imgproc, imgcodecs, objdetect, highgui, and videoio modules
* A webcam on device index 0

Platform specific setup:

macOS
```
brew install cmake ninja opencv
```
Homebrew's OpenCV is found automatically via the keg only opt paths in `CMakeLists.txt`.

Linux (Debian and Ubuntu)
```
sudo apt update
sudo apt install cmake ninja-build g++ libopencv-dev
```
The system OpenCV is usually fine, but if your distro ships an old one you can install a newer build manually. The CMake path probing only touches Homebrew's opt folders, which do not exist on Linux, so the default `find_package` path is used.

Windows
```
git clone https://github.com/microsoft/vcpkg
cd vcpkg && bootstrap-vcpkg.bat
vcpkg install opencv4[core,imgproc,imgcodecs,objdetect,highgui,videoio]:x64-windows
```
Then configure with the vcpkg toolchain file:

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>\scripts\buildsystems\vcpkg.cmake
cmake --build build
```

## Download and run

Clone the repository:

```
git clone https://github.com/IBaksteenI/Facial-Recognition.git
cd facial_recognition
```

The model files in `models/` are committed to the repo, so no extra download is needed. They are the face detection (YuNet) and face recognition (SFace) models from the OpenCV Zoo.

The application expects relative paths, so always run it from the project root.

### Unix (macOS and Linux)

```
./run.sh
```

The script configures, builds if needed, and launches the app.

### Or build manually

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/facial_recognition
```

### Windows

Four commands as shown above, then:

```
.\build\facial_recognition.exe
```

## Quitting

* Press `q` or `Q`
* Press ESC
* Close the program window

On any of these the embedding cache is written back to `data/embed_cache.yml` and the camera is released.

## How identities are stored

Add a folder under `known_faces/` named `person_N` (automatic as N grows, for example `person_1`, `person_2`). Put one or more clear face photos inside as `.jpg`, `.jpeg`, or `.png`. On startup each image is embedded at its highest scoring face and the mean of the images is used as the match centroid. If you add or replace images after the first run, the cache detects the changed file timestamp and reembeds only those files.

New people appear automatically: an unknown face that stays on screen and collects 10 samples is promoted to a new `person_N` folder and shows a name on the next frame.

## Troubleshooting

* No camera opened. Grant camera permission. On macOS this lives in System Settings, Privacy and Security, Camera. The program only opens device index 0.
* "Models not found". You started from the wrong directory. Run from the project root.
* No faces recognized. Check exposure and lighting, the face should cover at least 6% of the frame. Try tuning `kScoreThreshold` and the cosine thresholds in `src/core/config.hpp`.
* Slow detection. Detection is already downscaled to 640px. Embedding is throttled per track; reducing `kVerifyEvery` lowers verification load.