cmake_minimum_required(VERSION 4.3)
project(facial_recognition)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Homebrew keg-only / opt paths.
foreach(prefix /opt/homebrew/opt /usr/local/opt)
    if(EXISTS "${prefix}/opencv")
        list(APPEND CMAKE_PREFIX_PATH "${prefix}/opencv")
    endif()
endforeach()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs objdetect highgui videoio)

add_executable(facial_recognition
    src/main.cpp
    src/core/embedding.cpp
    src/capture/camera.cpp
    src/vision/face_pipeline.cpp
    src/tracking/tracker.cpp
    src/recognition/person_store.cpp
    src/recognition/face_index.cpp
    src/recognition/embed_cache.cpp
    src/recognition/gallery.cpp
    src/recognition/face_database.cpp
    src/enrollment/enroll.cpp
    src/ui/overlay.cpp
)

target_include_directories(facial_recognition PRIVATE
    src
    src/core
    src/capture
    src/vision
    src/tracking
    src/recognition
    src/enrollment
    src/ui
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(facial_recognition PRIVATE ${OpenCV_LIBS})
