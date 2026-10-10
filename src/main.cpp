#include "camera.hpp"
#include "config.hpp"
#include "enroll.hpp"
#include "face_database.hpp"
#include "face_pipeline.hpp"
#include "overlay.hpp"
#include "tracker.hpp"

#include <opencv2/highgui.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>

namespace {

    // Quit request set either by the GUI (waitKey) or by a key typed in the
    // terminal. On macOS the imshow window often never receives focus when the
    // app is launched from a shell, so waitKey() never sees q/ESC; reading the
    // terminal directly guarantees the quit keys always work.
    std::atomic<bool> g_quitRequested{false};
    termios g_origTermios{};
    bool g_termiosSaved = false;

    void restoreTerminal() {
        if (g_termiosSaved) {
            tcsetattr(STDIN_FILENO, TCSANOW, &g_origTermios);
            g_termiosSaved = false;
        }
    }

    void onSignal(int) {
        restoreTerminal();
        std::signal(SIGINT, SIG_DFL);
        std::raise(SIGINT);
    }

    // Puts the terminal in raw mode and watches stdin for q/Q/ESC on a detached
    // thread. Returns false when stdin is not an interactive terminal.
    bool watchTerminalQuit() {
        if (isatty(STDIN_FILENO) != 1) {
            return false;
        }

        if (tcgetattr(STDIN_FILENO, &g_origTermios) != 0) {
            return false;
        }

        termios raw = g_origTermios;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN] = 1;
        raw.c_cc[VTIME] = 0;

        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
            return false;
        }
        g_termiosSaved = true;

        std::signal(SIGINT, onSignal);
        std::signal(SIGTERM, onSignal);

        std::thread([] {
            while (!g_quitRequested.load()) {
                unsigned char c = 0;
                if (::read(STDIN_FILENO, &c, 1) <= 0) {
                    break;
                }
                if (c == 'q' || c == 'Q' || c == 27) {
                    g_quitRequested.store(true);
                    break;
                }
            }
        }).detach();

        return true;
    }


    double nowSeconds() {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    struct WindowStats {
        double detectMs = 0;
        double embedMs = 0;
        double matchMs = 0;
        int embeds = 0;
        int frames = 0;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        void add(double d, double e, double m, int n) {
            detectMs += d;
            embedMs += e;
            matchMs += m;
            embeds += n;
            frames++;
        }

        // Prints a stats line and resets the window when the interval elapses.
        HudStats flush(HudStats hud) {
            const double elapsed = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - start)
                                       .count();

            if (elapsed < cfg::kStatsIntervalS || frames == 0) {
                return hud;
            }

            hud.fps = frames / elapsed;
            hud.detectMs = detectMs / frames;
            hud.embedMs = embedMs / frames;
            hud.matchMs = matchMs / frames;
            hud.embeds = embeds;

            std::printf(
                "detect %.2fms | embed %.2fms/frame (%d total) | match %.3fms | "
                "%.1f FPS (%d frames in %.1fs)\n",
                hud.detectMs, hud.embedMs, embeds, hud.matchMs, hud.fps, frames,
                elapsed);

            detectMs = embedMs = matchMs = 0;
            embeds = frames = 0;
            start = std::chrono::steady_clock::now();

            return hud;
        }
    };

}  // namespace

int main() {
    std::cout << "Facial Recognition (C++/OpenCV)\n";

    if (!std::filesystem::exists(cfg::kYuNetPath) ||
        !std::filesystem::exists(cfg::kSFacePath)) {
        std::cerr << "Models not found (" << cfg::kYuNetPath << ", "
                  << cfg::kSFacePath << "). Run from the project root.\n";
        return 1;
    }

    FacePipeline pipe;
    if (!pipe.init()) {
        std::cerr << "Failed to load models.\n";
        return 1;
    }

    FaceDatabase db;
    db.loadFromDisk(pipe);

    Camera cam;
    if (!cam.open()) {
        return 1;
    }

    Tracker tracker;
    Enroller enroller;
    WindowStats acc;
    HudStats hud;

    const bool terminalQuit = watchTerminalQuit();
    std::cout << "Start... press q / ESC";
    if (terminalQuit) {
        std::cout << " (in this terminal or the video window)";
    }
    std::cout << " to quit (or close the window)\n";

    cv::Mat frame;
    bool running = true;
    while (running) {
        if (g_quitRequested.load()) {
            std::cout << "Quit key pressed\n";
            break;
        }

        if (!cam.read(frame)) {
            std::cerr << "Frame read failed.\n";
            break;
        }

        if (!pipe.detect(frame)) {
            std::cerr << "Detection failed.\n";
            break;
        }

        const auto& dets = pipe.detections();
        tracker.update(dets, frame.size());

        double embedMs = 0;
        double matchMs = 0;
        int embeds = 0;

        for (size_t i = 0; i < dets.size(); ++i) {
            Track* t = tracker.trackForDet(i);

            if (t == nullptr || t->misses != 0) {
                continue;
            }

            const int interval = t->locked ? cfg::kVerifyEvery
                                           : cfg::kEmbedEveryUnknown;

            if (t->framesSinceEmbed < interval) {
                continue;
            }

            const cv::Mat emb = pipe.embed(frame, i);
            embedMs += pipe.lastEmbedMs();
            embeds++;
            t->framesSinceEmbed = 0;

            if (emb.empty()) {
                continue;
            }

            const auto m0 = std::chrono::steady_clock::now();
            const auto m = db.match(emb);
            matchMs += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - m0)
                           .count();

            tracker.applyMatch(*t, emb, m);

            if (cfg::kVerboseMatches) {
                std::printf("#%d cos=%.3f -> %s (lockScore %d, hold %d)\n",
                            t->id, m ? m->cosine : 0.0,
                            t->locked ? t->label.c_str() : "unknown",
                            t->lockScore, t->holdFrames);
            }

            if (t->locked) {
                // Recognized: discard any pending enrollment captures.
                if (!t->captureFiles.empty()) {
                    tracker.discardCaptures(*t);
                }
            }
            else {
                // Enroll only faces that match no one (not warm up frames of a known person).
                const bool unmatched = !m || m->cosine < cfg::kCosineLoose;
                if (unmatched) {
                    enroller.maybeCapture(*t, frame, emb, nowSeconds(), db);
                }
            }
        }

        std::vector<Track*> active = tracker.activeTracks();
        acc.add(pipe.lastDetectMs(), embedMs, matchMs, embeds);
        hud = acc.flush(hud);

        // People currently on screen: distinct recognized identities among active tracks.
        std::unordered_set<PersonId> onScreen;
        for (const Track* t : active) {
            if (t->locked && t->personId >= 0) {
                onScreen.insert(t->personId);
            }
        }
        hud.people = static_cast<std::int32_t>(onScreen.size());

        int pending = 0;
        for (const auto& [id, t] : tracker.tracks()) {
            pending += t.captureCount;
        }

        hud.pendingCaptures = pending;
        overlay::draw(frame, active, hud);

        cv::imshow("Face Recognition", frame);

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 'Q' || key == 27) {
            std::cout << "Quit key pressed\n";
            running = false;
            break;
        }

        try {
            if (cv::getWindowProperty("Face Recognition", cv::WND_PROP_VISIBLE) < 1) {
                std::cout << "Window closed\n";
                break;
            }
        }
        catch (const cv::Exception&) {
            break;
        }
    }

    db.saveCache();
    cam.release();
    cv::destroyAllWindows();
    restoreTerminal();
    std::cout << "Done.\n";
    return 0;
}
