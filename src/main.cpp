#include <INectaCamera.h>
#include <SDL.h>
#include <SDL_opengl.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl2.h>
#include <shared/GlobalOptions.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "backward.hpp"
#include "review.hpp"
#include "utils.hpp"

using namespace CAlkUSB3;
namespace backward {
backward::SignalHandling signal_handler;
}

static std::atomic<uint32_t>* g_frame_lost_counter = nullptr;

class WorkQueue {
    std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<std::packaged_task<void()>> tasks_;
    bool stop_ = false;
    std::vector<std::thread> work_threads_;

   public:
    int num_waiting_tasks() {
        std::unique_lock<std::mutex> lock(mtx_);
        return tasks_.size();
    }

    void enqueue(std::packaged_task<void()>&& t) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            tasks_.push_back(std::move(t));
        }
        cv_.notify_one();
    }

    void run_tasks(int n_threads) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            stop_ = false;
        }
        for (int i = 0; i < n_threads; i++) {
            work_threads_.push_back(std::thread([this]() {
                while (true) {
                    std::unique_lock<std::mutex> lock(mtx_);
                    cv_.wait(lock,
                             [this]() { return !tasks_.empty() || stop_; });
                    if (tasks_.empty() && stop_) break;
                    auto task = std::move(tasks_.front());
                    tasks_.pop_front();
                    lock.unlock();
                    task();
                }
            }));
        }
    }

    void stop() {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : work_threads_) t.join();
        work_threads_.clear();
    }
};

class NectarCapturer {
   public:
    std::atomic<bool> save{false};
    std::atomic<int> analog_gain{10};
    std::atomic<int> cds_gain{1};
    std::atomic<int> shutterspeed{2000};
    std::atomic<int> frame_id{0};

    NectarCapturer()
        : rgb_image((buffer_rows / 2) * preview_cols * 3, 0),
          rgb_image_cropped((buffer_rows / 2) * crop_cols * 3, 0),
          latest_preview(buffer_rows * cols * 2, 0) {
        glGenTextures(1, &rgb_texture);
        glGenTextures(1, &rgb_crop_texture);
        glGenTextures(1, &hist_texture);
    }

   private:
    std::chrono::time_point<std::chrono::steady_clock> last_frame_time;
    std::chrono::steady_clock::time_point last_preview_ts;
    int capture_rows = 192;
    static constexpr int buffer_rows = 512;
    static constexpr int cols = 4096;
    static constexpr int preview_cols = cols / 2;
    static constexpr int crop_cols = cols / 8;

    std::mutex cam_mtx;
    std::mutex preview_mtx;
    std::vector<uint8_t> latest_preview;
    bool preview_ready = false;
    std::atomic<bool> request_preview{false};
    std::atomic<bool> capture_running{false};
    std::thread capture_thread;

    std::vector<Array<uint8_t>> raw_buffers;

    std::vector<uint8_t> rgb_image;
    std::vector<uint8_t> rgb_image_cropped;

    static constexpr int hist_h = 256;
    static constexpr int hist_w = 512;
    static constexpr int histogram_size = hist_h * hist_w * 3;

    GLuint rgb_texture;
    GLuint rgb_crop_texture;
    GLuint hist_texture;
    WorkQueue task_queue;

    std::chrono::steady_clock::time_point last_capture_ts;
    int dropped_frames = 0;
    std::atomic<int64_t> last_capture_interval_ns{0};
    std::atomic<int64_t> expected_capture_interval_ns{0};
    std::atomic<uint32_t> acquired_counter{0};
    std::atomic<uint32_t> cam_acquired_total{0};
    std::atomic<unsigned int> cam_line_period{0};
    std::atomic<float> cam_frame_rate{0.0f};
    std::atomic<unsigned int> cam_packet_size{0};
    std::atomic<unsigned int> cam_max_packet_size{0};
    int last_analog_gain = -1;
    int last_cds_gain = -1;
    int last_shutterspeed = -1;

    std::array<uint8_t, histogram_size> histogram;
    std::atomic<int> preview_crop_offset{(preview_cols - crop_cols) / 2};
    void viz(const uint8_t* const raw_buf) {
        std::array<int, 256> reds;
        std::array<int, 256> greens;
        std::array<int, 256> blues;
        std::fill(reds.begin(), reds.end(), 0);
        std::fill(greens.begin(), greens.end(), 0);
        std::fill(blues.begin(), blues.end(), 0);

        std::fill(histogram.begin(), histogram.end(), 0);
        // make image from raw buffer
        for (int i = 0; i < capture_rows / 2; i++) {
            for (int j = 0; j < preview_cols; j++) {
                const int ind = (j + preview_cols * i) * 3;
                rgb_image[ind] = raw_buf[2 * ((2 * i + 1) * cols + 2 * j) + 1];
                rgb_image[ind + 1] =
                    (raw_buf[2 * ((2 * i + 1) * cols + (2 * j + 1)) + 1] +
                     raw_buf[2 * (2 * i * cols + 2 * j) + 1]) /
                    2;
                rgb_image[ind + 2] =
                    raw_buf[2 * (2 * i * cols + (2 * j + 1)) + 1];

                reds[rgb_image[ind]]++;
                greens[rgb_image[ind + 1]]++;
                blues[rgb_image[ind + 2]]++;
            }
        }
        // punched in crop
        const int crop_offset = get_crop_offset();
        for (int i = 0; i < capture_rows / 2; i++) {
            for (int j = 0; j < crop_cols; j++) {
                const int ind_cropped = (j + crop_cols * i) * 3;
                const int ind = (j + crop_offset + preview_cols * i) * 3;
                for (int k = 0; k < 3; k++) {
                    rgb_image_cropped[ind_cropped + k] = rgb_image[ind + k];
                }
            }
        }
        // make a histogram
        for (int i = 0; i < hist_h; i++) {
            for (int j = 0; j < hist_w; j++) {
                const int ind = (j + i * hist_w) * 3;
                if (i * 100 < reds[j / 2]) {
                    histogram[ind + 0] = 255;
                } else {
                    histogram[ind + 0] = 0;
                }
                if (i * 100 < greens[j / 2]) {
                    histogram[ind + 1] = 255;
                } else {
                    histogram[ind + 1] = 0;
                }
                if (i * 100 < blues[j / 2]) {
                    histogram[ind + 2] = 255;
                } else {
                    histogram[ind + 2] = 0;
                }
            }
        }
    }

    void draw_image(const GLuint image_texture, const uint8_t* const image_data,
                    const int w, const int h, const int w_disp,
                    const int h_disp) {
        glBindTexture(GL_TEXTURE_2D, image_texture);

        // Setup filtering parameters for display
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                        GL_CLAMP_TO_EDGE);  // This is required on WebGL for non
                                            // power-of-two textures
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                        GL_CLAMP_TO_EDGE);  // Same

        // Upload pixels into texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, image_data);

        ImGui::Image((intptr_t)image_texture, ImVec2(w_disp, h_disp));
    }

    void capture_loop(INectaCamera& cam) {
        last_capture_ts = std::chrono::steady_clock::now();
        last_preview_ts =
            std::chrono::steady_clock::now() - std::chrono::milliseconds(33);
        auto last_cam_acq_check = std::chrono::steady_clock::now();
        while (capture_running.load()) {
            int current_shutter = shutterspeed.load();
            if (!save.load()) {
                const int current_gain = analog_gain.load();
                const int current_cds = cds_gain.load();
                if (current_gain != last_analog_gain ||
                    current_cds != last_cds_gain ||
                    current_shutter != last_shutterspeed) {
                    std::lock_guard<std::mutex> lock(cam_mtx);
                    if (current_gain != last_analog_gain) {
                        cam.SetGain(current_gain);
                        last_analog_gain = current_gain;
                    }
                    if (current_cds != last_cds_gain) {
                        cam.SetCDSGain(current_cds);
                        last_cds_gain = current_cds;
                    }
                    if (current_shutter != last_shutterspeed) {
                        cam.SetShutter(current_shutter);
                        last_shutterspeed = current_shutter;
                    }
                }
            }
            const double line_period_s = current_shutter * 1e-7 + 2.1e-6;
            expected_capture_interval_ns.store(
                static_cast<int64_t>(line_period_s * (capture_rows / 2) * 1e9));
            const int new_rows_desired =
                2 * std::ceil(1.0 / 30.0 / line_period_s);

            int new_rows = 1;
            while (new_rows <= new_rows_desired) {
                new_rows *= 2;
            }
            const int new_capture_rows =
                std::max(static_cast<int>(cam.GetMinImageSizeY()),
                         std::min(static_cast<int>(cam.GetMaxImageSizeY()),
                                  std::min(buffer_rows, new_rows)));
            if (capture_rows != new_capture_rows) {
                std::lock_guard<std::mutex> lock(cam_mtx);
                cam.SetAcquire(false);
                cam.SetImageSizeX(cols);
                cam.SetImageSizeY(new_capture_rows);
                cam.SetAcquire(true);
                capture_rows = new_capture_rows;
                raw_buffers.clear();
                const size_t needed =
                    static_cast<size_t>(capture_rows) * cols * 2;
                std::lock_guard<std::mutex> preview_lock(preview_mtx);
                if (latest_preview.size() < needed) {
                    latest_preview.resize(needed);
                }
            }

            BufferPtr raw_image;
            {
                std::lock_guard<std::mutex> lock(cam_mtx);
                raw_image = cam.GetRawDataPtr(true);
            }
            const auto t_capture_1 = std::chrono::steady_clock::now();
            last_capture_interval_ns.store(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t_capture_1 - last_capture_ts)
                    .count());
            last_capture_ts = t_capture_1;

            if (raw_image == BufferPtr::Null) {
                continue;
            }
            acquired_counter.fetch_add(1);
            const auto now = std::chrono::steady_clock::now();
            if (now - last_cam_acq_check >= std::chrono::seconds(1)) {
                std::lock_guard<std::mutex> lock(cam_mtx);
                try {
                    cam_acquired_total.store(cam.GetNumOfAcquiredFrames());
                    if (cam.GetLinePeriodAvailable()) {
                        cam_line_period.store(cam.GetLinePeriod());
                    }
                    if (cam.GetFrameRateAvailable()) {
                        cam_frame_rate.store(cam.GetFrameRate());
                    }
                    if (cam.GetPacketSizeAvailable()) {
                        cam_packet_size.store(cam.GetPacketSize());
                        cam_max_packet_size.store(cam.GetMaxPacketSize());
                    }
                } catch (...) {
                }
                last_cam_acq_check = now;
            }

            if (save.load()) {
                const size_t needed =
                    static_cast<size_t>(capture_rows) * cols * 2;
                Array<uint8_t> raw_copy(needed);
                std::memcpy(raw_copy.Data(), raw_image.Body(), needed);
                raw_buffers.push_back(std::move(raw_copy));
                if (raw_buffers.size() * capture_rows == buffer_rows) {
                    task_queue.enqueue(std::packaged_task<void()>(
                        [frame_id = frame_id.load(), buffer_rows = buffer_rows,
                         capture_rows = capture_rows, cols = cols,
                         raw_buffers = std::move(raw_buffers)]() mutable {
                            std::vector<uint8_t> buf(buffer_rows * cols * 2);
                            int buffer_row_id = 0;
                            for (const auto& raw_image : raw_buffers) {
                                for (int i = 0; i < capture_rows; i++) {
                                    for (int j = 0; j < cols; j++) {
                                        for (int k = 0; k < 2; k++) {
                                            buf[2 * (buffer_row_id * cols + j) +
                                                k] =
                                                raw_image[2 * (i * cols + j) +
                                                          k];
                                        }
                                    }
                                    buffer_row_id =
                                        (buffer_row_id + 1) % buffer_rows;
                                }
                            }

                            std::ofstream fout;
                            std::stringstream ss;

                            ss << "im" << std::setfill('0') << std::setw(6)
                               << frame_id << ".bin";
                            fout.open(ss.str(),
                                      std::ios::out | std::ios::binary);
                            fout.write(reinterpret_cast<char*>(buf.data()),
                                       buf.size());
                            fout.close();
                        }));
                    frame_id.store(frame_id.load() + 1);
                    raw_buffers.clear();
                }
            } else {
                const bool preview_due =
                    now - last_preview_ts >= std::chrono::milliseconds(33);
                if (request_preview.load() || preview_due) {
                    const size_t needed =
                        static_cast<size_t>(capture_rows) * cols * 2;
                    if (raw_image.Body() != nullptr) {
                        std::lock_guard<std::mutex> lock(preview_mtx);
                        std::memcpy(latest_preview.data(), raw_image.Body(),
                                    needed);
                        preview_ready = true;
                        last_preview_ts = now;
                        request_preview.store(false);
                    }
                }
            }
        }
    }

   public:
    void start_capture_thread(INectaCamera& cam) {
        if (capture_running.load()) return;
        capture_running.store(true);
        capture_thread = std::thread([this, &cam]() { capture_loop(cam); });
    }

    void stop_capture_thread() {
        if (!capture_running.load()) return;
        capture_running.store(false);
        if (capture_thread.joinable()) {
            capture_thread.join();
        }
    }

    bool is_capture_running() const { return capture_running.load(); }

    int64_t get_expected_interval_ns() const {
        return expected_capture_interval_ns.load();
    }

    uint32_t get_acquired_count() const { return acquired_counter.load(); }

    uint32_t get_cam_acquired_total() const {
        return cam_acquired_total.load();
    }

    unsigned int get_cam_line_period() const { return cam_line_period.load(); }

    float get_cam_frame_rate() const { return cam_frame_rate.load(); }

    unsigned int get_cam_packet_size() const { return cam_packet_size.load(); }

    unsigned int get_cam_max_packet_size() const {
        return cam_max_packet_size.load();
    }

    int get_preview_width() const { return preview_cols; }
    int get_crop_width() const { return crop_cols; }
    int get_max_crop_offset() const { return preview_cols - crop_cols; }
    int get_crop_offset() const {
        return std::clamp(preview_crop_offset.load(), 0, get_max_crop_offset());
    }
    void set_crop_offset(int offset) {
        const int clamped = std::clamp(offset, 0, get_max_crop_offset());
        preview_crop_offset.store(clamped);
    }

    void capture(INectaCamera& cam) {
        const auto t0 = std::chrono::steady_clock::now();
        request_preview.store(true);
        bool have_preview = false;
        {
            std::lock_guard<std::mutex> lock(preview_mtx);
            if (preview_ready) {
                viz(latest_preview.data());
                have_preview = true;
            }
        }

        const auto t1 = std::chrono::steady_clock::now();
        ImGui::Text("rows = %d", capture_rows);
        if (have_preview) {
            const int preview_data_width = preview_cols;
            const int preview_data_height = capture_rows / 2;
            const float avail_width = ImGui::GetContentRegionAvail().x;
            float desired_width = static_cast<float>(preview_data_width);
            if (avail_width > 0.0f && avail_width < desired_width) {
                desired_width = avail_width;
            }
            float preview_scale = desired_width / preview_data_width;
            if (preview_scale <= 0.0f) preview_scale = 1.0f;
            const int preview_disp_width =
                std::max(1, static_cast<int>(std::round(preview_data_width *
                                                        preview_scale)));
            const int preview_disp_height =
                std::max(1, static_cast<int>(std::round(preview_data_height *
                                                        preview_scale)));
            draw_image(rgb_texture, rgb_image.data(), preview_cols,
                       capture_rows / 2, preview_disp_width,
                       preview_disp_height);
            draw_image(rgb_crop_texture, rgb_image_cropped.data(), crop_cols,
                       capture_rows / 2, preview_disp_width,
                       preview_disp_height);
            draw_image(hist_texture, histogram.data(), hist_w, hist_h, hist_w,
                       hist_h);
        }
        const auto t2 = std::chrono::steady_clock::now();
        ImGui::Text("capture interval: %ld ns",
                    last_capture_interval_ns.load());
        ImGui::Text("main loop takes:  %ld ns", (t1 - t0).count());
        ImGui::Text("drawing takes:    %ld ns", (t2 - t1).count());
        ImGui::Text("fps = %lf", 1e9 / (t2 - last_frame_time).count());
        if (g_frame_lost_counter) {
            dropped_frames = static_cast<int>(g_frame_lost_counter->load());
        }
        ImGui::Text("dropped frames %d", dropped_frames);
        last_frame_time = t2;
    }

    void start_saving() {
        dropped_frames = 0;
        save.store(true);
        frame_id.store(0);
        last_capture_ts = std::chrono::steady_clock::now();
        if (g_frame_lost_counter) {
            g_frame_lost_counter->store(0);
        }
        task_queue.run_tasks(4);
    }

    void stop_saving() {
        dropped_frames = 0;
        save.store(false);
        task_queue.stop();
    }
};

void print_video_modes(INectaCamera& cam) {
    auto acc = cam.GetAvailableVideoModes();
    for (size_t cc = 0; cc < acc.Size(); cc++) {
        std::cerr << (int)(acc[cc].GetVideoMode()) << std::endl;
    }
    std::cerr << "selected: " << (int)cam.GetVideoMode() << std::endl;
}

struct CameraStatus {
    bool connected = false;
    std::string last_error;
    std::string usb_speed;
    std::string name;
    std::string serial;
    std::chrono::steady_clock::time_point last_poll;
};

void __stdcall on_frame_lost(IVideoSource& source) {
    (void)source;
    if (g_frame_lost_counter) {
        g_frame_lost_counter->fetch_add(1);
    }
}

bool connect_camera(INectaCamera& cam, CameraStatus& status) {
    try {
        Array<String> cam_list = cam.GetCameraList();
        if (cam_list.Size() == 0) {
            status.connected = false;
            status.last_error = "No AlkUSB3 camera detected";
            return false;
        }

        if (cam.IsOwnedByAnotherProcess(0)) {
            status.connected = false;
            status.last_error = "Camera is owned by another process";
            return false;
        }

        cam.SetCamera(0);
        cam.Init();

        cam.SetUseBulkEndPoint(false);
        cam.SetADCResolution(12);
        cam.SetVideoMode(1);
        cam.SetColorCoding(ColorCoding::Raw16);
        cam.SetEnableImageThread(false);
        cam.AllocRawFrames(64);
        cam.SetPacketSize(cam.GetMaxPacketSize());
        cam.SetAcquire(true);
        cam.FrameLost().SetVideoSourceCallback(on_frame_lost);
        if (g_frame_lost_counter) {
            g_frame_lost_counter->store(0);
        }

        status.connected = true;
        status.last_error.clear();
        status.usb_speed =
            cam.GetSuperSpeed() ? "5000 Mb/s (USB 3.x)" : "480 Mb/s (USB 2.0)";
        status.name = cam.GetName();
        status.serial = cam.GetSerialNumber();
        return true;
    } catch (const Exception& ex) {
        status.connected = false;
        status.last_error =
            std::string("Connect failed: ") + ex.Name() + " " + ex.Message();
    } catch (...) {
        status.connected = false;
        status.last_error = "Connect failed: unknown error";
    }
    return false;
}

void disconnect_camera(INectaCamera& cam, CameraStatus& status) {
    if (!status.connected) return;
    try {
        cam.SetAcquire(false);
        cam.Close();
    } catch (...) {
    }
    status.connected = false;
    status.usb_speed.clear();
    status.name.clear();
    status.serial.clear();
}

struct SdlGlGui {
    std::string glsl_version;
    SDL_GLContext gl_context;
    SDL_Window* window;
    SdlGlGui() {
        nectar::configure_sdl_touch();
        // Setup SDL
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER |
                     SDL_INIT_GAMECONTROLLER) != 0) {
            printf("Error: %s\n", SDL_GetError());
            std::exit(1);
        }

        // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
        // GL ES 2.0 + GLSL 100
        glsl_version = "#version 100";
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                            SDL_GL_CONTEXT_PROFILE_ES);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#elif defined(__APPLE__)
        // GL 3.2 Core + GLSL 150
        glsl_version = "#version 150";
        SDL_GL_SetAttribute(
            SDL_GL_CONTEXT_FLAGS,
            SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);  // Always required on Mac
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                            SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
        // GL 3.0 + GLSL 130
        glsl_version = "#version 130";
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                            SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif

        // Create window with graphics context
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
        SDL_WindowFlags window_flags =
            (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE |
                              SDL_WINDOW_ALLOW_HIGHDPI);
        window =
            SDL_CreateWindow("Nectar UI", SDL_WINDOWPOS_CENTERED,
                             SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
        gl_context = SDL_GL_CreateContext(window);
        SDL_GL_MakeCurrent(window, gl_context);
        SDL_GL_SetSwapInterval(1);  // Enable vsync
    }

    ~SdlGlGui() {
        SDL_GL_DeleteContext(gl_context);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
};

int main(int argc, char* argv[]) {
    bool enable_diag_logging = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--diag-log") == 0 ||
            std::strcmp(argv[i], "--verbose-diagnostics") == 0) {
            enable_diag_logging = true;
        }
    }
    SdlGlGui sdl_gl_gui;
    GlobalOptions::DisableFrequencyMeters = true;
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForOpenGL(sdl_gl_gui.window, sdl_gl_gui.gl_context);
    ImGui_ImplOpenGL3_Init(sdl_gl_gui.glsl_version.c_str());

    // Build atlas
    unsigned char* tex_pixels = NULL;
    int tex_w, tex_h;
    io.Fonts->GetTexDataAsRGBA32(&tex_pixels, &tex_w, &tex_h);
    ImVec4 clear_color = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    INectaCamera& cam = INectaCamera::Create();
    CameraStatus cam_status;
    const auto poll_interval = std::chrono::milliseconds(250);
    cam_status.last_poll = std::chrono::steady_clock::now() - poll_interval;
    bool done = false;
    bool request_quit_popup = false;
    enum class UiMode { Capture, Review };
    UiMode ui_mode = UiMode::Capture;
    nectar::TouchHandler touch_handler;
    std::atomic<uint32_t> frame_lost{0};
    g_frame_lost_counter = &frame_lost;

    NectarCapturer nc;
    std::string output_dir;
    auto start_capture_session = [&]() {
        if (nc.save.load() || !cam_status.connected) {
            return;
        }
        nc.start_saving();
        std::stringstream ss;
        const auto now = std::chrono::system_clock::now();
        const time_t itt = std::chrono::system_clock::to_time_t(now);
        const auto gt = std::gmtime(&itt);
        ss << nectar::k_capture_root << "/" << std::put_time(gt, "%F-%H-%M-%S");
        output_dir = ss.str();
        std::filesystem::create_directory(output_dir);
        std::filesystem::current_path(output_dir);
        std::ofstream settings_file(output_dir + "/capture_settings.txt");
        if (settings_file.is_open()) {
            settings_file << "cds_gain=" << nc.cds_gain.load() << '\n';
            settings_file << "analog_gain=" << nc.analog_gain.load() << '\n';
            settings_file << "shutterspeed=" << nc.shutterspeed.load() << '\n';
        }
    };
    auto stop_capture_session = [&]() {
        if (!nc.save.load()) {
            return;
        }
        nc.stop_saving();
    };

    nectar::ReviewController review_controller;

    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            touch_handler.handle_event(event, sdl_gl_gui.window);
            if (!touch_handler.should_ignore_event(event)) {
                ImGui_ImplSDL2_ProcessEvent(&event);
            }
            if (event.type == SDL_QUIT) request_quit_popup = true;
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE &&
                event.window.windowID == SDL_GetWindowID(sdl_gl_gui.window))
                request_quit_popup = true;
            if (event.type == SDL_KEYDOWN && event.key.repeat == 0 &&
                event.key.keysym.sym == SDLK_SPACE) {
                if (ui_mode == UiMode::Capture) {
                    if (nc.save.load()) {
                        stop_capture_session();
                    } else {
                        start_capture_session();
                    }
                }
            }
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        touch_handler.apply_to_imgui(sdl_gl_gui.window);
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::Begin("nectar", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize);
        const auto now = std::chrono::steady_clock::now();
        if (ui_mode == UiMode::Capture) {
            if (now - cam_status.last_poll >= poll_interval) {
                cam_status.last_poll = now;
                if (!cam_status.connected) {
                    if (connect_camera(cam, cam_status)) {
                        if (!nc.is_capture_running()) {
                            nc.start_capture_thread(cam);
                        }
                    }
                }
            }
            const ImVec2 save_button_size(220.0f,
                                          nectar::k_slider_track_height);
            const ImVec2 quit_button_size(220.0f,
                                          nectar::k_slider_track_height);
            const float row_start_y = ImGui::GetCursorPosY();
            if (nc.save.load()) {
                if (ImGui::Button("Stop saving", save_button_size)) {
                    stop_capture_session();
                }
            } else {
                if (ImGui::Button("Start saving", save_button_size) &&
                    cam_status.connected) {
                    start_capture_session();
                }
            }
            const ImVec2 content_max = ImGui::GetWindowContentRegionMax();
            const float right_group_width =
                quit_button_size.x + save_button_size.x + 8.0f;
            const float right_group_x =
                std::max(ImGui::GetCursorPosX(),
                         content_max.x - right_group_width);
            ImGui::SetCursorPos(ImVec2(right_group_x, row_start_y));
            ImGui::BeginDisabled(nc.save.load());
            if (ImGui::Button("Review captures", save_button_size)) {
                ui_mode = UiMode::Review;
            }
            ImGui::EndDisabled();
            ImGui::SameLine(0.0f, 8.0f);
            if (ImGui::Button("Quit", quit_button_size)) {
                request_quit_popup = true;
            }
            ImGui::SetCursorPosY(row_start_y + save_button_size.y + 6.0f);
            if (!cam_status.connected) {
                ImGui::Text("Connect a camera to start saving.");
            }
            if (!nc.save.load()) {
                bool cds_gain_enabled = nc.cds_gain.load() != 0;
                const float slider_width =
                    ImGui::GetContentRegionAvail().x * 0.7f;
                ImGui::BeginGroup();
                if (nectar::draw_large_checkbox_inline(
                        "cds_gain", &cds_gain_enabled)) {
                    nc.cds_gain.store(cds_gain_enabled ? 1 : 0);
                }
                ImGui::EndGroup();
                ImGui::SameLine(0.0f, 24.0f);
                ImGui::BeginGroup();
                int analog_gain = nc.analog_gain.load();
                int shutter = nc.shutterspeed.load();
                if (nectar::draw_thick_slider_int_width(
                        "analog_gain", &analog_gain, 0, 20, slider_width)) {
                    nc.analog_gain.store(analog_gain);
                }
                ImGui::EndGroup();
                if (nectar::draw_thick_slider_int("shutterspeed (* 100 ns)",
                                                  &shutter, 0, 10000)) {
                    nc.shutterspeed.store(shutter);
                }
            } else {
                ImGui::Text("Saving to: %s", output_dir.c_str());
            }
            ImGui::Text("Zoom preview position");
            const float slider_height = nectar::k_slider_track_height;
            const float slider_width = ImGui::GetContentRegionAvail().x;
            const float handle_ratio =
                static_cast<float>(nc.get_crop_width()) /
                static_cast<float>(nc.get_preview_width());
            const float handle_width = slider_width * handle_ratio;
            const int max_crop_offset = nc.get_max_crop_offset();
            const int crop_offset = nc.get_crop_offset();
            const float slider_travel =
                std::max(slider_width - handle_width, 0.0f);
            const float normalized =
                max_crop_offset > 0
                    ? static_cast<float>(crop_offset) / max_crop_offset
                    : 0.0f;
            ImVec2 slider_pos = ImGui::GetCursorScreenPos();
            const float handle_x =
                slider_pos.x +
                (slider_travel > 0 ? normalized * slider_travel : 0.0f);
            ImGui::InvisibleButton("##crop_slider",
                                   ImVec2(slider_width, slider_height));
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            const ImU32 track_color = ImGui::GetColorU32(ImGuiCol_FrameBg);
            const ImU32 border_color = ImGui::GetColorU32(ImGuiCol_Border);
            const float rounding =
                std::min(nectar::k_slider_corner_radius, slider_height * 0.5f);
            ImVec2 slider_end(slider_pos.x + slider_width,
                              slider_pos.y + slider_height);
            draw_list->AddRectFilled(slider_pos, slider_end, track_color,
                                     rounding);
            ImVec2 handle_min(handle_x, slider_pos.y);
            ImVec2 handle_max(handle_x + handle_width,
                              slider_pos.y + slider_height);
            const ImU32 handle_color = ImGui::GetColorU32(ImGuiCol_SliderGrab);
            draw_list->AddRectFilled(handle_min, handle_max, handle_color,
                                     rounding);
            draw_list->AddRect(handle_min, handle_max, border_color, rounding);
            if ((ImGui::IsItemActive() && ImGui::GetIO().MouseDown[0]) ||
                (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0))) {
                const float mouse_x = ImGui::GetIO().MousePos.x;
                float new_pos = mouse_x - slider_pos.x - handle_width * 0.5f;
                new_pos = std::clamp(new_pos, 0.0f, slider_travel);
                const float denom = slider_travel > 0 ? slider_travel : 1.0f;
                const float new_norm = new_pos / denom;
                const int new_offset =
                    static_cast<int>(std::round(new_norm * max_crop_offset));
                nc.set_crop_offset(new_offset);
            }
            ImGui::Dummy(ImVec2(0.0f, 10.0f));

            try {
                if (cam_status.connected) {
                    nc.capture(cam);
                }
            } catch (const Exception& ex) {
                std::cerr << "Exception " << ex.Name() << " occurred"
                          << std::endl
                          << ex.Message() << std::endl;
                ImGui::Text("Exception %s %s", ex.Name(), ex.Message());
                disconnect_camera(cam, cam_status);
                nc.stop_saving();
            } catch (...) {
                std::cerr << "Unhandled exception" << std::endl;
                disconnect_camera(cam, cam_status);
                nc.stop_saving();
            }
        } else {
            const ImVec2 review_button_size(220.0f,
                                            nectar::k_slider_track_height);
            const ImVec2 quit_button_size(220.0f,
                                          nectar::k_slider_track_height);
            const float row_start_y = ImGui::GetCursorPosY();
            if (ImGui::Button("Back to Capture", review_button_size)) {
                ui_mode = UiMode::Capture;
            }
            const ImVec2 content_max = ImGui::GetWindowContentRegionMax();
            const float right_x =
                std::max(ImGui::GetCursorPosX(),
                         content_max.x - quit_button_size.x);
            ImGui::SetCursorPos(ImVec2(right_x, row_start_y));
            if (ImGui::Button("Quit", quit_button_size)) {
                request_quit_popup = true;
            }
            ImGui::SetCursorPosY(row_start_y + review_button_size.y + 6.0f);
            ImGui::Separator();
            review_controller.update(now);
            if (review_controller.mode_is_listing()) {
                review_controller.render_listing();
            } else {
                review_controller.render_review();
            }
        }
        static auto last_diag = std::chrono::steady_clock::now();
        static uint32_t last_acq_count = 0;
        static double last_acq_fps = 0.0;
        static uint32_t last_acq_total = 0;
        static uint32_t last_cam_acq_total = 0;
        static uint32_t last_cam_acq_delta = 0;
        static uint32_t last_lost_count = 0;
        static double last_lost_fps = 0.0;
        static uint32_t last_lost_total = 0;
        if (ui_mode == UiMode::Capture && cam_status.connected &&
            nc.is_capture_running()) {
            const auto dt =
                std::chrono::duration<double>(now - last_diag).count();
            if (dt >= 1.0) {
                const uint32_t acq_total = nc.get_acquired_count();
                const uint32_t acq_delta = acq_total >= last_acq_total
                                               ? (acq_total - last_acq_total)
                                               : acq_total;
                last_acq_total = acq_total;
                last_acq_count = acq_delta;
                last_acq_fps = acq_delta / dt;
                const uint32_t cam_acq_total = nc.get_cam_acquired_total();
                const uint32_t cam_delta =
                    cam_acq_total >= last_cam_acq_total
                        ? (cam_acq_total - last_cam_acq_total)
                        : cam_acq_total;
                last_cam_acq_total = cam_acq_total;
                last_cam_acq_delta = cam_delta;
                if (g_frame_lost_counter) {
                    const uint32_t lost_total =
                        static_cast<uint32_t>(g_frame_lost_counter->load());
                    const uint32_t lost_delta =
                        lost_total >= last_lost_total
                            ? (lost_total - last_lost_total)
                            : lost_total;
                    last_lost_total = lost_total;
                    last_lost_count = lost_delta;
                    last_lost_fps = lost_delta / dt;
                }
                const int64_t expected_interval = nc.get_expected_interval_ns();
                const double expected_fps =
                    expected_interval > 0 ? 1e9 / expected_interval : 0.0;
                const unsigned int cam_line_period = nc.get_cam_line_period();
                const float cam_frame_rate = nc.get_cam_frame_rate();
                const unsigned int cam_packet = nc.get_cam_packet_size();
                const unsigned int cam_packet_max =
                    nc.get_cam_max_packet_size();
                if (enable_diag_logging) {
                    std::cerr << "diag: acquired_fps=" << std::fixed
                              << std::setprecision(2) << last_acq_fps
                              << " acquired_frames=" << last_acq_count
                              << " cam_acquired_frames=" << last_cam_acq_delta
                              << " app_vs_cam_delta="
                              << (static_cast<int64_t>(last_cam_acq_delta) -
                                  static_cast<int64_t>(last_acq_count))
                              << " lost_fps=" << last_lost_fps
                              << " lost_frames=" << last_lost_count
                              << " expected_fps=" << expected_fps
                              << " expected_interval_ns=" << expected_interval
                              << " cam_line_period=" << cam_line_period
                              << " cam_frame_rate=" << cam_frame_rate
                              << " cam_packet=" << cam_packet
                              << " cam_packet_max=" << cam_packet_max
                              << std::endl;
                }
                last_diag = now;
            }
        }

        if (ui_mode == UiMode::Capture && cam_status.connected) {
            ImGui::Text("Camera: %s", cam_status.name.c_str());
            ImGui::Text("Serial: %s", cam_status.serial.c_str());
            ImGui::Text("USB link: %s", cam_status.usb_speed.c_str());
            ImGui::Text("acquired fps: %.2f", last_acq_fps);
            ImGui::Text("acquired frames (1s): %u", last_acq_count);
            ImGui::Text("lost fps: %.2f", last_lost_fps);
            ImGui::Text("lost frames (1s): %u", last_lost_count);
            ImGui::Text("expected interval: %ld ns",
                        nc.get_expected_interval_ns());
            const int64_t expected_interval = nc.get_expected_interval_ns();
            if (expected_interval > 0) {
                const double expected_fps = 1e9 / expected_interval;
                ImGui::Text("expected fps: %.2f", expected_fps);
            }
        } else if (ui_mode == UiMode::Capture) {
            ImGui::Text("Camera: not connected");
            if (!cam_status.last_error.empty()) {
                ImGui::Text("Status: %s", cam_status.last_error.c_str());
            } else {
                ImGui::Text("Status: polling for camera...");
            }
        }
        if (request_quit_popup) {
            ImGui::OpenPopup("Confirm Quit");
            request_quit_popup = false;
        }
        if (ImGui::BeginPopupModal("Confirm Quit", nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            const ImVec2 confirm_button_size(180.0f,
                                             nectar::k_slider_track_height);
            ImGui::Text("Are you sure you want to exit?");
            if (ImGui::Button("Cancel", confirm_button_size)) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            bool should_close = false;
            if (ImGui::Button("Quit", confirm_button_size)) {
                should_close = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Shut Down", confirm_button_size)) {
                std::system("sudo shutdown -h now");
                should_close = true;
            }
            if (should_close) {
                done = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        ImGui::End();
        ImGui::PopStyleVar(1);
        ImGui::Render();
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(clear_color.x * clear_color.w,
                     clear_color.y * clear_color.w,
                     clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(sdl_gl_gui.window);
    }
    nc.stop_capture_thread();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    disconnect_camera(cam, cam_status);
    INectaCamera::Destroy(cam);
    return 0;
}
