#include <SDL.h>
#include <SDL_opengl.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl2.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "backward.hpp"
#include "utils.hpp"

namespace backward {
backward::SignalHandling signal_handler;
}

namespace fs = std::filesystem;

struct SdlGlGui {
    SDL_Window* window = nullptr;
    SDL_GLContext gl_context = nullptr;
    std::string glsl_version;

    SdlGlGui() {
        nectar::configure_sdl_touch();
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER |
                     SDL_INIT_GAMECONTROLLER) != 0) {
            std::cerr << "SDL init failed: " << SDL_GetError() << std::endl;
            std::exit(1);
        }

#if defined(IMGUI_IMPL_OPENGL_ES2)
        glsl_version = "#version 100";
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                            SDL_GL_CONTEXT_PROFILE_ES);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#elif defined(__APPLE__)
        glsl_version = "#version 150";
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS,
                            SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                            SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
        glsl_version = "#version 130";
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                            SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif

        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
        SDL_WindowFlags window_flags =
            (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE |
                              SDL_WINDOW_ALLOW_HIGHDPI);
        window =
            SDL_CreateWindow("Nectar Review", SDL_WINDOWPOS_CENTERED,
                             SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
        gl_context = SDL_GL_CreateContext(window);
        SDL_GL_MakeCurrent(window, gl_context);
        SDL_GL_SetSwapInterval(1);
    }

    ~SdlGlGui() {
        SDL_GL_DeleteContext(gl_context);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
};

struct CaptureEntry {
    fs::path path;
    std::string name;
    fs::file_time_type timestamp;
};

class CapturePreviewer {
   public:
    static constexpr int buffer_rows = 512;
    static constexpr int cols = 4096;
    static constexpr int preview_cols = cols / 2;
    static constexpr int crop_cols = cols / 8;
    static constexpr int hist_h = 256;
    static constexpr int hist_w = 512;
    static constexpr int histogram_size = hist_h * hist_w * 3;

    CapturePreviewer()
        : rgb_image((buffer_rows / 2) * preview_cols * 3, 0),
          rgb_image_cropped((buffer_rows / 2) * crop_cols * 3, 0) {
        glGenTextures(1, &rgb_texture);
        glGenTextures(1, &rgb_crop_texture);
        glGenTextures(1, &hist_texture);
    }

    ~CapturePreviewer() {
        glDeleteTextures(1, &rgb_texture);
        glDeleteTextures(1, &rgb_crop_texture);
        glDeleteTextures(1, &hist_texture);
    }

    bool load_file(const fs::path& file_path, std::string& error) {
        error.clear();
        const size_t expected_bytes =
            static_cast<size_t>(buffer_rows) * cols * 2;
        std::error_code ec;
        const auto file_size = fs::file_size(file_path, ec);
        if (ec || file_size != expected_bytes) {
            std::stringstream ss;
            ss << "Unexpected file size for " << file_path.filename() << " ("
               << file_size << " vs " << expected_bytes << ")";
            error = ss.str();
            return false;
        }
        std::vector<uint8_t> raw(expected_bytes);
        std::ifstream fin(file_path, std::ios::binary);
        if (!fin) {
            error = "Failed to open bin file";
            return false;
        }
        fin.read(reinterpret_cast<char*>(raw.data()), raw.size());
        if (!fin) {
            error = "Failed to read bin file";
            return false;
        }
        generate_preview(raw.data());
        return true;
    }

    void draw_preview(float max_width) {
        const int preview_data_width = preview_cols;
        const int preview_data_height = buffer_rows / 2;
        float desired_width = static_cast<float>(preview_data_width);
        if (max_width > 0.0f && max_width < desired_width) {
            desired_width = max_width;
        }
        float preview_scale = desired_width / preview_data_width;
        if (preview_scale <= 0.0f) preview_scale = 1.0f;
        const int preview_disp_width = std::max(
            1,
            static_cast<int>(std::round(preview_data_width * preview_scale)));
        const int preview_disp_height = std::max(
            1,
            static_cast<int>(std::round(preview_data_height * preview_scale)));
        draw_image(rgb_texture, rgb_image.data(), preview_cols, buffer_rows / 2,
                   preview_disp_width, preview_disp_height);
        draw_image(rgb_crop_texture, rgb_image_cropped.data(), crop_cols,
                   buffer_rows / 2, preview_disp_width, preview_disp_height);
        draw_image(hist_texture, histogram.data(), hist_w, hist_h, hist_w,
                   hist_h);
    }

   private:
    void generate_preview(const uint8_t* raw_buf) {
        std::array<int, 256> reds;
        std::array<int, 256> greens;
        std::array<int, 256> blues;
        std::fill(reds.begin(), reds.end(), 0);
        std::fill(greens.begin(), greens.end(), 0);
        std::fill(blues.begin(), blues.end(), 0);
        std::fill(histogram.begin(), histogram.end(), 0);

        for (int i = 0; i < buffer_rows / 2; i++) {
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

        const int crop_offset = (preview_cols - crop_cols) / 2;
        for (int i = 0; i < buffer_rows / 2; i++) {
            for (int j = 0; j < crop_cols; j++) {
                const int ind_cropped = (j + crop_cols * i) * 3;
                const int ind = (j + crop_offset + preview_cols * i) * 3;
                for (int k = 0; k < 3; k++) {
                    rgb_image_cropped[ind_cropped + k] = rgb_image[ind + k];
                }
            }
        }

        for (int i = 0; i < hist_h; i++) {
            for (int j = 0; j < hist_w; j++) {
                const int ind = (j + i * hist_w) * 3;
                histogram[ind + 0] = (i * 100 < reds[j / 2]) ? 255 : 0;
                histogram[ind + 1] = (i * 100 < greens[j / 2]) ? 255 : 0;
                histogram[ind + 2] = (i * 100 < blues[j / 2]) ? 255 : 0;
            }
        }
    }

    void draw_image(GLuint texture, const uint8_t* data, int w, int h,
                    int w_disp, int h_disp) {
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, data);
        ImGui::Image((intptr_t)texture, ImVec2(static_cast<float>(w_disp),
                                               static_cast<float>(h_disp)));
    }

    std::vector<uint8_t> rgb_image;
    std::vector<uint8_t> rgb_image_cropped;
    std::array<uint8_t, histogram_size> histogram{};
    GLuint rgb_texture = 0;
    GLuint rgb_crop_texture = 0;
    GLuint hist_texture = 0;
};

class ReviewController {
   public:
    ReviewController() { refresh_capture_dirs(); }

    bool mode_is_listing() const { return mode == Mode::Listing; }

    void update(std::chrono::steady_clock::time_point now) {
        if (mode == Mode::Listing && now - last_scan_time >= scan_interval) {
            refresh_capture_dirs();
            last_scan_time = now;
        }
    }

    void render_listing() {
        ImGui::Text("Captures in %s", nectar::k_capture_root);
        ImGui::Separator();
        ImVec2 child_size = ImGui::GetContentRegionAvail();
        if (ImGui::BeginChild("capture_list", child_size, true)) {
            if (capture_entries.empty()) {
                ImGui::Text("No captures found.");
            } else {
                for (size_t i = 0; i < capture_entries.size(); ++i) {
                    const auto& entry = capture_entries[i];
                    const std::string ts = format_time(entry.timestamp);
                    ImGui::PushID(static_cast<int>(i));
                    ImGui::TextUnformatted(entry.name.c_str());
                    ImGui::SameLine(250.0f);
                    ImGui::TextUnformatted(ts.c_str());
                    ImGui::SameLine();
                    if (ImGui::Button(
                            "View", ImVec2(120.0f, nectar::k_button_height))) {
                        open_capture(entry);
                    }
                    ImGui::Separator();
                    ImGui::PopID();
                }
            }
        }
        ImGui::EndChild();
    }

    void render_review() {
        if (ImGui::Button("Back", ImVec2(150.0f, nectar::k_button_height))) {
            mode = Mode::Listing;
            return;
        }
        ImGui::SameLine();
        ImGui::Text("Capture: %s", selected_capture_name.c_str());
        ImGui::Text("Directory: %s", selected_capture_path.c_str());
        ImGui::Separator();
        if (bin_files.empty()) {
            ImGui::Text("No .bin files found in this capture.");
            return;
        }
        ImGui::Text("File %d / %zu: %s", selected_bin_index + 1,
                    bin_files.size(),
                    bin_files[selected_bin_index].filename().c_str());
        if (bin_files.size() > 1) {
            int slider_value = selected_bin_index;
            if (nectar::draw_thick_slider_int(
                    "frame_index", &slider_value, 0,
                    static_cast<int>(bin_files.size() - 1))) {
                select_bin(slider_value);
            }
        }
        if (!last_error.empty()) {
            ImGui::TextColored(ImVec4(1, 0, 0, 1), "%s", last_error.c_str());
        }
        const float avail_width = ImGui::GetContentRegionAvail().x;
        previewer.draw_preview(avail_width);
        ImGui::Separator();
        ImGui::Text("Metadata:");
        if (capture_metadata.empty()) {
            ImGui::Text("<none>");
        } else {
            ImGui::TextUnformatted(capture_metadata.c_str());
        }
    }

   private:
    enum class Mode { Listing, Reviewing };

    void refresh_capture_dirs() {
        capture_entries.clear();
        std::error_code ec;
        if (!fs::exists(nectar::k_capture_root, ec)) {
            return;
        }
        fs::directory_iterator it(nectar::k_capture_root, ec);
        if (ec) {
            return;
        }
        fs::directory_iterator end;
        for (; it != end; it.increment(ec)) {
            if (ec) break;
            const fs::directory_entry& entry = *it;
            if (!entry.is_directory()) continue;
            CaptureEntry ce;
            ce.path = entry.path();
            ce.name = entry.path().filename().string();
            std::error_code ts_ec;
            ce.timestamp = entry.last_write_time(ts_ec);
            capture_entries.push_back(std::move(ce));
        }
        std::sort(capture_entries.begin(), capture_entries.end(),
                  [](const CaptureEntry& a, const CaptureEntry& b) {
                      if (a.timestamp == b.timestamp) {
                          return a.name > b.name;
                      }
                      return a.timestamp > b.timestamp;
                  });
    }

    static std::string format_time(fs::file_time_type tp) {
        using namespace std::chrono;
        const auto sctp =
            system_clock::now() + (tp - fs::file_time_type::clock::now());
        const std::time_t tt = system_clock::to_time_t(sctp);
        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &tt);
#else
        localtime_r(&tt, &tm_buf);
#endif
        char buffer[64];
        std::strftime(buffer, sizeof(buffer), "%F %T", &tm_buf);
        return buffer;
    }

    void open_capture(const CaptureEntry& entry) {
        selected_capture_path = entry.path.string();
        selected_capture_name = entry.name;
        capture_metadata = read_metadata(entry.path);
        load_bin_files(entry.path);
        if (!bin_files.empty()) {
            select_bin(0);
        } else {
            last_error = "No .bin files found";
        }
        mode = Mode::Reviewing;
    }

    void load_bin_files(const fs::path& path) {
        bin_files.clear();
        std::error_code ec;
        fs::directory_iterator it(path, ec);
        if (ec) return;
        fs::directory_iterator end;
        for (; it != end; it.increment(ec)) {
            if (ec) break;
            const fs::directory_entry& entry = *it;
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() == ".bin") {
                bin_files.push_back(entry.path());
            }
        }
        std::sort(bin_files.begin(), bin_files.end());
        selected_bin_index = bin_files.empty() ? -1 : 0;
    }

    std::string read_metadata(const fs::path& dir) {
        const fs::path meta_path = dir / "capture_settings.txt";
        std::ifstream fin(meta_path);
        if (!fin) {
            return {};
        }
        std::stringstream buffer;
        buffer << fin.rdbuf();
        return buffer.str();
    }

    void select_bin(int index) {
        if (index < 0 || index >= static_cast<int>(bin_files.size())) return;
        selected_bin_index = index;
        if (!previewer.load_file(bin_files[index], last_error)) {
            if (last_error.empty()) {
                last_error = "Failed to load bin file";
            }
        } else {
            last_error.clear();
        }
    }

    Mode mode = Mode::Listing;
    std::vector<CaptureEntry> capture_entries;
    std::chrono::steady_clock::time_point last_scan_time =
        std::chrono::steady_clock::now();
    const std::chrono::seconds scan_interval{1};

    std::vector<fs::path> bin_files;
    int selected_bin_index = -1;
    std::string selected_capture_path;
    std::string selected_capture_name;
    std::string capture_metadata;
    std::string last_error;

    CapturePreviewer previewer;
};

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    SdlGlGui sdl_gl_gui;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForOpenGL(sdl_gl_gui.window, sdl_gl_gui.gl_context);
    ImGui_ImplOpenGL3_Init(sdl_gl_gui.glsl_version.c_str());

    ReviewController controller;
    bool done = false;
    nectar::TouchHandler touch_handler;
    const ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);

    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            touch_handler.handle_event(event, sdl_gl_gui.window);
            if (!touch_handler.should_ignore_event(event)) {
                ImGui_ImplSDL2_ProcessEvent(&event);
            }
            if (event.type == SDL_QUIT) done = true;
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE &&
                event.window.windowID == SDL_GetWindowID(sdl_gl_gui.window)) {
                done = true;
            }
        }

        controller.update(std::chrono::steady_clock::now());

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::Begin("nectar review", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize);

        if (controller.mode_is_listing()) {
            controller.render_listing();
        } else {
            controller.render_review();
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

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    return 0;
}
