#include <INectaCamera.h>
#include <SDL.h>
#include <SDL_opengl.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <sstream>

using namespace CAlkUSB3;

class NectarCapturer {
   public:
    bool save = false;
    int analog_gain = 10;
    int cds_gain = 1;
    int shutterspeed = 2000;
    int frame_id = 0;

    NectarCapturer()
        : buffer(buffer_rows * cols * 2, 0),
          rgb_image((buffer_rows / 2) * (cols / 2) * 3, 0),
          rgb_image_cropped((buffer_rows / 2) * (cols / 8) * 3, 0) {
        glGenTextures(1, &rgb_texture);
        glGenTextures(1, &rgb_crop_texture);
        glGenTextures(1, &hist_texture);
    }

   private:
    std::chrono::time_point<std::chrono::steady_clock> last_frame_time;
    int capture_rows = 192;
    static constexpr int buffer_rows = 512;
    static constexpr int cols = 4096;
    std::vector<uint8_t> buffer;
    std::vector<uint8_t> rgb_image;
    std::vector<uint8_t> rgb_image_cropped;
    int buffer_row_id = 0;

    static constexpr int hist_h = 256;
    static constexpr int hist_w = 512;
    static constexpr int histogram_size = hist_h * hist_w * 3;

    GLuint rgb_texture;
    GLuint rgb_crop_texture;
    GLuint hist_texture;

    std::array<uint8_t, histogram_size> histogram;
    void viz() {
        std::array<int, 256> reds;
        std::array<int, 256> greens;
        std::array<int, 256> blues;
        std::fill(reds.begin(), reds.end(), 0);
        std::fill(greens.begin(), greens.end(), 0);
        std::fill(blues.begin(), blues.end(), 0);

        std::fill(histogram.begin(), histogram.end(), 0);
        // make image
        for (int i = buffer_row_id / 2; i < (buffer_row_id + capture_rows) / 2;
             i++) {
            for (int j = 0; j < cols / 2; j++) {
                const int ind = (j + cols / 2 * i) * 3;
                rgb_image[ind] = buffer[2 * ((2 * i + 1) * cols + 2 * j) + 1];
                rgb_image[ind + 1] =
                    (buffer[2 * ((2 * i + 1) * cols + (2 * j + 1)) + 1] +
                     buffer[2 * (2 * i * cols + 2 * j) + 1]) /
                    2;
                rgb_image[ind + 2] =
                    buffer[2 * (2 * i * cols + (2 * j + 1)) + 1];

                reds[rgb_image[ind]]++;
                greens[rgb_image[ind + 1]]++;
                blues[rgb_image[ind + 2]]++;
            }
        }
        // punched in
        for (int i = buffer_row_id / 2; i < (buffer_row_id + capture_rows) / 2;
             i++) {
            for (int j = 0; j < cols / 8; j++) {
                const int ind_cropped = (j + cols / 8 * i) * 3;
                const int ind =
                    (j + (cols / 2 - cols / 8) / 2 + cols / 2 * i) * 3;
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

    void draw_image(const GLuint image_texture, const uint8_t *const image_data,
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

        ImGui::Image((void *)(intptr_t)image_texture, ImVec2(w_disp, h_disp));
    }

   public:
    void capture(INectaCamera &cam) {
        cam.SetGain(analog_gain);
        cam.SetCDSGain(cds_gain);
        cam.SetShutter(shutterspeed);

        // rows * shutter = 1000000 / 30 microseconds
        const int new_rows_desired = std::ceil(1.0e6 / 30.0 / shutterspeed);

        int new_rows = 1;
        while (new_rows <= new_rows_desired) {
            new_rows *= 2;
        }
        // whyy
        new_rows *= 16;
        const int new_capture_rows = std::min(buffer_rows, new_rows);

        if (capture_rows != new_capture_rows) {
            cam.SetAcquire(false);
            cam.SetImageSizeX(cols);
            cam.SetImageSizeY(new_capture_rows);
            cam.SetAcquire(true);
            buffer_row_id = 0;
            capture_rows = new_capture_rows;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int capture_frames = std::max(1, new_rows / buffer_rows);
        for (int capture_ind = 0; capture_ind < capture_frames; capture_ind++) {
            // Get new frame
            const auto raw_image = cam.GetRawData();

            for (int i = 0; i < capture_rows; i++) {
                for (int j = 0; j < cols; j++) {
                    for (int k = 0; k < 2; k++) {
                        buffer[2 * ((i + buffer_row_id) * cols + j) + k] =
                            raw_image[2 * (i * cols + j) + k];
                    }
                }
            }
            viz();
            buffer_row_id = (buffer_row_id + capture_rows) % buffer_rows;

            if (save && buffer_row_id == 0) {
                std::ofstream fout;
                std::stringstream ss;

                ss << "im" << std::setfill('0') << std::setw(6) << frame_id++
                   << ".bin";
                fout.open(ss.str(), std::ios::out | std::ios::binary);
                fout.write((char *)buffer.data(), buffer.size());
                fout.close();
            }
        }

        const auto t1 = std::chrono::steady_clock::now();
        ImGui::Text("rows = %d", capture_rows);
        draw_image(rgb_texture, rgb_image.data(), cols / 2, buffer_rows / 2,
                   cols / 2, buffer_rows / 2);
        draw_image(rgb_crop_texture, rgb_image_cropped.data(), cols / 8,
                   buffer_rows / 2, cols / 2, buffer_rows / 2);
        draw_image(hist_texture, histogram.data(), hist_w, hist_h, hist_w,
                   hist_h);
        const auto t2 = std::chrono::steady_clock::now();
        ImGui::Text("capture takes %ld ns", (t1 - t0).count());
        ImGui::Text("drawing takes %ld ns", (t2 - t1).count());
        ImGui::Text("fps = %lf", 1e9 / (t2 - last_frame_time).count());
        last_frame_time = t2;
    }
};

void print_video_modes(INectaCamera &cam) {
    auto acc = cam.GetAvailableVideoModes();
    for (size_t cc = 0; cc < acc.Size(); cc++) {
        std::cerr << (int)(acc[cc].GetVideoMode()) << std::endl;
    }
    std::cerr << "selected: " << (int)cam.GetVideoMode() << std::endl;
}

INectaCamera &get_camera() {
    INectaCamera &cam = INectaCamera::Create();
    // Get list of connected cameras
    Array<String> camList = cam.GetCameraList();
    if (camList.Size() == 0) {
        std::cerr << "No AlkUSB3 camera found" << std::endl;
        std::exit(1);
    }

    cam.SetCamera(0);
    cam.Init();

    cam.SetADCResolution(12);
    cam.SetVideoMode(1);
    cam.SetColorCoding(ColorCoding::Raw16);

    cam.SetPacketSize(std::min(16384U, cam.GetMaxPacketSize()));
    cam.SetAcquire(true);
    return cam;
}

struct SdlGlGui {
    std::string glsl_version;
    SDL_GLContext gl_context;
    SDL_Window *window;
    SdlGlGui() {
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

int main(int argc, char *argv[]) {
    SdlGlGui sdl_gl_gui;
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();

    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForOpenGL(sdl_gl_gui.window, sdl_gl_gui.gl_context);
    ImGui_ImplOpenGL3_Init(sdl_gl_gui.glsl_version.c_str());

    // Build atlas
    unsigned char *tex_pixels = NULL;
    int tex_w, tex_h;
    io.Fonts->GetTexDataAsRGBA32(&tex_pixels, &tex_w, &tex_h);
    ImVec4 clear_color = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    INectaCamera &cam = get_camera();
    bool done = false;

    NectarCapturer nc;
    std::string output_dir;

    while (!done) {
        io.DisplaySize = ImVec2(2560, 1440);
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) done = true;
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE &&
                event.window.windowID == SDL_GetWindowID(sdl_gl_gui.window))
                done = true;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        if (ImGui::Button("Quit")) {
            done = true;
        }
        if (nc.save) {
            if (ImGui::Button("Stop saving")) {
                nc.save = false;
            }
        } else {
            if (ImGui::Button("Start saving")) {
                nc.save = true;
                nc.frame_id = 0;
                std::stringstream ss;
                const auto now = std::chrono::system_clock::now();
                const time_t itt = std::chrono::system_clock::to_time_t(now);
                const auto gt = std::gmtime(&itt);
                ss << "/home/dllu/pictures/linescan/"
                   << std::put_time(gt, "%F-%T");
                output_dir = ss.str();
                std::filesystem::create_directory(output_dir);
                std::filesystem::current_path(output_dir);
            }
        }
        if (!nc.save) {
            ImGui::SliderInt("analog_gain", &nc.analog_gain, 0, 20);
            ImGui::SliderInt("cds_gain", &nc.cds_gain, 0, 1);
            ImGui::SliderInt("shutterspeed", &nc.shutterspeed, 0, 10000);
        } else {
            ImGui::Text("Saving to: %s; saved %d", output_dir.c_str(), nc.frame_id);
        }
        try {
            nc.capture(cam);
        } catch (const Exception &ex) {
            std::cerr << "Exception " << ex.Name() << " occurred" << std::endl
                      << ex.Message() << std::endl;
            ImGui::Text("Exception %s %s", ex.Name(), ex.Message());
        } catch (...) {
            std::cerr << "Unhandled exception" << std::endl;
        }
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
