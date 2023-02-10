#include <INectaCamera.h>
#include <SDL.h>
#include <SDL_opengl.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl.h>

#include <cmath>
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

   private:
    int frame_id = 0;
    int rows = 192;
    int cols = 4096;

    const int hist_h = 256;
    const int hist_w = 512;
    const int histogram_size = hist_h * hist_w * 3;

    void printVideoModes(INectaCamera &cam) {
        auto acc = cam.GetAvailableVideoModes();
        for (size_t cc = 0; cc < acc.Size(); cc++) {
            std::cerr << (int)(acc[cc].GetVideoMode()) << std::endl;
        }
        std::cerr << "selected: " << (int)cam.GetVideoMode() << std::endl;
    }

    template <class T>
    std::tuple<std::vector<uint8_t>, std::vector<uint8_t>, std::vector<uint8_t>>
    viz(const T &raw_image) {
        std::vector<uint8_t> rgb_image((rows / 2) * (cols / 2) * 3, 0);
        std::vector<uint8_t> rgb_image_cropped((rows / 2) * (cols / 8) * 3, 0);

        std::array<int, 256> reds;
        std::array<int, 256> greens;
        std::array<int, 256> blues;
        std::fill(reds.begin(), reds.end(), 0);
        std::fill(greens.begin(), greens.end(), 0);
        std::fill(blues.begin(), blues.end(), 0);

        std::vector<uint8_t> histogram(histogram_size, 0);
        // make image
        for (int i = 0; i < rows / 2; i++) {
            for (int j = 0; j < cols / 2; j++) {
                int ind = (j + cols / 2 * i) * 3;
                rgb_image[ind] =
                    raw_image[2 * (2 * i * cols + (2 * j + 1)) + 1];
                rgb_image[ind + 1] =
                    (raw_image[2 * ((2 * i + 1) * cols + (2 * j + 1)) + 1] +
                     raw_image[2 * (2 * i * cols + 2 * j) + 1]) /
                    2;
                rgb_image[ind + 2] =
                    raw_image[2 * ((2 * i + 1) * cols + 2 * j) + 1];

                reds[rgb_image[ind]]++;
                greens[rgb_image[ind + 1]]++;
                blues[rgb_image[ind + 2]]++;
            }
        }
        // punched in
        for (int i = 0; i < rows / 2; i++) {
            for (int j = 0; j < cols / 8; j++) {
                // 2048
                // 512
                // 768 + 768 + 512 = 2048
                int ind_cropped = (j + cols / 8 * i) * 3;
                int ind = (j + (cols / 2 - cols - 8) / 2 + cols / 2 * i) * 3;
                for (int k = 0; k < 3; k++) {
                    rgb_image_cropped[ind_cropped + k] = rgb_image[ind + k];
                }
            }
        }
        // make a histogram
        for (int i = 0; i < hist_h; i++) {
            for (int j = 0; j < hist_w; j++) {
                int ind = (j + i * hist_w) * 3;
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
        return make_tuple(rgb_image, rgb_image_cropped, histogram);
    }

    void draw_image(const uint8_t *const image_data, const int w, const int h,
                    const int w_disp, const int h_disp) {
        GLuint image_texture;
        glGenTextures(1, &image_texture);
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
        ImGui::End();
    }

   public:
    void capture(INectaCamera &cam) {
        cam.SetGain(analog_gain);
        cam.SetCDSGain(cds_gain);
        cam.SetShutter(shutterspeed);

        // rows * shutter = 1000000 / 30 microseconds
        rows = std::ceil(1.0e6 / 30.0 / shutterspeed);
        rows = (rows / 16 + 1) * 16;
        cam.SetImageSizeX(cols);
        cam.SetImageSizeY(rows);

        // Get new frame
        const auto raw_image = cam.GetRawData();
        const auto [rgb_image, rgb_image_cropped, histogram] = viz(raw_image);
        ImGui::Begin("Capture preview");
        draw_image(rgb_image.data(), cols / 2, rows / 2, cols / 2, rows / 2);
        draw_image(rgb_image_cropped.data(), cols / 8, rows / 2, cols / 2,
                   rows / 2);
        draw_image(histogram.data(), hist_w, hist_h, hist_w, hist_h);
        if (save) {
            std::ofstream fout;
            std::stringstream ss;

            ss << "im" << std::setfill('0') << std::setw(4) << frame_id++
               << ".bin";
            fout.open(ss.str(), std::ios::out | std::ios::binary);
            fout.write((char *)raw_image.Data(), raw_image.Size());
            fout.close();
        }
    }
};

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
        window = SDL_CreateWindow(
            "Dear ImGui SDL2+OpenGL3 example", SDL_WINDOWPOS_CENTERED,
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
    bool capturing = false;
    bool done = false;

    NectarCapturer nc;
    while (!done) {
        io.DisplaySize = ImVec2(1920, 1080);
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
        if (ImGui::Button("Save")) {
            nc.save = !nc.save;
        }
        if (ImGui::Button("Quit")) {
            done = true;
        }
        ImGui::SliderInt("analog_gain", &nc.analog_gain, 0, 20);
        ImGui::SliderInt("cds_gain", &nc.cds_gain, 0, 1);
        ImGui::SliderInt("shutterspeed", &nc.shutterspeed, 0, 10000);
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
