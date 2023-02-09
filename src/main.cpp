#include <INectaCamera.h>
#include <SDL.h>
#include <SDL_opengl.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl.h>

#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <sstream>

using namespace CAlkUSB3;

constexpr int rows = 192;
constexpr int cols = 4096;
constexpr int imsize = rows * (cols / 2) * 3;

/*
void printColorCodings(INectaCamera &cam) {
    auto acc = cam.GetAvailableColorCoding();
    for (auto cc = acc.Front(); 1; cc++) {
        std::cerr << cc->ToString() << std::endl;
        if (cc == acc.Back()) {
            break;
        }
    }
}
*/

void printVideoModes(INectaCamera &cam) {
    auto acc = cam.GetAvailableVideoModes();
    for (size_t cc = 0; cc < acc.Size(); cc++) {
        std::cerr << (int)(acc[cc].GetVideoMode()) << std::endl;
    }
    std::cerr << "selected: " << (int)cam.GetVideoMode() << std::endl;
}

template <class T>
std::array<uint8_t, imsize> viz(const T &raw_image) {
    std::array<uint8_t, imsize> rgb_image;
    std::array<int, 256> reds;
    std::array<int, 256> greens;
    std::array<int, 256> blues;
    std::fill(rgb_image.begin(), rgb_image.end(), 0);
    std::fill(reds.begin(), reds.end(), 0);
    std::fill(greens.begin(), greens.end(), 0);
    std::fill(blues.begin(), blues.end(), 0);

    // make image
    for (int i = 0; i < rows / 2; i++) {
        for (int j = 0; j < cols / 2; j++) {
            int ind = (j + cols / 2 * i) * 3;
            rgb_image[ind] = raw_image[2 * (2 * i * cols + (2 * j + 1)) + 1];
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
    // make a histogram
    for (int i = 0; i < rows / 2; i++) {
        for (int j = 0; j < 256; j++) {
            int ind = (j + cols / 2 * (rows - i - 1)) * 3;
            if (i * 100 < reds[j]) {
                rgb_image[ind + 0] = 255;
            } else {
                rgb_image[ind + 0] = 0;
            }
            if (i * 100 < greens[j]) {
                rgb_image[ind + 1] = 255;
            } else {
                rgb_image[ind + 1] = 0;
            }
            if (i * 100 < blues[j]) {
                rgb_image[ind + 2] = 255;
            } else {
                rgb_image[ind + 2] = 0;
            }
        }
    }
    return rgb_image;
}

void draw_image(const uint8_t *const image_data) {
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cols / 2, rows, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, image_data);

    ImGui::Begin("Capture preview");
    ImGui::Text("size = %d x %d", cols / 2, rows);
    ImGui::Image((void *)(intptr_t)image_texture, ImVec2(cols / 2, rows));
    ImGui::End();
}

void capture(INectaCamera &cam, int *gain, int *cds_gain, int *shutter) {
    cam.SetGain(*gain);
    cam.SetCDSGain(*cds_gain);
    cam.SetShutter(*shutter);
    // Get new frame
    const auto raw_image = cam.GetRawData();
    const auto rgb_image = viz(raw_image);
    draw_image(rgb_image.data());
}

INectaCamera &get_camera() {
    INectaCamera &cam = INectaCamera::Create();
    // Get list of connected cameras
    Array<String> camList = cam.GetCameraList();
    if (camList.Size() == 0) {
        std::cerr << "No AlkUSB3 camera found" << std::endl;
        exit(1);
    }

    cam.SetCamera(0);
    cam.Init();

    cam.SetADCResolution(12);
    cam.SetVideoMode(1);
    cam.SetColorCoding(ColorCoding::Raw16);
    cam.SetImageSizeX(cols);
    cam.SetImageSizeY(rows);

    cam.SetPacketSize(std::min(16384U, cam.GetMaxPacketSize()));
    return cam;
}

int main(int argc, char *argv[]) {
    // Setup SDL
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) !=
        0) {
        printf("Error: %s\n", SDL_GetError());
        return -1;
    }

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char *glsl_version = "#version 100";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#elif defined(__APPLE__)
    // GL 3.2 Core + GLSL 150
    const char *glsl_version = "#version 150";
    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_FLAGS,
        SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);  // Always required on Mac
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
    // GL 3.0 + GLSL 130
    const char *glsl_version = "#version 130";
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
    SDL_Window *window = SDL_CreateWindow(
        "Dear ImGui SDL2+OpenGL3 example", SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1);  // Enable vsync
                                //
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);

    int analog_gain = 10;
    int cds_gain = 1;
    int shutterspeed = 2000;

    // Build atlas
    unsigned char *tex_pixels = NULL;
    int tex_w, tex_h;
    io.Fonts->GetTexDataAsRGBA32(&tex_pixels, &tex_w, &tex_h);
    ImVec4 clear_color = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    INectaCamera &cam = get_camera();
    bool capturing = false;
    bool done = false;
    while (!done) {
        io.DisplaySize = ImVec2(1920, 1080);
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) done = true;
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE &&
                event.window.windowID == SDL_GetWindowID(window))
                done = true;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        if (ImGui::Button("Capture")) {
            capturing = !capturing;
            cam.SetAcquire(capturing);
        }
        if (ImGui::Button("Save")) {
        }
        if (ImGui::Button("Quit")) {
            done = true;
        }
        ImGui::SliderInt("analog_gain", &analog_gain, 0, 20);
        ImGui::SliderInt("cds_gain", &cds_gain, 0, 1);
        ImGui::SliderInt("shutterspeed", &shutterspeed, 0, 10000);
        if (capturing) {
            try {
                capture(cam, &analog_gain, &cds_gain, &shutterspeed);
            } catch (const Exception &ex) {
                std::cerr << "Exception " << ex.Name() << " occurred"
                          << std::endl
                          << ex.Message() << std::endl;
                ImGui::Text("Exception %s %s", ex.Name(), ex.Message());
            } catch (...) {
                std::cerr << "Unhandled exception" << std::endl;
            }
        }
        ImGui::Render();
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(clear_color.x * clear_color.w,
                     clear_color.y * clear_color.w,
                     clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
