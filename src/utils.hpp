#pragma once

#include <SDL.h>
#include <imgui.h>

#include <string>

namespace nectar {

inline constexpr const char* k_capture_root = "/home/dllu/pictures/linescan";
inline constexpr float k_slider_track_height = 70.0f;
inline constexpr float k_slider_corner_radius = 20.0f;
inline constexpr float k_button_height = 30.0f;

bool draw_thick_slider_int(const char* label, int* value, int min_value,
                           int max_value);
bool draw_thick_slider_int_width(const char* label, int* value, int min_value,
                                 int max_value, float max_width);
bool draw_large_checkbox(const char* label, bool* value);
bool draw_large_checkbox_inline(const char* label, bool* value);
void configure_sdl_touch();

class TouchHandler {
   public:
    bool handle_event(const SDL_Event& event, SDL_Window* window);
    void apply_to_imgui(SDL_Window* window);
    bool should_ignore_event(const SDL_Event& event) const;
    void reset();

   private:
    bool active_ = false;
    SDL_FingerID finger_id_ = 0;
    float norm_x_ = 0.0f;
    float norm_y_ = 0.0f;
    bool have_pos_ = false;
    bool pending_down_ = false;
    bool pending_up_ = false;
    bool button_down_ = false;
};

}  // namespace nectar
