#pragma once

#include <string>

#include <imgui.h>

namespace nectar {

inline constexpr const char* k_capture_root = "/home/dllu/pictures/linescan";
inline constexpr float k_slider_track_height = 70.0f;
inline constexpr float k_slider_corner_radius = 20.0f;
inline constexpr float k_button_height = 30.0f;

bool draw_thick_slider_int(const char* label, int* value, int min_value,
                           int max_value);
bool draw_large_checkbox(const char* label, bool* value);

}  // namespace nectar
