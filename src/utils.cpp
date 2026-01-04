#include "utils.hpp"

#include <algorithm>
#include <cmath>

namespace nectar {

bool draw_thick_slider_int(const char* label, int* value, int min_value,
                           int max_value) {
    const int range = max_value - min_value;
    if (range <= 0) {
        return false;
    }
    ImGui::Text("%s: %d", label, *value);
    ImGui::PushID(label);
    const float slider_height = k_slider_track_height;
    float slider_width = ImGui::GetContentRegionAvail().x;
    slider_width = std::max(slider_width, 100.0f);
    const float handle_width = std::max(50.0f, slider_width * 0.12f);
    const float slider_travel = std::max(slider_width - handle_width, 0.0f);
    const float normalized =
        std::clamp(static_cast<float>(*value - min_value) / range, 0.0f, 1.0f);
    ImVec2 slider_pos = ImGui::GetCursorScreenPos();
    const float handle_x =
        slider_pos.x + (slider_travel > 0 ? normalized * slider_travel : 0.0f);
    ImGui::InvisibleButton("##thick_slider",
                           ImVec2(slider_width, slider_height));
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImU32 track_color = ImGui::GetColorU32(ImGuiCol_FrameBg);
    const ImU32 border_color = ImGui::GetColorU32(ImGuiCol_Border);
    const ImU32 handle_color = ImGui::GetColorU32(ImGuiCol_SliderGrab);
    ImVec2 slider_end(slider_pos.x + slider_width,
                      slider_pos.y + slider_height);
    const float rounding =
        std::min(k_slider_corner_radius, slider_height * 0.5f);
    draw_list->AddRectFilled(slider_pos, slider_end, track_color, rounding);
    ImVec2 handle_min(handle_x, slider_pos.y);
    ImVec2 handle_max(handle_x + handle_width, slider_pos.y + slider_height);
    draw_list->AddRectFilled(handle_min, handle_max, handle_color, rounding);
    draw_list->AddRect(handle_min, handle_max, border_color, rounding);
    bool changed = false;
    const bool interacting =
        (ImGui::IsItemActive() && ImGui::GetIO().MouseDown[0]) ||
        (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0));
    if (interacting) {
        const float mouse_x = ImGui::GetIO().MousePos.x;
        float new_pos = mouse_x - slider_pos.x - handle_width * 0.5f;
        new_pos = std::clamp(new_pos, 0.0f, slider_travel);
        const float denom = slider_travel > 0 ? slider_travel : 1.0f;
        const float new_norm = new_pos / denom;
        const int new_value =
            min_value + static_cast<int>(std::round(new_norm * range));
        if (new_value != *value) {
            *value = new_value;
            changed = true;
        }
    }
    ImGui::Dummy(ImVec2(0.0f, 10.0f));
    ImGui::PopID();
    return changed;
}

bool draw_large_checkbox(const char* label, bool* value) {
    ImGui::Text("%s: %s", label, *value ? "on" : "off");
    ImGui::PushID(label);
    const ImVec2 box_size(k_slider_track_height, k_slider_track_height);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##large_checkbox", box_size);
    const bool clicked = ImGui::IsItemClicked();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const float rounding =
        std::min(k_slider_corner_radius, k_slider_track_height * 0.5f);
    const ImU32 bg_color = ImGui::GetColorU32(ImGuiCol_FrameBg);
    const ImU32 active_color = ImGui::GetColorU32(ImGuiCol_SliderGrabActive);
    const ImU32 border_color = ImGui::GetColorU32(ImGuiCol_Border);
    const ImU32 check_color = ImGui::GetColorU32(ImGuiCol_CheckMark);
    const ImVec2 box_end(pos.x + box_size.x, pos.y + box_size.y);
    draw_list->AddRectFilled(pos, box_end, *value ? active_color : bg_color,
                             rounding);
    draw_list->AddRect(pos, box_end, border_color, rounding);
    if (*value) {
        const ImVec2 start(pos.x + box_size.x * 0.25f,
                           pos.y + box_size.y * 0.55f);
        const ImVec2 mid(pos.x + box_size.x * 0.45f,
                         pos.y + box_size.y * 0.75f);
        const ImVec2 end(pos.x + box_size.x * 0.75f,
                         pos.y + box_size.y * 0.3f);
        draw_list->AddLine(start, mid, check_color, 5.0f);
        draw_list->AddLine(mid, end, check_color, 5.0f);
    }
    bool changed = false;
    if (clicked) {
        *value = !*value;
        changed = true;
    }
    ImGui::Dummy(ImVec2(0.0f, 10.0f));
    ImGui::PopID();
    return changed;
}

}  // namespace nectar
