#include "utils.hpp"

#include <algorithm>
#include <cmath>

namespace nectar {

void configure_sdl_touch() {
    SDL_SetHint(SDL_HINT_TOUCH_MOUSE_EVENTS, "0");
    SDL_SetHint(SDL_HINT_MOUSE_TOUCH_EVENTS, "0");
}

bool TouchHandler::handle_event(const SDL_Event& event, SDL_Window* window) {
    if (event.type != SDL_FINGERDOWN && event.type != SDL_FINGERMOTION &&
        event.type != SDL_FINGERUP) {
        return false;
    }
    const SDL_FingerID finger_id = event.tfinger.fingerId;
    if (event.type == SDL_FINGERDOWN) {
        if (active_) {
            return false;
        }
        active_ = true;
        finger_id_ = finger_id;
        pending_down_ = true;
    } else if (!active_ || finger_id != finger_id_) {
        return false;
    }

    norm_x_ = event.tfinger.x;
    norm_y_ = event.tfinger.y;
    have_pos_ = true;

    if (event.type == SDL_FINGERUP) {
        pending_up_ = true;
        active_ = false;
    }
    return true;
}

void TouchHandler::apply_to_imgui(SDL_Window* window) {
    if (window == nullptr) {
        return;
    }
    if (!have_pos_ && !pending_down_ && !pending_up_ && !active_ &&
        !button_down_) {
        return;
    }
    int window_w = 0;
    int window_h = 0;
    SDL_GetWindowSize(window, &window_w, &window_h);
    if (window_w <= 0 || window_h <= 0) {
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    io.AddMouseSourceEvent(ImGuiMouseSource_TouchScreen);
    if (have_pos_ &&
        (active_ || button_down_ || pending_down_ || pending_up_)) {
        const float pos_x = norm_x_ * window_w;
        const float pos_y = norm_y_ * window_h;
        io.AddMousePosEvent(pos_x, pos_y);
    }
    if (pending_down_) {
        io.AddMouseButtonEvent(0, true);
        button_down_ = true;
        pending_down_ = false;
    }
    if (pending_up_) {
        io.AddMouseButtonEvent(0, false);
        button_down_ = false;
        pending_up_ = false;
    }
    if (!active_ && !button_down_) {
        have_pos_ = false;
    }
}

void TouchHandler::reset() {
    active_ = false;
    finger_id_ = 0;
    norm_x_ = 0.0f;
    norm_y_ = 0.0f;
    have_pos_ = false;
    pending_down_ = false;
    pending_up_ = false;
    button_down_ = false;
}

bool TouchHandler::should_ignore_event(const SDL_Event& event) const {
    if (event.type == SDL_MOUSEBUTTONDOWN || event.type == SDL_MOUSEBUTTONUP) {
        return event.button.which == SDL_TOUCH_MOUSEID;
    }
    if (event.type == SDL_MOUSEMOTION) {
        return event.motion.which == SDL_TOUCH_MOUSEID;
    }
    if (event.type == SDL_MOUSEWHEEL) {
        return event.wheel.which == SDL_TOUCH_MOUSEID;
    }
    return false;
}

static bool draw_thick_slider_int_impl(const char* label, int* value,
                                       int min_value, int max_value,
                                       float max_width) {
    const int range = max_value - min_value;
    if (range <= 0) {
        return false;
    }
    ImGui::Text("%s: %d", label, *value);
    ImGui::PushID(label);
    const float slider_height = k_slider_track_height;
    float slider_width = ImGui::GetContentRegionAvail().x;
    if (max_width > 0.0f) {
        slider_width = std::min(slider_width, max_width);
    }
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

bool draw_thick_slider_int(const char* label, int* value, int min_value,
                           int max_value) {
    return draw_thick_slider_int_impl(label, value, min_value, max_value, -1.0f);
}

bool draw_thick_slider_int_width(const char* label, int* value, int min_value,
                                 int max_value, float max_width) {
    return draw_thick_slider_int_impl(label, value, min_value, max_value,
                                      max_width);
}

static bool draw_large_checkbox_impl(const char* label, bool* value,
                                     bool add_spacing) {
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
        const ImVec2 end(pos.x + box_size.x * 0.75f, pos.y + box_size.y * 0.3f);
        draw_list->AddLine(start, mid, check_color, 5.0f);
        draw_list->AddLine(mid, end, check_color, 5.0f);
    }
    bool changed = false;
    if (clicked) {
        *value = !*value;
        changed = true;
    }
    if (add_spacing) {
        ImGui::Dummy(ImVec2(0.0f, 10.0f));
    }
    ImGui::PopID();
    return changed;
}

bool draw_large_checkbox(const char* label, bool* value) {
    return draw_large_checkbox_impl(label, value, true);
}

bool draw_large_checkbox_inline(const char* label, bool* value) {
    return draw_large_checkbox_impl(label, value, false);
}

}  // namespace nectar
