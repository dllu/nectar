#pragma once

#include <chrono>
#include <memory>

namespace nectar {

class ReviewController {
   public:
    ReviewController();
    ~ReviewController();
    ReviewController(const ReviewController&) = delete;
    ReviewController& operator=(const ReviewController&) = delete;
    ReviewController(ReviewController&&) noexcept;
    ReviewController& operator=(ReviewController&&) noexcept;

    bool mode_is_listing() const;
    void update(std::chrono::steady_clock::time_point now);
    void render_listing();
    void render_review();

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nectar
