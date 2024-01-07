#pragma once

#include <glm/glm.hpp>

namespace ornament::math {
struct Aabb {
public:
    Aabb() noexcept;
    Aabb(const glm::vec3& min, const glm::vec3& max) noexcept;
    glm::vec3 min() const noexcept;
    glm::vec3 max() const noexcept;
    void grow(const glm::vec3& p) noexcept;

private:
    glm::vec3 m_min;
    glm::vec3 m_max;
};

Aabb transform(const glm::mat4& m, const Aabb& aabb);
}