#pragma once

#include <glm/glm.hpp>

namespace ornament::math {
glm::vec3 transformPoint(const glm::mat4& m, const glm::vec3& p);
glm::vec3 transformVector(const glm::mat4& m, const glm::vec3& p);
}