#include "transform.hpp"

namespace ornament::math {
glm::vec3 transformPoint(const glm::mat4& m, const glm::vec3& p)
{
    return glm::vec3(m * glm::vec4(p, 1.0));
}

glm::vec3 transformVector(const glm::mat4& m, const glm::vec3& v)
{
    return glm::vec3(m * glm::vec4(v, 0.0));
}
}