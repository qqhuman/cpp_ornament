#include "math.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

namespace ornament::math {

float lengthSq(const glm::vec3& v)
{
    return glm::dot(v, v);
}

bool approxEql(float val1, float val2)
{
    const float eps = 0.0000001f;
    if (val1 == val2) {
        return true;
    }

    if (std::isnan(val1) || std::isnan(val2)) {
        return false;
    }

    return std::abs(val1 - val2) <= eps;
}

glm::mat4 rotationBetweenVectors(const glm::vec3& a, const glm::vec3& b)
{
    float kCosTheta = glm::dot(a, b);
    if (approxEql(kCosTheta, 1.0f)) {
        return glm::mat4(1.0f);
    }

    float k = std::sqrtf(lengthSq(a) * lengthSq(b));
    if (approxEql(kCosTheta / k, -1.0f)) {
        glm::vec3 orthogonal = glm::cross(a, UnitX);
        if (approxEql(lengthSq(orthogonal), 0.0f)) {
            orthogonal = glm::cross(a, UnitY);
        }

        orthogonal = glm::normalize(orthogonal);
        return glm::toMat4(glm::quat(0.0f, orthogonal[0], orthogonal[1], orthogonal[2]));
    }

    glm::vec3 v3 = glm::cross(a, b);
    glm::vec4 v4 = glm::normalize(glm::vec4(v3[0], v3[1], v3[2], k + kCosTheta));
    return glm::toMat4(glm::quat(v4.w, v4[0], v4[1], v4[2]));
}

glm::vec2 getSphereTexCoord(const glm::vec3& p)
{
    float theta = std::acos(-p[1]);
    float phi = std::atan2(-p[2], p[0]) + glm::pi<float>();
    return { phi / (2.0f * glm::pi<float>()), theta / glm::pi<float>() };
}

}