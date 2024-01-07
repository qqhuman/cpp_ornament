#pragma once

#include "Aabb.hpp"
#include "transform.hpp"
#include <glm/glm.hpp>

namespace ornament::math {

const glm::vec3 UnitX(1.0f, 0.0f, 0.0f);
const glm::vec3 UnitY(0.0f, 1.0f, 0.0f);
const glm::vec3 UnitZ(0.0f, 0.0f, 1.0f);

float lengthSq(const glm::vec3& v);
bool approxEql(float val1, float val2);
glm::mat4 rotationBetweenVectors(const glm::vec3& a, const glm::vec3& b);
glm::vec2 getSphereTexCoord(const glm::vec3& p);
}