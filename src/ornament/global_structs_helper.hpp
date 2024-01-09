#pragma once

#include "Scene.hpp"
#include "State.hpp"
#include <cstdint>
#include <glm/glm.hpp>
#include "hip/kernals/global_structs.hip.hpp"

namespace ornament::kernals {

float3 glmToHipFloat3(const glm::vec3& v);
Material toKernalMaterial(const ornament::Material& material);
Camera toKernalCamera(const ornament::Camera& camera);
ConstantParams toKernalConstantParams(const ornament::Camera& camera, const ornament::State& state, uint32_t textures);

}