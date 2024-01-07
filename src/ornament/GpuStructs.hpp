#pragma once

#include "Scene.hpp"
#include "State.hpp"
#include <cstdint>
#include <glm/glm.hpp>

namespace ornament::gpu_structs {

typedef glm::vec4 Normal;
typedef glm::vec2 Uv;
typedef glm::mat4 Transform;

enum BvhNodeType : uint32_t {
    InternalNodeType = 0,
    SphereType = 1,
    MeshType = 2,
    TriangleType = 3,
};

struct BvhNode {
    glm::vec3 left_aabb_min_or_v0;
    uint32_t left_or_custom_id; // bvh node/top of mesh bvh/triangle id
    glm::vec3 left_aabb_max_or_v1;
    uint32_t right_or_material_index;
    glm::vec3 right_aabb_min_or_v2;
    BvhNodeType node_type;
    glm::vec3 right_aabb_max_or_v3;
    uint32_t transform_id;
};

struct Material {
    glm::vec3 albedo = { 1.0f, 0.0f, 1.0f };
    uint32_t albedo_texture_index = std::numeric_limits<uint32_t>::max();
    float fuzz;
    float ior;
    ornament::MaterialType type;
    uint32_t _padding = 0;

    Material(const ornament::Material& material)
    {
        fuzz = material.fuzz;
        ior = material.ior;
        type = material.type;

        switch (material.albedo.type) {
        case VectorType: {
            albedo = material.albedo.vector;
            break;
        }
        case TextureType: {
            albedo_texture_index = material.albedo.texture->textureId.value();
            break;
        }
        default: {
            break;
        }
        }
    }
};

struct Camera {
    glm::vec3 origin;
    float lens_radius;
    glm::vec3 lower_left_corner;
    uint32_t _padding0 = 0;
    glm::vec3 horizontal;
    uint32_t _padding1 = 0;
    glm::vec3 vertical;
    uint32_t _padding2 = 0;
    glm::vec3 u;
    uint32_t _padding3 = 0;
    glm::vec3 v;
    uint32_t _padding4 = 0;
    glm::vec3 w;
    uint32_t _padding5 = 0;

    Camera(const ornament::Camera& camera)
    {
        origin = camera.getLookFrom();
        lower_left_corner = camera.getLowerLeftCorner();
        horizontal = camera.getHorizontal();
        vertical = camera.getVertical();
        u = camera.getU();
        v = camera.getV();
        w = camera.getW();
        lens_radius = camera.getLensRadius();
    }
};

struct ConstantParams {
    Camera camera;
    uint32_t depth;
    uint32_t width;
    uint32_t height;
    uint32_t flip_y;
    float inverted_gamma;
    float ray_cast_epsilon;
    uint32_t textures_count;
    float current_iteration = 0.0f;

    ConstantParams(const ornament::Camera& cam, const ornament::State& state, uint32_t textures)
        : camera(cam)
    {
        depth = state.getDepth();
        width = state.getResolution().x;
        height = state.getResolution().y;
        flip_y = state.getFlipY() ? 1 : 0;
        inverted_gamma = state.getInvertedGamma();
        ray_cast_epsilon = state.getRayCastEpsilon();
        textures_count = textures;
        current_iteration = state.getCurrentIteration();
    }
};

}