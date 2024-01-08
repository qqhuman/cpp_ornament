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

#pragma pack(push, 1)
struct InternalBvhNode 
{
    glm::vec3 leftAabbMin;
    uint32_t leftNodeId;
    glm::vec3 leftAabbMax;
    uint32_t rightNodeId;
    glm::vec3 rightAabbMin;
    uint32_t _padding;
    glm::vec3 rightAabbMax;
};

struct SphereBvhNode 
{
    uint32_t materialId;
    uint32_t transformId;
};

struct MeshBvhNode
{
    uint32_t materialId;
    uint32_t transformId;
    uint32_t blasNodeId;
};

struct TriangleBvhNode
{
    glm::vec3 v0;
    uint32_t triangleId;
    glm::vec3 v1;
    uint32_t _padding;
    glm::vec3 v2;
};
#pragma pack(pop)

struct BvhNode
{
    union {
        InternalBvhNode internalNode;
        SphereBvhNode sphereNode;
        MeshBvhNode meshNode;
        TriangleBvhNode triangleNode;
    };
    BvhNodeType type;
};

#pragma pack(push, 1)
struct Lambertian
{
    glm::vec3 albedo;
    uint32_t albedoTextureId;
};

struct Metal
{
    glm::vec3 albedo;
    uint32_t albedoTextureId;
    float fuzz;
};

struct Dielectric
{
    float ior;
};

struct DiffuseLight
{
    glm::vec3 albedo;
    uint32_t albedoTextureId;
};
#pragma pack(pop)

struct Material {
    union {
        Lambertian lambertian;
        Metal metal;
        Dielectric dielectric;
        DiffuseLight diffuseLight;
    };
    MaterialType type;
    uint32_t _padding[2];

    Material(const ornament::Material& material)
    {
        glm::vec3 albedo = {1.0f, 0.0f, 1.0f};
        uint32_t albedoTextureId = std::numeric_limits<uint32_t>::max();
        switch (material.albedo.type) {
        case VectorType: {
            albedo = material.albedo.vector;
            break;
        }
        case TextureType: {
            albedoTextureId = material.albedo.texture->textureId.value();
            break;
        }
        default: {
            break;
        }
        }

        type = material.type;
        switch (material.type)
        {
        case ornament::Lambertian: {
            lambertian.albedo = albedo;
            lambertian.albedoTextureId = albedoTextureId;
            break;
        }
        case ornament::Metal: {
            metal.albedo = albedo;
            metal.albedoTextureId = albedoTextureId;
            metal.fuzz = material.fuzz;
            break;
        }
        case ornament::Dielectric: {
            dielectric.ior = material.ior;
            break;
        }
        case ornament::DiffuseLight: {
            diffuseLight.albedo = albedo;
            diffuseLight.albedoTextureId = albedoTextureId;
            break;
        }
        default:
            break;
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