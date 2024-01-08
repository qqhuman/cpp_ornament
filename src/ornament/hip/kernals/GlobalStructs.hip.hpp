#pragma once

#include <hip/hip_runtime.h>
#include "vec_math.hip.hpp"
#include "Array.hip.hpp"

namespace ornament::kernals {
template <typename T> 
struct Array
{
    T* ptr;
    uint32_t len;

    HOST_DEVICE INLINE T operator[](uint32_t index) const
    {
        return ptr[index];
    }
};

struct Camera
{
    float3 origin;
    float lensRadius;
    float3 lowerLeftCorner;
    uint32_t _padding0;
    float3 horizontal;
    uint32_t _padding1;
    float3 vertical;
    uint32_t _padding2;
    float3 u;
    uint32_t _padding3;
    float3 v;
    uint32_t _padding4;
    float3 w;
    uint32_t _padding5;
};

struct ConstantParams
{
    Camera camera;
    uint32_t depth;
    uint32_t width;
    uint32_t height;
    uint32_t flipY;
    float invertedGamma;
    float rayCastEpsilon;
    uint32_t texturesCount;
    float currentIteration;
};

enum BvhNodeType : uint32_t
{
    InternalNodeType = 0,
    SphereType = 1,
    MeshType = 2,
    TriangleType = 3,
};

#pragma pack(push, 1)
struct InternalNode 
{
    float3 leftAabbMin;
    uint32_t leftNodeId;
    float3 leftAabbMax;
    uint32_t rightNodeId;
    float3 rightAabbMin;
    uint32_t _padding;
    float3 rightAabbMax;
};

struct Sphere
{
    uint32_t materialId;
    uint32_t transformId;
};

struct Mesh
{
    uint32_t materialId;
    uint32_t transformId;
    uint32_t blasNodeId;
};

struct Triangle
{
    float3 v0;
    uint32_t triangleId;
    float3 v1;
    uint32_t _padding;
    float3 v2;
};
#pragma pack(pop)

struct BvhNode
{
    union {
        InternalNode internalNode;
        Sphere sphereNode;
        Mesh meshNode;
        Triangle triangleNode;
    };
    BvhNodeType type;
};

struct Bvh
{
    Array<BvhNode> tlasNodes;
    Array<BvhNode> blasNodes;
    Array<float4> normals;
    Array<uint32_t> normalIndices;
    Array<float2> uvs;
    Array<uint32_t> uvIndices;
    Array<float4x4> transforms;
};

enum MaterialType : uint32_t
{
    LambertianType = 0,
    MetalType = 1,
    DielectricType = 2,
    DiffuseLightType = 3,
};

#pragma pack(push, 1)
struct Lambertian
{
    float3 albedo;
    uint32_t albedoTextureId;
};

struct Metal
{
    float3 albedo;
    uint32_t albedoTextureId;
    float fuzz;
};

struct Dielectric
{
    float ior;
};

struct DiffuseLight
{
    float3 albedo;
    uint32_t albedoTextureId;
};
#pragma pack(pop)

struct Material 
{
    union {
        Lambertian lambertian;
        Metal metal;
        Dielectric dielectric;
        DiffuseLight diffuseLight;
    };
    MaterialType type;
    uint32_t _padding[2];
};

struct KernalBuffers
{
    Bvh bvh;
    Array<Material> materials;
    Array<hipTextureObject_t> textures;
    Array<float4> frameBuffer;
    Array<float4> accumulationBuffer;
    Array<uint32_t> rngSeedBuffer;
};
}