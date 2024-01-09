#pragma once

#include "hip/kernals/global_structs.hip.hpp"
#include "Scene.hpp"

namespace ornament {

enum LeafType {
    SphereType,
    MeshType,
    MeshInstanceType,
};

struct Leaf {
    LeafType type;
    union {
        Sphere* sphere;
        Mesh* mesh;
        MeshInstance* meshInstance;
    };
};

struct Triangle {
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
    uint32_t triangleIndex;
    math::Aabb aabb;
};

class Bvh {
public:
    Bvh(const Scene& scene);
    Bvh(const Bvh&) = delete;
    Bvh& operator=(const Bvh&) = delete;
    const std::vector<kernals::BvhNode>& getTlasNodes() const noexcept;
    const std::vector<kernals::BvhNode>& getBlasNodes() const noexcept;
    const std::vector<float4>& getNormals() const noexcept;
    const std::vector<uint32_t>& getNormalIndices() const noexcept;
    const std::vector<float2>& getUvs() const noexcept;
    const std::vector<uint32_t>& getUvIndices() const noexcept;
    const std::vector<float4x4>& getTransforms() const noexcept;
    const std::vector<kernals::Material>& getMaterials() const noexcept;
    const std::vector<ornament::Texture*>& getTextures() const noexcept;

private:
    // TLAS nodes count:
    // shapes = meshes + mesh_instances + spheres
    // nodes = shapes * 2 - 1
    // BLAS nodes count of one mesh:
    // nodes = triangles * 2 - 1
    std::vector<kernals::BvhNode> m_tlasNodes;
    std::vector<kernals::BvhNode> m_blasNodes;
    std::vector<float4> m_normals;
    std::vector<uint32_t> m_normalIndices;
    std::vector<float2> m_uvs;
    std::vector<uint32_t> m_uvIndices;
    std::vector<float4x4> m_transforms;
    std::vector<kernals::Material> m_materials;
    std::vector<ornament::Texture*> m_textures;

    void build(const Scene& scene);
    void buildMeshBvhRecursive(Mesh& mesh);
    kernals::BvhNode buildBvhTlasRecursive(std::vector<Leaf>& leafs, size_t start, size_t end);
    kernals::BvhNode buildBvhBlasRecursive(std::vector<Triangle>& leafs, size_t start, size_t end);
    void appendTransform(const glm::mat4& transform);
    uint32_t getMaterialIndex(Material& m);
};

}
