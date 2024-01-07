#pragma once

#include "GpuStructs.hpp"
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
    const std::vector<gpu_structs::BvhNode>& getTlasNodes() const noexcept;
    const std::vector<gpu_structs::BvhNode>& getBlasNodes() const noexcept;
    const std::vector<gpu_structs::Normal>& getNormals() const noexcept;
    const std::vector<uint32_t>& getNormalIndices() const noexcept;
    const std::vector<gpu_structs::Uv>& getUvs() const noexcept;
    const std::vector<uint32_t>& getUvIndices() const noexcept;
    const std::vector<gpu_structs::Transform>& getTransforms() const noexcept;
    const std::vector<gpu_structs::Material>& getMaterials() const noexcept;
    const std::vector<ornament::Texture*>& getTextures() const noexcept;

private:
    // TLAS nodes count:
    // shapes = meshes + mesh_instances + spheres
    // nodes = shapes * 2 - 1
    // BLAS nodes count of one mesh:
    // nodes = triangles * 2 - 1
    std::vector<gpu_structs::BvhNode> m_tlasNodes;
    std::vector<gpu_structs::BvhNode> m_blasNodes;
    std::vector<gpu_structs::Normal> m_normals;
    std::vector<uint32_t> m_normalIndices;
    std::vector<gpu_structs::Uv> m_uvs;
    std::vector<uint32_t> m_uvIndices;
    std::vector<gpu_structs::Transform> m_transforms;
    std::vector<gpu_structs::Material> m_materials;
    std::vector<ornament::Texture*> m_textures;

    void build(const Scene& scene);
    void buildMeshBvhRecursive(Mesh& mesh);
    gpu_structs::BvhNode buildBvhTlasRecursive(std::vector<Leaf>& leafs, size_t start, size_t end);
    gpu_structs::BvhNode buildBvhBlasRecursive(std::vector<Triangle>& leafs, size_t start, size_t end);
    void appendTransform(const glm::mat4& transform);
    uint32_t getMaterialIndex(Material& m);
};

}
