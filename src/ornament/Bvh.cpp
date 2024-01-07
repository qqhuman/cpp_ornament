#include "Bvh.hpp"
#include <random>
#include <stdexcept>

namespace ornament {

std::mt19937 gen32x;

float randomi32(int min, int max)
{
    return std::uniform_int<int>(min, max)(gen32x);
}

math::Aabb getAabb(const Leaf& l)
{
    math::Aabb aabb;
    switch (l.type) {
    case SphereType: {
        return l.sphere->aabb;
    }
    case MeshType: {
        return l.mesh->aabb;
    }
    case MeshInstanceType: {
        return l.meshInstance->aabb;
    }
    default: {
        throw std::runtime_error("[ornament] not implemented switch case.");
    }
    }
}

math::Aabb calculateBoundingBox(const std::vector<Triangle>& leafs, size_t start, size_t end)
{
    glm::vec3 min(std::numeric_limits<float>::infinity());
    glm::vec3 max(-std::numeric_limits<float>::infinity());

    for (auto l = leafs.begin() + start; l != leafs.begin() + end; ++l) {
        min = glm::min(min, l->aabb.min());
        max = glm::max(max, l->aabb.max());
    }

    return math::Aabb(min, max);
}

math::Aabb calculateBoundingBox(const std::vector<Leaf>& leafs, size_t start, size_t end)
{
    glm::vec3 min(std::numeric_limits<float>::infinity());
    glm::vec3 max(-std::numeric_limits<float>::infinity());

    for (auto l = leafs.begin() + start; l != leafs.begin() + end; ++l) {
        math::Aabb aabb = getAabb(*l);
        min = glm::min(min, aabb.min());
        max = glm::max(max, aabb.max());
    }

    return math::Aabb(min, max);
}

Bvh::Bvh(const Scene& scene)
{
    size_t shapesCount = scene.getAttachedSpheres().size() + scene.getAttachedMeshes().size() + scene.getAttachedMeshInstances().size();
    if (shapesCount == 0) {
        throw std::runtime_error("[ornament] scene cannot be empty.");
    }

    size_t tlasNodesCount = shapesCount * 2 - 1;
    size_t blasNodesCount = 0;
    size_t normalsCount = 0;
    size_t normalIndicesCount = 0;
    size_t uvsCount = 0;
    size_t uvIndicesCount = 0;
    for (auto& m : scene.getAttachedMeshes()) {
        size_t triangles = m.get()->vertexIndices.size() / 3;
        blasNodesCount += triangles * 2 - 1;
        normalsCount += m.get()->normals.size();
        normalIndicesCount += m.get()->normalIndices.size();
        uvsCount += m.get()->uvs.size();
        uvIndicesCount += m.get()->uvIndices.size();
    }

    m_tlasNodes.reserve(tlasNodesCount);
    m_blasNodes.reserve(blasNodesCount);
    m_normals.reserve(normalsCount);
    m_normalIndices.reserve(normalIndicesCount);
    m_uvs.reserve(uvsCount);
    m_uvIndices.reserve(uvIndicesCount);
    m_transforms.reserve(shapesCount);
    m_materials.reserve(scene.getMaterials().size());
    m_textures.reserve(scene.getTextures().size());

    build(scene);

    if (tlasNodesCount != m_tlasNodes.size()) {
        throw std::runtime_error("[ornament] expected tlas noodes count is not equal to actual tlas noodes count.");
    }
    if (blasNodesCount != m_blasNodes.size()) {
        throw std::runtime_error("[ornament] expected blas noodes count is not equal to actual blas noodes count.");
    }
}

void Bvh::build(const Scene& scene)
{
    std::vector<Leaf> leafs;
    leafs.reserve(scene.getAttachedSpheres().size() + scene.getAttachedMeshes().size() + scene.getAttachedMeshInstances().size());

    for (auto& s : scene.getAttachedSpheres()) {
        leafs.push_back({ .type = SphereType, .sphere = s.get() });
    }

    for (auto& mi : scene.getAttachedMeshInstances()) {
        leafs.push_back({ .type = MeshInstanceType, .meshInstance = mi.get() });
        if (!mi.get()->mesh.get()->bvhId.has_value()) {
            buildMeshBvhRecursive(*mi.get()->mesh);
        }
    }

    for (auto& m : scene.getAttachedMeshes()) {
        leafs.push_back({ .type = MeshType, .mesh = m.get() });
        if (!m.get()->bvhId.has_value()) {
            buildMeshBvhRecursive(*m);
        }
    }

    gpu_structs::BvhNode root = buildBvhTlasRecursive(leafs, 0, leafs.size());
    m_tlasNodes.push_back(root);
}

void Bvh::buildMeshBvhRecursive(Mesh& mesh)
{
    size_t trianglesCount = mesh.vertexIndices.size() / 3;
    std::vector<Triangle> leafs;
    leafs.reserve(trianglesCount);

    for (size_t meshTriangleIndex = 0; meshTriangleIndex < trianglesCount; meshTriangleIndex++) {
        auto v0 = mesh.vertices[mesh.vertexIndices[meshTriangleIndex * 3]];
        auto v1 = mesh.vertices[mesh.vertexIndices[meshTriangleIndex * 3 + 1]];
        auto v2 = mesh.vertices[mesh.vertexIndices[meshTriangleIndex * 3 + 2]];
        math::Aabb aabb(glm::min(glm::min(v0, v1), v2), glm::max(glm::max(v0, v1), v2));

        size_t globalTriangleIndex = m_normalIndices.size() / 3 + meshTriangleIndex;
        leafs.push_back({
            .v0 = v0,
            .v1 = v1,
            .v2 = v2,
            .triangleIndex = (uint32_t)globalTriangleIndex,
            .aabb = aabb,
        });
    }

    for (uint32_t ni : mesh.normalIndices) {
        m_normalIndices.push_back(ni + m_normals.size());
    }

    for (glm::vec3 n : mesh.normals) {
        m_normals.push_back(glm::vec4(n, 0.0f));
    }

    for (uint32_t uvi : mesh.uvIndices) {
        m_uvIndices.push_back(uvi + m_uvs.size());
    }

    m_uvs.insert(
        m_uvs.end(),
        mesh.uvs.begin(),
        mesh.uvs.end());

    gpu_structs::BvhNode root = buildBvhBlasRecursive(leafs, 0, leafs.size());
    m_blasNodes.push_back(root);
    mesh.bvhId = (uint32_t)(m_blasNodes.size() - 1);
}

gpu_structs::BvhNode Bvh::buildBvhBlasRecursive(std::vector<Triangle>& leafs, size_t start, size_t end)
{
    size_t leafsSize = end - start;
    if (leafsSize == 0) {
        throw std::runtime_error("[ornament] mesh cannot be empty.");
    } else if (leafsSize == 1) {
        Triangle t = leafs[start];
        gpu_structs::BvhNode node;
        node.left_aabb_min_or_v0 = t.v0;
        node.left_aabb_max_or_v1 = t.v1;
        node.right_aabb_min_or_v2 = t.v2;
        node.left_or_custom_id = t.triangleIndex;
        node.node_type = gpu_structs::TriangleType;
        return node;
    } else {
        int axis = randomi32(0, 2);
        std::sort(
            leafs.begin() + start,
            leafs.begin() + end,
            [axis](const Triangle& a, const Triangle& b) {
                return a.aabb.min()[axis] < b.aabb.min()[axis];
            });

        size_t mid = start + leafsSize / 2;
        gpu_structs::BvhNode left = buildBvhBlasRecursive(leafs, start, mid);
        math::Aabb leftAabb = calculateBoundingBox(leafs, start, mid);
        m_blasNodes.push_back(left);
        uint32_t leftId = m_blasNodes.size() - 1;

        gpu_structs::BvhNode right = buildBvhBlasRecursive(leafs, mid, end);
        math::Aabb rightAabb = calculateBoundingBox(leafs, mid, end);
        m_blasNodes.push_back(right);
        uint32_t rightId = m_blasNodes.size() - 1;

        gpu_structs::BvhNode node;
        node.left_aabb_min_or_v0 = leftAabb.min();
        node.left_or_custom_id = leftId;
        node.left_aabb_max_or_v1 = leftAabb.max();
        node.right_or_material_index = rightId;
        node.right_aabb_min_or_v2 = rightAabb.min();
        node.node_type = gpu_structs::InternalNodeType;
        node.right_aabb_max_or_v3 = rightAabb.max();
        return node;
    }
}

gpu_structs::BvhNode Bvh::buildBvhTlasRecursive(std::vector<Leaf>& leafs, size_t start, size_t end)
{
    size_t leafsSize = end - start;
    if (leafsSize == 0) {
        throw std::runtime_error("[ornament] the scene cannot be empty.");
    } else if (leafsSize == 1) {
        Leaf leaf = leafs[start];
        switch (leaf.type) {
        case SphereType: {
            auto obj = leaf.sphere;
            appendTransform(glm::inverse(obj->transform));
            appendTransform(obj->transform);
            uint32_t transformId = m_transforms.size() / 2 - 1;
            gpu_structs::BvhNode node;
            node.left_aabb_min_or_v0 = obj->aabb.min();
            node.left_aabb_max_or_v1 = obj->aabb.max();
            node.right_or_material_index = getMaterialIndex(*obj->material);
            node.node_type = gpu_structs::SphereType;
            node.transform_id = transformId;
            return node;
        }
        case MeshType: {
            auto obj = leaf.mesh;
            appendTransform(glm::inverse(obj->transform));
            appendTransform(obj->transform);
            uint32_t transformId = m_transforms.size() / 2 - 1;
            gpu_structs::BvhNode node;
            node.left_aabb_min_or_v0 = obj->aabb.min();
            node.left_aabb_max_or_v1 = obj->aabb.max();
            node.right_or_material_index = getMaterialIndex(*obj->material);
            node.left_or_custom_id = obj->bvhId.value();
            node.node_type = gpu_structs::MeshType;
            node.transform_id = transformId;
            return node;
        }
        case MeshInstanceType: {
            auto obj = leaf.meshInstance;
            appendTransform(glm::inverse(obj->transform));
            appendTransform(obj->transform);
            uint32_t transformId = m_transforms.size() / 2 - 1;
            gpu_structs::BvhNode node;
            node.left_aabb_min_or_v0 = obj->aabb.min();
            node.left_aabb_max_or_v1 = obj->aabb.max();
            node.right_or_material_index = getMaterialIndex(*obj->material);
            node.left_or_custom_id = obj->mesh->bvhId.value();
            node.node_type = gpu_structs::MeshType;
            node.transform_id = transformId;
            return node;
        }
        default: {
            throw std::runtime_error("[ornament] not implemented switch case.");
        }
        }
    } else {
        int axis = randomi32(0, 2);
        std::sort(
            leafs.begin() + start,
            leafs.begin() + end,
            [axis](const Leaf& a, const Leaf& b) {
                return getAabb(a).min()[axis] < getAabb(b).min()[axis];
            });

        size_t mid = start + leafsSize / 2;
        gpu_structs::BvhNode left = buildBvhTlasRecursive(leafs, start, mid);
        math::Aabb leftAabb = calculateBoundingBox(leafs, start, mid);
        m_tlasNodes.push_back(left);
        uint32_t leftId = m_tlasNodes.size() - 1;

        gpu_structs::BvhNode right = buildBvhTlasRecursive(leafs, mid, end);
        math::Aabb rightAabb = calculateBoundingBox(leafs, mid, end);
        m_tlasNodes.push_back(right);
        uint32_t rightId = m_tlasNodes.size() - 1;

        gpu_structs::BvhNode node;
        node.left_aabb_min_or_v0 = leftAabb.min();
        node.left_or_custom_id = leftId;
        node.left_aabb_max_or_v1 = leftAabb.max();
        node.right_or_material_index = rightId;
        node.right_aabb_min_or_v2 = rightAabb.min();
        node.node_type = gpu_structs::InternalNodeType;
        node.right_aabb_max_or_v3 = rightAabb.max();
        return node;
    }
}

void Bvh::appendTransform(const glm::mat4& transform)
{
    m_transforms.push_back(glm::transpose(transform));
}

uint32_t Bvh::getMaterialIndex(Material& m)
{
    if (m.materialId.has_value()) {
        return m.materialId.value();
    }

    if (m.albedo.type == TextureType) {
        m_textures.push_back(m.albedo.texture);
        m.albedo.texture->textureId = (uint32_t)(m_textures.size() - 1);
    }

    m_materials.push_back(gpu_structs::Material(m));
    uint32_t materialId = m_materials.size() - 1;
    m.materialId = materialId;
    return materialId;
}

const std::vector<gpu_structs::BvhNode>& Bvh::getTlasNodes() const noexcept
{
    return m_tlasNodes;
}

const std::vector<gpu_structs::BvhNode>& Bvh::getBlasNodes() const noexcept
{
    return m_blasNodes;
}

const std::vector<gpu_structs::Normal>& Bvh::getNormals() const noexcept
{
    return m_normals;
}

const std::vector<uint32_t>& Bvh::getNormalIndices() const noexcept
{
    return m_normalIndices;
}

const std::vector<gpu_structs::Uv>& Bvh::getUvs() const noexcept
{
    return m_uvs;
}

const std::vector<uint32_t>& Bvh::getUvIndices() const noexcept
{
    return m_uvIndices;
}

const std::vector<gpu_structs::Transform>& Bvh::getTransforms() const noexcept
{
    return m_transforms;
}

const std::vector<gpu_structs::Material>& Bvh::getMaterials() const noexcept
{
    return m_materials;
}

const std::vector<ornament::Texture*>& Bvh::getTextures() const noexcept
{
    return m_textures;
}

}