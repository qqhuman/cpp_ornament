#pragma once

#include <glm/glm.hpp>
#include <math/math.hpp>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "Camera.hpp"
#include "State.hpp"

namespace ornament {

struct Texture {
    Texture() = default;
    Texture(Texture&& texture) = default;
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    std::vector<uint8_t> data;
    uint32_t width;
    uint32_t height;
    uint32_t numComponents;
    uint32_t bytesPerComponent;
    uint32_t bytesPerRow;
    bool isHdr;
    float gamma;
    std::optional<uint32_t> textureId;
};

enum ColorValueType {
    VectorType,
    TextureType
};

struct Color {
    ColorValueType type;
    union {
        Texture* texture;
        glm::vec3 vector;
    };

    Color() noexcept
    {
        type = VectorType;
        vector = {};
    }

    Color(const Color& color)
    {
        type = color.type;
        switch (color.type) {
        case VectorType: {
            vector = color.vector;
            break;
        }
        case TextureType: {
            texture = color.texture;
            break;
        }
        default: {
            throw std::runtime_error("[ornament] not implemented switch case.");
        }
        }
    }

    Color(const glm::vec3& value) noexcept
    {
        type = VectorType;
        vector = value;
    }

    Color(const std::shared_ptr<Texture>& value) noexcept
    {
        type = TextureType;
        texture = value.get();
    }
};

enum MaterialType : uint32_t {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
    DiffuseLight = 3,
};

struct Material {
    MaterialType type;
    Color albedo;
    float fuzz;
    float ior;
    std::optional<uint32_t> materialId;
};

struct Mesh {
    Mesh() = default;
    Mesh(Mesh&& mesh) = default;
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    std::vector<glm::vec3> vertices;
    std::vector<uint32_t> vertexIndices;
    std::vector<glm::vec3> normals;
    std::vector<uint32_t> normalIndices;
    std::vector<glm::vec2> uvs;
    std::vector<uint32_t> uvIndices;
    glm::mat4 transform;
    std::shared_ptr<Material> material;
    std::optional<uint32_t> bvhId;
    math::Aabb aabb;
    math::Aabb notTransformedAabb;
};

struct MeshInstance {
    std::shared_ptr<Mesh> mesh;
    std::shared_ptr<Material> material;
    glm::mat4 transform;
    math::Aabb aabb;
};

struct Sphere {
    std::shared_ptr<Material> material;
    glm::mat4 transform;
    math::Aabb aabb;
};

class Scene {
public:
    Scene(Camera camera) noexcept;
    Scene(Scene&& scene) = default;
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    std::shared_ptr<Material> lambertian(const Color& albedo);
    std::shared_ptr<Material> metal(const Color& albedo, float fuzz);
    std::shared_ptr<Material> dielectric(float ior);
    std::shared_ptr<Material> diffuseLight(const Color& albedo);
    std::shared_ptr<Texture> texture(std::vector<uint8_t> data,
        uint32_t width,
        uint32_t height,
        uint32_t numComponents,
        uint32_t bytesPerComponent,
        bool isHdr,
        float gamma);
    std::shared_ptr<Sphere> sphere(const glm::vec3& center, float radius, const std::shared_ptr<Material>& material);
    std::shared_ptr<Mesh> mesh(std::vector<glm::vec3> vertices,
        std::vector<uint32_t> vertexIndices,
        std::vector<glm::vec3> normals,
        std::vector<uint32_t> normalIndices,
        std::vector<glm::vec2> uvs,
        std::vector<uint32_t> uvIndices,
        const glm::mat4& transform,
        const std::shared_ptr<Material>& material);
    std::shared_ptr<Mesh> sphereMesh(const glm::vec3& center, float radius, const std::shared_ptr<Material>& material);
    std::shared_ptr<Mesh> planeMesh(const glm::vec3& center, float side1_length, float side2_length, const glm::vec3& normal, const std::shared_ptr<Material>& material);
    std::shared_ptr<MeshInstance> meshInstance(const std::shared_ptr<Mesh>& mesh,
        const glm::mat4& transform,
        const std::shared_ptr<Material>& material);

    void attach(const std::shared_ptr<Sphere>& sphere);
    void attach(const std::shared_ptr<Mesh>& mesh);
    void attach(const std::shared_ptr<MeshInstance>& meshInstance);

    State& getState() noexcept;
    Camera& getCamera() noexcept;
    const std::vector<std::shared_ptr<Sphere>>& getAttachedSpheres() const noexcept;
    const std::vector<std::shared_ptr<Mesh>>& getAttachedMeshes() const noexcept;
    const std::vector<std::shared_ptr<MeshInstance>>& getAttachedMeshInstances() const noexcept;
    const std::vector<std::shared_ptr<Material>>& getMaterials() const noexcept;
    const std::vector<std::shared_ptr<Texture>>& getTextures() const noexcept;

private:
    Camera m_camera;
    State m_state;
    std::vector<std::shared_ptr<Sphere>> m_spheres;
    std::vector<std::shared_ptr<Mesh>> m_meshes;
    std::vector<std::shared_ptr<MeshInstance>> m_meshInstances;
    std::vector<std::shared_ptr<Material>> m_materials;
    std::vector<std::shared_ptr<Texture>> m_textures;
    std::vector<std::shared_ptr<Sphere>> m_attachedSpheres;
    std::vector<std::shared_ptr<Mesh>> m_attachedMeshes;
    std::vector<std::shared_ptr<MeshInstance>> m_attachedMeshInstances;
};

}