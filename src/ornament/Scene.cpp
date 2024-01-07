#include "Scene.hpp"
#include "math/math.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace ornament {

Scene::Scene(Camera camera) noexcept
    : m_camera(camera)
{
}

std::shared_ptr<Material> Scene::lambertian(const Color& albedo)
{
    auto material = std::make_shared<Material>(Material { .type = Lambertian, .albedo = albedo });
    m_materials.push_back(material);
    return material;
}

std::shared_ptr<Material> Scene::metal(const Color& albedo, float fuzz)
{
    auto material = std::make_shared<Material>(Material { .type = Metal, .albedo = albedo, .fuzz = fuzz });
    m_materials.push_back(material);
    return material;
}

std::shared_ptr<Material> Scene::dielectric(float ior)
{
    auto material = std::make_shared<Material>(Material { .type = Dielectric, .ior = ior });
    m_materials.push_back(material);
    return material;
}

std::shared_ptr<Material> Scene::diffuseLight(const Color& albedo)
{
    auto material = std::make_shared<Material>(Material { .type = DiffuseLight, .albedo = albedo });
    m_materials.push_back(material);
    return material;
}

std::shared_ptr<Texture> Scene::texture(std::vector<uint8_t> data,
    uint32_t width,
    uint32_t height,
    uint32_t numComponents,
    uint32_t bytesPerComponent,
    bool isHdr,
    float gamma)
{
    uint32_t bytesPerRow = width * numComponents * bytesPerComponent;
    Texture txt;
    txt.data = std::move(data);
    txt.width = width;
    txt.height = height;
    txt.numComponents = numComponents;
    txt.bytesPerComponent = bytesPerComponent;
    txt.bytesPerRow = bytesPerRow;
    txt.isHdr = isHdr;
    txt.gamma = gamma;

    auto texture = std::make_shared<Texture>(std::move(txt));
    m_textures.push_back(texture);
    return texture;
}

std::shared_ptr<Sphere> Scene::sphere(const glm::vec3& center, float radius, const std::shared_ptr<Material>& material)
{
    auto sphere = std::make_shared<Sphere>(Sphere {
        .material = material,
        .transform = glm::translate(glm::mat4(1.0f), center) * glm::scale(glm::mat4(1.0f), glm::vec3(radius)),
        .aabb = math::Aabb(center - glm::vec3(radius), center + glm::vec3(radius)) });
    m_spheres.push_back(sphere);
    return sphere;
}

std::shared_ptr<Mesh> Scene::mesh(std::vector<glm::vec3> vertices,
    std::vector<uint32_t> vertexIndices,
    std::vector<glm::vec3> normals,
    std::vector<uint32_t> normalIndices,
    std::vector<glm::vec2> uvs,
    std::vector<uint32_t> uvIndices,
    const glm::mat4& transform,
    const std::shared_ptr<Material>& material)
{
    glm::vec3 min(std::numeric_limits<float>::infinity());
    glm::vec3 max(-std::numeric_limits<float>::infinity());

    for (size_t triangleIndex = 0; triangleIndex < vertexIndices.size() / 3; triangleIndex++) {
        auto v0 = vertices[vertexIndices[triangleIndex * 3]];
        auto v1 = vertices[vertexIndices[triangleIndex * 3 + 1]];
        auto v2 = vertices[vertexIndices[triangleIndex * 3 + 2]];
        min = glm::min(min, glm::min(glm::min(v0, v1), v2));
        max = glm::max(max, glm::max(glm::max(v0, v1), v2));
    }

    math::Aabb notTransformedAabb(min, max);
    math::Aabb aabb = math::transform(transform, notTransformedAabb);

    Mesh m;
    m.vertices = std::move(vertices);
    m.vertexIndices = std::move(vertexIndices);
    m.normals = std::move(normals);
    m.normalIndices = std::move(normalIndices);
    m.uvs = std::move(uvs);
    m.uvIndices = std::move(uvIndices);
    m.transform = transform;
    m.material = material;
    m.aabb = aabb;
    m.notTransformedAabb = notTransformedAabb;
    if (m.uvs.size() == 0) {
        m.uvs.resize(m.vertices.size(), glm::vec2(0.5f));
        m.uvIndices = m.vertexIndices;
    }

    auto mesh = std::make_shared<Mesh>(std::move(m));
    m_meshes.push_back(mesh);
    return mesh;
}

std::shared_ptr<Mesh> Scene::sphereMesh(const glm::vec3& center, float radius, const std::shared_ptr<Material>& material)
{
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<uint32_t> indices;
    float facing = 1.0f;
    int hSegments = 60;
    int vSegments = 30;

    // Add the top vertex.
    vertices.push_back({ 0.0f, 1.0f, 0.0f });
    normals.push_back({ 0.0f, facing, 0.0f });
    uvs.push_back(math::getSphereTexCoord(normals.back()));

    for (size_t v = 0; v < vSegments; v++) {
        if (v == 0) {
            continue;
        }

        float theta = v / (float)vSegments * glm::pi<float>();
        float sinTheta = std::sin(theta);

        for (size_t h = 0; h < hSegments; h++) {
            float phi = h / (float)hSegments * glm::pi<float>() * 2.0f;
            float x = sinTheta * std::sin(phi) * 1.0f;
            float z = sinTheta * std::cos(phi) * 1.0f;
            float y = std::cos(theta) * 1.0f;

            vertices.push_back({ x, y, z });
            normals.push_back(glm::normalize(glm::vec3(x, y, z) * glm::vec3(facing)));
            uvs.push_back(math::getSphereTexCoord(normals.back()));

            // Top triangle fan.
            if (v == 1) {
                indices.push_back(0);
                indices.push_back(h + 1);
                if (h < hSegments - 1) {
                    indices.push_back(h + 2);
                } else {
                    indices.push_back(1);
                }
            }
            // Vertical slice.
            else {
                uint32_t i = h + ((v - 1) * hSegments) + 1;
                uint32_t j = i - hSegments;
                uint32_t k = h < hSegments - 1 ? j + 1 : j - (hSegments - 1);
                uint32_t l = h < hSegments - 1 ? i + 1 : i - (hSegments - 1);

                indices.push_back(j);
                indices.push_back(i);
                indices.push_back(k);
                indices.push_back(k);
                indices.push_back(i);
                indices.push_back(l);
            }
        }
    }

    // Bottom vertex.
    vertices.push_back({ 0.0f, -1.0f, 0.0f });
    normals.push_back({ 0.0f, -facing, 0.0f });
    uvs.push_back(math::getSphereTexCoord(normals.back()));

    // Bottom triangle fan.
    uint32_t vertexCount = vertices.size();
    uint32_t end = vertexCount - 1;
    for (size_t h = 0; h < hSegments; h++) {
        uint32_t i = end - hSegments + h;
        indices.push_back(i);
        indices.push_back(end);
        indices.push_back(h < hSegments - 1 ? i + 1 : end - hSegments);
    }

    glm::mat4 transform = glm::translate(glm::mat4(1.0f), center) * glm::scale(glm::mat4(1.0f), glm::vec3(radius));
    std::vector<uint32_t> normalIndices = indices;
    std::vector<uint32_t> uvIndices = indices;
    return mesh(
        std::move(vertices),
        std::move(indices),
        std::move(normals),
        std::move(normalIndices),
        std::move(uvs),
        std::move(uvIndices),
        transform,
        material);
}

std::shared_ptr<Mesh> Scene::planeMesh(const glm::vec3& center,
    float side1_length, float side2_length,
    const glm::vec3& normal,
    const std::shared_ptr<Material>& material)
{
    std::vector<glm::vec3> vertices {
        { -0.5f, 0.0f, -0.5f },
        { -0.5f, 0.0f, 0.5f },
        { 0.5f, 0.0f, 0.5f },
        { 0.5f, 0.0f, -0.5f },
    };

    std::vector<uint32_t> indices { 3, 1, 0, 2, 1, 3 };
    std::vector<uint32_t> normalIndices = indices;
    std::vector<glm::vec3> normals { math::UnitY, math::UnitY, math::UnitY, math::UnitY };
    glm::mat4 rotation = math::rotationBetweenVectors(normal, math::UnitY);
    glm::mat4 transform = glm::translate(glm::mat4(1.0f), center) * rotation * glm::scale(glm::mat4(1.0f), glm::vec3(side1_length, 1.0f, side2_length));
    return mesh(
        std::move(vertices),
        std::move(indices),
        std::move(normals),
        std::move(normalIndices),
        {},
        {},
        transform,
        material);
}

std::shared_ptr<MeshInstance> Scene::meshInstance(const std::shared_ptr<Mesh>& mesh,
    const glm::mat4& transform,
    const std::shared_ptr<Material>& material)
{
    auto mi = std::make_shared<MeshInstance>(MeshInstance {
        .mesh = mesh,
        .material = material,
        .transform = transform,
        .aabb = math::transform(transform, mesh.get()->notTransformedAabb) });
    m_meshInstances.push_back(mi);
    return mi;
}

void Scene::attach(const std::shared_ptr<Sphere>& sphere)
{
    m_attachedSpheres.push_back(sphere);
}

void Scene::attach(const std::shared_ptr<Mesh>& mesh)
{
    m_attachedMeshes.push_back(mesh);
}

void Scene::attach(const std::shared_ptr<MeshInstance>& meshInstance)
{
    m_attachedMeshInstances.push_back(meshInstance);
}

State& Scene::getState() noexcept
{
    return m_state;
}

Camera& Scene::getCamera() noexcept
{
    return m_camera;
}

const std::vector<std::shared_ptr<Sphere>>& Scene::getAttachedSpheres() const noexcept
{
    return m_attachedSpheres;
}

const std::vector<std::shared_ptr<Mesh>>& Scene::getAttachedMeshes() const noexcept
{
    return m_attachedMeshes;
}

const std::vector<std::shared_ptr<MeshInstance>>& Scene::getAttachedMeshInstances() const noexcept
{
    return m_attachedMeshInstances;
}

const std::vector<std::shared_ptr<Material>>& Scene::getMaterials() const noexcept
{
    return m_materials;
}

const std::vector<std::shared_ptr<Texture>>& Scene::getTextures() const noexcept
{
    return m_textures;
}

}