#include "examples.hpp"
#include "utils.hpp"
#include <assimp/cimport.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace examples {

std::mt19937 gen32x;

float randomf(float min = 0.0f, float max = 1.0f)
{
    return std::uniform_real_distribution<float>(min, max)(gen32x);
}

glm::vec3 random_vec3(float min = 0.0f, float max = 1.0f)
{
    return { randomf(min, max), randomf(min, max), randomf(min, max) };
}

std::shared_ptr<ornament::Texture> loadTexture(ornament::Scene& scene, const char* filename)
{
    utils::StbImage img = utils::loadImageFromFile(filename, 4);
    std::vector<uint8_t> data(img.data, img.data + img.bytesPerRow * img.height);
    auto txt = scene.texture(std::move(data), img.width, img.height, img.numComponents, img.bytesPerComponent, img.isHdr, 1.0f);
    utils::freeImage(img);
    return txt;
}

std::shared_ptr<ornament::Mesh> loadMesh(ornament::Scene& scene,
    const char* filename,
    const glm::mat4& transform,
    const std::shared_ptr<ornament::Material>& material)
{
    auto aiScene = aiImportFile(filename, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType | aiProcess_GenSmoothNormals);

    if (aiScene->mNumMeshes != 1) {
        throw std::runtime_error("The scene has 0 or more than 1 mesh");
    }

    auto mesh = aiScene->mMeshes[0];

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<uint32_t> indices;

    vertices.reserve(mesh->mNumVertices);
    normals.reserve(mesh->mNumVertices);
    uvs.reserve(mesh->mNumVertices);
    indices.reserve(mesh->mNumFaces * 3);

    glm::vec3 min(std::numeric_limits<float>::infinity());
    glm::vec3 max(-std::numeric_limits<float>::infinity());
    for (size_t i = 0; i < mesh->mNumVertices; i++) {
        auto vertex = mesh->mVertices[i];
        auto normal = mesh->mNormals[i];
        auto textureCoords = mesh->mTextureCoords[0];

        vertices.push_back({ vertex.x, vertex.y, vertex.z });
        normals.push_back(glm::normalize(glm::vec3(normal.x, normal.y, normal.z)));
        if (textureCoords != nullptr) {
            auto uv = textureCoords[i];
            uvs.push_back({ uv.x, uv.y });
        }

        min = glm::min(min, vertices.back());
        max = glm::max(max, vertices.back());
    }

    for (size_t i = 0; i < mesh->mNumFaces; i++) {
        auto face = mesh->mFaces[i];
        indices.push_back(face.mIndices[0]);
        indices.push_back(face.mIndices[1]);
        indices.push_back(face.mIndices[2]);
    }
    std::vector<uint32_t> normalIndices = indices;
    std::vector<uint32_t> uvIndices = indices;

    aiReleaseImport(aiScene);

    glm::vec3 t = (min - glm::vec3(0.0f)) + (max - min) * glm::vec3(0.5f);
    glm::mat4 translate = glm::inverse(glm::translate(glm::mat4(1.0f), t));
    glm::mat4 normalizeMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f / (max.y - min.y))) * translate;
    for (auto& v : vertices) {
        v = glm::vec3(normalizeMatrix * glm::vec4(v, 1.0));
    }

    return scene.mesh(std::move(vertices),
        std::move(indices),
        std::move(normals),
        std::move(normalIndices),
        std::move(uvs),
        std::move(uvIndices),
        transform,
        material);
}

ornament::Scene spheres(float aspectRatio)
{
    float vfov = 20.0f;
    glm::vec3 lookfrom(13.0f, 2.0f, 3.0f);
    glm::vec3 lookat(0.0f, 0.0f, 0.0f);
    glm::vec3 vup(0.0f, 1.0f, 0.0f);
    float aperture = 0.1f;
    float focusDist = 10.0f;
    ornament::Camera camera(lookfrom, lookat, vup, aspectRatio, vfov, aperture, focusDist);
    ornament::Scene scene(camera);

    scene.attach(scene.sphere(
        { 0.0f, -1000.0f, 0.0f },
        1000.0f,
        scene.lambertian(ornament::Color(glm::vec3(0.5f, 0.5f, 0.5f)))));

    std::vector<int> range = { -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    for (auto a : range) {
        for (auto b : range) {
            float chooseMat = randomf();
            glm::vec3 center = { a + 0.9f * randomf(), 0.2f, b + 0.9f * randomf() };

            if (glm::length(center - glm::vec3(4.0f, 0.2f, 0.0f)) > 0.9f) {
                std::shared_ptr<ornament::Material> material;
                if (chooseMat < 0.8f) {
                    material = scene.lambertian(ornament::Color(random_vec3() * random_vec3()));
                } else if (chooseMat < 0.95f) {
                    material = scene.metal(ornament::Color(random_vec3(0.5f, 1.0f)), 0.5f * randomf());
                } else {
                    material = scene.dielectric(1.5f);
                }
                scene.attach(scene.sphere(center, 0.2f, material));
            }
        }
    }

    scene.attach(scene.sphere({ 0.0f, 1.0f, 0.0f }, 1.0f, scene.dielectric(1.5f)));
    scene.attach(scene.sphere({ -4.0f, 1.0f, 0.0f }, 1.0f, scene.lambertian(ornament::Color(glm::vec3(0.4f, 0.2f, 0.1f)))));
    scene.attach(scene.sphere({ 4.0f, 1.0f, 0.0f }, 1.0f, scene.metal(ornament::Color(glm::vec3(0.7f, 0.6f, 0.5f)), 0.0f)));

    return scene;
}

ornament::Scene lucy_and_spheres_with_textures(float aspectRatio)
{
    float vfov = 20.0f;
    glm::vec3 lookfrom(13.0f, 2.0f, 3.0f);
    glm::vec3 lookat(0.0f, 0.0f, 0.0f);
    glm::vec3 vup(0.0f, 1.0f, 0.0f);
    float aperture = 0.1f;
    float focusDist = 10.0f;
    ornament::Camera camera(lookfrom, lookat, vup, aspectRatio, vfov, aperture, focusDist);
    ornament::Scene scene(camera);

    scene.attach(scene.sphere(
        { 0.0f, -1000.0f, 0.0f },
        1000.0f,
        scene.lambertian(ornament::Color(glm::vec3(0.5f, 0.5f, 0.5f)))));

    {
        auto txt = ornament::Color(loadTexture(scene, "C:/my_space/code/cpp/cpp_ornament/apps/glfwapp/assets/textures/earthmap.jpg"));
        scene.attach(scene.sphereMesh({ 0.0f, 1.0f, 0.0f }, 1.0f, scene.lambertian(txt)));
    }
    {
        auto txt = ornament::Color(loadTexture(scene, "C:/my_space/code/cpp/cpp_ornament/apps/glfwapp/assets/textures/2k_mars.jpg"));
        scene.attach(scene.sphere({ -4.0f, 1.0f, 0.0f }, 1.0f, scene.lambertian(txt)));
    }
    {
        auto txt = ornament::Color(loadTexture(scene, "C:/my_space/code/cpp/cpp_ornament/apps/glfwapp/assets/textures/2k_neptune.jpg"));
        scene.attach(scene.sphere({ 4.0f, 1.0f, 0.0f }, 1.0f, scene.lambertian(txt)));
    }

    std::vector<int> range = { -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    for (auto a : range) {
        for (auto b : range) {
            float chooseMat = randomf();
            glm::vec3 center = { a + 0.9f * randomf(), 0.2f, b + 0.9f * randomf() };

            if (glm::length(center - glm::vec3(4.0f, 0.2f, 0.0f)) > 0.9f) {
                std::shared_ptr<ornament::Material> material;
                if (chooseMat < 0.8f) {
                    material = scene.lambertian(ornament::Color(random_vec3() * random_vec3()));
                } else if (chooseMat < 0.95f) {
                    material = scene.metal(ornament::Color(random_vec3(0.5f, 1.0f)), 0.5f * randomf());
                } else {
                    material = scene.dielectric(1.5f);
                }
                scene.attach(scene.sphere(center, 0.2f, material));
            }
        }
    }

    glm::mat4 baseLucyTransform = glm::rotate(glm::mat4(1.0f), glm::pi<float>() / 2.0f, glm::vec3(0.0f, 1.0f, 0.0f)) * glm::scale(glm::mat4(1.0f), glm::vec3(2.0f));
    auto lucyMesh = loadMesh(
        scene,
        "C:/my_space/code/cpp/cpp_ornament/apps/glfwapp/assets/models/lucy.obj",
        glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 1.0f, 2.0f)) * baseLucyTransform,
        scene.dielectric(1.5f));
    scene.attach(lucyMesh);

    scene.attach(scene.meshInstance(
        lucyMesh,
        glm::translate(glm::mat4(1.0f), glm::vec3(-4.0f, 1.0f, 2.0f)) * baseLucyTransform,
        scene.lambertian(ornament::Color(glm::vec3(0.4f, 0.2f, 0.1f)))));

    scene.attach(scene.meshInstance(
        lucyMesh,
        glm::translate(glm::mat4(1.0f), glm::vec3(4.0f, 1.0f, 2.0f)) * baseLucyTransform,
        scene.metal(ornament::Color(glm::vec3(0.7f, 0.6f, 0.5f)), 0.0f)));

    return scene;
}

ornament::Scene spheres_and_sphere_meshes(float aspectRatio)
{
    float vfov = 20.0f;
    glm::vec3 lookfrom(13.0f, 2.0f, 3.0f);
    glm::vec3 lookat(0.0f, 0.0f, 0.0f);
    glm::vec3 vup(0.0f, 1.0f, 0.0f);
    float aperture = 0.1f;
    float focusDist = 10.0f;
    ornament::Camera camera(lookfrom, lookat, vup, aspectRatio, vfov, aperture, focusDist);
    ornament::Scene scene(camera);

    scene.attach(scene.sphere(
        { 0.0f, -1000.0f, 0.0f },
        1000.0f,
        scene.lambertian(ornament::Color(glm::vec3(0.5f, 0.5f, 0.5f)))));

    auto sphereMesh = scene.sphereMesh({ 0.0f, 1.0f, 0.0f }, 1.0f, scene.dielectric(1.5f));
    scene.attach(sphereMesh);

    scene.attach(scene.meshInstance(
        sphereMesh,
        glm::translate(glm::mat4(1.0f), glm::vec3(-4.0f, 1.0f, 0.0f)),
        scene.lambertian(ornament::Color(glm::vec3(0.4f, 0.2f, 0.1f)))));

    scene.attach(scene.meshInstance(
        sphereMesh,
        glm::translate(glm::mat4(1.0f), glm::vec3(4.0f, 1.0f, 0.0f)),
        scene.metal(ornament::Color(glm::vec3(0.7f, 0.6f, 0.5f)), 0.0f)));

    std::vector<int> range = { -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    for (auto a : range) {
        for (auto b : range) {
            float chooseMat = randomf();
            glm::vec3 center = { a + 0.9f * randomf(), 0.2f, b + 0.9f * randomf() };

            if (glm::length(center - glm::vec3(4.0f, 0.2f, 0.0f)) > 0.9f) {
                std::shared_ptr<ornament::Material> material;
                if (chooseMat < 0.8f) {
                    material = scene.lambertian(ornament::Color(random_vec3() * random_vec3()));
                } else if (chooseMat < 0.95f) {
                    material = scene.metal(ornament::Color(random_vec3(0.5f, 1.0f)), 0.5f * randomf());
                } else {
                    material = scene.dielectric(1.5f);
                }

                scene.attach(scene.meshInstance(
                    sphereMesh,
                    glm::translate(glm::mat4(1.0f), center) * glm::scale(glm::mat4(1.0f), glm::vec3(0.2f)),
                    material));
            }
        }
    }

    return scene;
}

ornament::Scene spheres_and_3_lucy(float aspectRatio)
{
    float vfov = 20.0f;
    glm::vec3 lookfrom(13.0f, 2.0f, 3.0f);
    glm::vec3 lookat(0.0f, 0.0f, 0.0f);
    glm::vec3 vup(0.0f, 1.0f, 0.0f);
    float aperture = 0.1f;
    float focusDist = 10.0f;
    ornament::Camera camera(lookfrom, lookat, vup, aspectRatio, vfov, aperture, focusDist);
    ornament::Scene scene(camera);

    scene.attach(scene.sphere(
        { 0.0f, -1000.0f, 0.0f },
        1000.0f,
        scene.lambertian(ornament::Color(glm::vec3(0.5f, 0.5f, 0.5f)))));

    glm::mat4 baseLucyTransform = glm::rotate(glm::mat4(1.0f), glm::pi<float>() / 2.0f, glm::vec3(0.0f, 1.0f, 0.0f));
    auto lucyMesh = loadMesh(
        scene,
        "C:/my_space/code/cpp/cpp_ornament/apps/glfwapp/assets/models/lucy.obj",
        glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 1.0f, 0.0f)) * baseLucyTransform,
        scene.dielectric(1.5f));
    scene.attach(lucyMesh);

    scene.attach(scene.meshInstance(
        lucyMesh,
        glm::translate(glm::mat4(1.0f), glm::vec3(-4.0f, 1.0f, 0.0f)) * baseLucyTransform,
        scene.lambertian(ornament::Color(glm::vec3(0.4f, 0.2f, 0.1f)))));

    scene.attach(scene.meshInstance(
        lucyMesh,
        glm::translate(glm::mat4(1.0f), glm::vec3(4.0f, 1.0f, 0.0f)) * baseLucyTransform,
        scene.metal(ornament::Color(glm::vec3(0.7f, 0.6f, 0.5f)), 0.0f)));

    std::vector<int> range = { -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::optional<std::shared_ptr<ornament::Mesh>> sphereMesh;
    for (auto a : range) {
        for (auto b : range) {
            float chooseMat = randomf();
            glm::vec3 center = { a + 0.9f * randomf(), 0.2f, b + 0.9f * randomf() };

            if (glm::length(center - glm::vec3(4.0f, 0.2f, 0.0f)) > 0.9f) {
                std::shared_ptr<ornament::Material> material;
                if (chooseMat < 0.8f) {
                    material = scene.lambertian(ornament::Color(random_vec3() * random_vec3()));
                } else if (chooseMat < 0.95f) {
                    material = scene.metal(ornament::Color(random_vec3(0.5f, 1.0f)), 0.5f * randomf());
                } else {
                    material = scene.dielectric(1.5f);
                }

                if (sphereMesh.has_value()) {
                    scene.attach(scene.meshInstance(
                        sphereMesh.value(),
                        glm::translate(glm::mat4(1.0f), center) * glm::scale(glm::mat4(1.0f), glm::vec3(0.2f)),
                        material));
                } else {
                    sphereMesh = scene.sphereMesh(center, 0.2f, material);
                    scene.attach(sphereMesh.value());
                }
            }
        }
    }

    return scene;
}

glm::vec3 quadCenterFromBook(const glm::vec3& q, const glm::vec3& u, const glm::vec3& v)
{
    return q + u * 0.5f + v * 0.5f;
}

ornament::Scene empty_cornell_box(float aspectRatio)
{
    float vfov = 40.0f;
    glm::vec3 lookfrom(278.0f, 278.0f, -800.0f);
    glm::vec3 lookat(278.0f, 278.0f, 0.0f);
    glm::vec3 vup(0.0f, 1.0f, 0.0f);
    float aperture = 0.0f;
    float focusDist = 10.0f;
    ornament::Camera camera(lookfrom, lookat, vup, aspectRatio, vfov, aperture, focusDist);
    ornament::Scene scene(camera);

    auto red = scene.lambertian(ornament::Color(glm::vec3(0.65f, 0.05f, 0.05f)));
    auto white = scene.lambertian(ornament::Color(glm::vec3(0.73f, 0.73f, 0.73f)));
    auto green = scene.lambertian(ornament::Color(glm::vec3(0.12f, 0.45f, 0.15f)));
    auto light = scene.diffuseLight(ornament::Color(glm::vec3(15.0f, 15.0f, 15.0f)));

    const glm::vec3 unitX(1.0f, 0.0f, 0.0f);
    const glm::vec3 unitY(0.0f, 1.0f, 0.0f);
    const glm::vec3 unitZ(0.0f, 0.0f, 1.0f);

    scene.attach(scene.planeMesh(
        quadCenterFromBook(
            { 555.0f, 0.0f, 0.0f },
            { 0.0f, 555.0f, 0.0f },
            { 0.0f, 0.0f, 555.0f }),
        555.0f,
        555.0f,
        unitX,
        green));

    scene.attach(scene.planeMesh(
        quadCenterFromBook(
            { 0.0f, 0.0f, 0.0f },
            { 0.0f, 555.0f, 0.0f },
            { 0.0f, 0.0f, 555.0f }),
        555.0f,
        555.0f,
        -unitX,
        red));

    scene.attach(scene.planeMesh(
        quadCenterFromBook(
            { 343.0f, 554.0f, 332.0f },
            { -130.0f, 0.0f, 0.0f },
            { 0.0f, 0.0f, -105.0f }),
        130.0f,
        105.0f,
        -unitY,
        light));

    scene.attach(scene.planeMesh(
        quadCenterFromBook(
            { 0.0f, 0.0f, 0.0f },
            { 555.0f, 0.0f, 0.0f },
            { 0.0f, 0.0f, 555.0f }),
        555.0f,
        555.0f,
        unitY,
        white));

    scene.attach(scene.planeMesh(
        quadCenterFromBook(
            { 555.0f, 555.0f, 555.0f },
            { -555.0f, 0.0f, 0.0f },
            { 0.0f, 0.0f, -555.0f }),
        555.0f,
        555.0f,
        -unitY,
        white));

    scene.attach(scene.planeMesh(
        quadCenterFromBook(
            { 0.0f, 0.0f, 555.0f },
            { 555.0f, 0.0f, 0.0f },
            { 0.0f, 555.0f, 0.0f }),
        555.0f,
        555.0f,
        unitZ,
        white));

    return scene;
}

ornament::Scene cornell_box_with_lucy(float aspectRatio)
{
    float height = 400.0f;
    auto scene = empty_cornell_box(aspectRatio);
    auto mesh = loadMesh(
        scene,
        "C:/my_space/code/cpp/cpp_ornament/apps/glfwapp/assets/models/lucy.obj",
        glm::translate(glm::mat4(1.0f), { 265.0f * 1.5f, height / 2.0f, 295.0f })
            * glm::rotate(glm::mat4(1.0f), glm::pi<float>() * 1.5f, glm::vec3(0.0f, 1.0f, 0.0f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(height)),
        scene.metal(ornament::Color(glm::vec3(0.7f, 0.6f, 0.5f)), 0.0f));
    scene.attach(mesh);

    height = 200.0f;
    scene.attach(scene.meshInstance(
        mesh,
        glm::translate(glm::mat4(1.0f), { 130.0f * 1.5f, height / 2.0f, 65.0f })
            * glm::rotate(glm::mat4(1.0f), glm::pi<float>() * 1.25f, glm::vec3(0.0f, 1.0f, 0.0f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(height)),
        scene.dielectric(1.5f)));

    return scene;
}

}