#include "global_structs_helper.hpp"

namespace ornament::kernals {

float3 glmToHipFloat3(const glm::vec3& v) {
    return make_float3(v[0], v[1], v[2]);
}

Material toKernalMaterial(const ornament::Material& material)
{
    float3 albedo = make_float3(1.0f, 0.0f, 1.0f);
    uint32_t albedoTextureId = std::numeric_limits<uint32_t>::max();
    switch (material.albedo.type) {
    case ornament::VectorType: {
        albedo = glmToHipFloat3(material.albedo.vector);
        break;
    }
    case ornament::TextureType: {
        albedoTextureId = material.albedo.texture->textureId.value();
        break;
    }
    default: {
        break;
    }
    }

    Material kernalMaterial;
    switch (material.type)
    {
    case ornament::Lambertian: {
        kernalMaterial.type = LambertianType;
        kernalMaterial.lambertian.albedo = albedo;
        kernalMaterial.lambertian.albedoTextureId = albedoTextureId;
        break;
    }
    case ornament::Metal: {
        kernalMaterial.type = MetalType;
        kernalMaterial.metal.albedo = albedo;
        kernalMaterial.metal.albedoTextureId = albedoTextureId;
        kernalMaterial.metal.fuzz = material.fuzz;
        break;
    }
    case ornament::Dielectric: {
        kernalMaterial.type = DielectricType;
        kernalMaterial.dielectric.ior = material.ior;
        break;
    }
    case ornament::DiffuseLight: {
        kernalMaterial.type = DiffuseLightType;
        kernalMaterial.diffuseLight.albedo = albedo;
        kernalMaterial.diffuseLight.albedoTextureId = albedoTextureId;
        break;
    }
    default:
        break;
    }

    return kernalMaterial;
}

Camera toKernalCamera(const ornament::Camera& camera)
{
    Camera kernalCamera;
    kernalCamera.origin = glmToHipFloat3(camera.getLookFrom());
    kernalCamera.lowerLeftCorner = glmToHipFloat3(camera.getLowerLeftCorner());
    kernalCamera.horizontal = glmToHipFloat3(camera.getHorizontal());
    kernalCamera.vertical = glmToHipFloat3(camera.getVertical());
    kernalCamera.u = glmToHipFloat3(camera.getU());
    kernalCamera.v = glmToHipFloat3(camera.getV());
    kernalCamera.w = glmToHipFloat3(camera.getW());
    kernalCamera.lensRadius = camera.getLensRadius();
    return kernalCamera;
}

ConstantParams toKernalConstantParams(const ornament::Camera& camera, const ornament::State& state, uint32_t textures)
{
    ConstantParams kernalConstantParams;
    kernalConstantParams.camera = toKernalCamera(camera);
    kernalConstantParams.depth = state.getDepth();
    kernalConstantParams.width = state.getResolution().x;
    kernalConstantParams.height = state.getResolution().y;
    kernalConstantParams.flipY = state.getFlipY() ? 1 : 0;
    kernalConstantParams.invertedGamma = state.getInvertedGamma();
    kernalConstantParams.rayCastEpsilon = state.getRayCastEpsilon();
    kernalConstantParams.texturesCount = textures;
    kernalConstantParams.currentIteration = state.getCurrentIteration();
    return kernalConstantParams;
}

}