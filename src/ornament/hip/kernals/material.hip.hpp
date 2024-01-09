#pragma once

#include <hip/hip_runtime.h>
#include "common.hip.hpp"
#include "random.hip.hpp"
#include "ray.hip.hpp"
#include "hitrecord.hip.hpp"

namespace ornament {
namespace kernals {

#define EPS 1E-8f
#define NEAR_ZERO(e) abs(e.x) < EPS && abs(e.y) < EPS && abs(e.z) < EPS

HOST_DEVICE INLINE  float3 getColor(const Array<hipTextureObject_t>& textures, 
    const float3& color, 
    uint32_t textureId, 
    const float2& uv)
{
    return textureId < textures.len ? make_float3(tex2D<float4>(textures[textureId], uv.x, uv.y)) : color;
}

HOST_DEVICE INLINE float reflectance(float cosine, float refIdx)
{
    // Use Schlick's approximation for reflectance.
    float r0 = (1.0f - refIdx) / (1.0f + refIdx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

HOST_DEVICE bool scatter(const Lambertian& lambertian,
    const Ray& r,
    const HitRecord& hit,
    RndGen& rnd,
    const Array<hipTextureObject_t>& textures,
    float3* attenuation,
    Ray* scattered)
{
    float3 scatteredDirection = hit.normal + rnd.genUnitVector();

    // Catch degenerate scatter direction
    if (NEAR_ZERO(scatteredDirection))
    {
        scatteredDirection = hit.normal;
    }

    *scattered = Ray(hit.p, scatteredDirection);
    *attenuation = getColor(textures, lambertian.albedo, lambertian.albedoTextureId, hit.uv);
    return true;
}

HOST_DEVICE bool scatter(const Metal& metal, 
    const Ray& r, 
    const HitRecord& hit, 
    RndGen& rnd,
    const Array<hipTextureObject_t>& textures,
    float3* attenuation,
    Ray* scattered)
{
    float3 scatteredDirection = reflect(normalize(r.direction), hit.normal) + metal.fuzz * rnd.genInUnitSphere();
    *scattered = Ray(hit.p, scatteredDirection);
    *attenuation = getColor(textures, metal.albedo, metal.albedoTextureId, hit.uv);
    return true;
}

HOST_DEVICE bool scatter(const Dielectric& dielectric,
    const Ray& r,
    const HitRecord& hit,
    RndGen& rnd,
    float3* attenuation,
    Ray* scattered)
{
    *attenuation = make_float3(1.0f);
    float refractionRatio = dielectric.ior;
    if (hit.frontFace)
    {
        refractionRatio = 1.0f / dielectric.ior;
    }

    float3 unitDirection = normalize(r.direction);
    float cosTheta = min(dot(-unitDirection, hit.normal), 1.0f);
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    bool cannotRefract = refractionRatio * sinTheta > 1.0f;
    float3 direction = cannotRefract || reflectance(cosTheta, refractionRatio) > rnd.genFloat()
        ? reflect(unitDirection, hit.normal)
        : refract(unitDirection, hit.normal, refractionRatio);

    *scattered = Ray(hit.p, direction);
    return true;
}

HOST_DEVICE bool materialScatter(const Material& material,
    const Ray& r,
    const HitRecord& hit,
    RndGen& rnd,
    const Array<hipTextureObject_t>& textures,
    float3* attenuation,
    Ray* scattered)
{
    switch(material.type) 
    {
        case LambertianType: return scatter(material.lambertian, r, hit, rnd, textures, attenuation, scattered);
        case MetalType: return scatter(material.metal, r, hit, rnd, textures, attenuation, scattered);
        case DielectricType: return scatter(material.dielectric, r, hit, rnd, attenuation, scattered);
        default: return false;
    }
}

HOST_DEVICE float3 materialEmit(const Material& material,
    const HitRecord& hit,
    const Array<hipTextureObject_t>& textures)
{
    switch(material.type) 
    {
        case DiffuseLightType: return getColor(textures, material.diffuseLight.albedo, material.diffuseLight.albedoTextureId, hit.uv);
        default: return make_float3(0.0f);
    }
}

}
}