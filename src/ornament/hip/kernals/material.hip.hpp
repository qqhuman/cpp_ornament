#pragma once

#include <hip/hip_runtime.h>
#include "common.hip.hpp"
#include "array.hip.hpp"
#include "random.hip.hpp"
#include "ray.hip.hpp"
#include "hitrecord.hip.hpp"

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
    
    #define EPS 1E-8f
    #define NEAR_ZERO(e) abs(e.x) < EPS && abs(e.y) < EPS && abs(e.z) < EPS

    HOST_DEVICE INLINE float3 get_color(const Array<hipTextureObject_t>& textures, float3& color, uint32_t texture_id, const float2& uv)
    {
        return texture_id < textures.len ? make_float3(tex2D<float4>(textures[texture_id], uv.x, uv.y)) : color;
    }

    HOST_DEVICE INLINE float reflectance(float cosine, float ref_idx)
    {
        // Use Schlick's approximation for reflectance.
        float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
    }

    HOST_DEVICE bool lambertian_scatter(const Ray& r, const HitRecord& hit, RndGen& rnd, const Array<hipTextureObject_t>& textures, float3* attenuation, Ray* scattered)
    {
        float3 scattered_direction = hit.normal + rnd.gen_unit_vector();

        // Catch degenerate scatter direction
        if (NEAR_ZERO(scattered_direction))
        {
            scattered_direction = hit.normal;
        }

        *scattered = Ray(hit.p, scattered_direction);
        *attenuation = get_color(textures, lambertian.albedo, lambertian.albedoTextureId, hit.uv);
        return true;
    }

    HOST_DEVICE bool metal_scatter(const Ray& r, const HitRecord& hit, RndGen& rnd, const Array<hipTextureObject_t>& textures, float3* attenuation, Ray* scattered)
    {
        float3 scattered_direction = reflect(normalize(r.direction), hit.normal) + metal.fuzz * rnd.gen_in_unit_sphere();
        *scattered = Ray(hit.p, scattered_direction);
        *attenuation = get_color(textures, metal.albedo, metal.albedoTextureId, hit.uv);
        return true;
    }

    HOST_DEVICE bool dielectric_scatter(const Ray& r, const HitRecord& hit, RndGen& rnd, float3* attenuation, Ray* scattered)
    {
        *attenuation = make_float3(1.0f);
        float refraction_ratio = dielectric.ior;
        if (hit.front_face)
        {
            refraction_ratio = 1.0f / dielectric.ior;
        }

        float3 unit_direction = normalize(r.direction);
        float cos_theta = min(dot(-unit_direction, hit.normal), 1.0f);
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        float3 direction = cannot_refract || reflectance(cos_theta, refraction_ratio) > rnd.gen_float()
            ? reflect(unit_direction, hit.normal)
            : refract(unit_direction, hit.normal, refraction_ratio);

        *scattered = Ray(hit.p, direction);
        return true;
    }

    HOST_DEVICE bool scatter(const Ray& r, const HitRecord& hit, RndGen& rnd, const Array<hipTextureObject_t>& textures, float3* attenuation, Ray* scattered)
    {
        switch(type) 
        {
            case LambertianType: return lambertian_scatter(r, hit, rnd, textures, attenuation, scattered);
            case MetalType: return metal_scatter(r, hit, rnd, textures, attenuation, scattered);
            case DielectricType: return dielectric_scatter(r, hit, rnd, attenuation, scattered);
            default: return false;
        }
    }

    HOST_DEVICE float3 emit(const HitRecord& hit, const Array<hipTextureObject_t>& textures)
    {
        switch(type) 
        {
            case DiffuseLightType: return get_color(textures, diffuseLight.albedo, diffuseLight.albedoTextureId, hit.uv);
            default: return make_float3(0.0f);
        }
    }
};