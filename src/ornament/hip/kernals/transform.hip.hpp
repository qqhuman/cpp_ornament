#pragma once

#include <hip/hip_runtime.h>
#include "vec_math.hip.hpp"
#include "ray.hip.hpp"


namespace ornament {
namespace kernals {

HOST_DEVICE INLINE float3 transformPoint(const float4x4& transform, const float3& point)
{
    float4 p = transform * make_float4(point, 1.0f);
    return make_float3(p);
}

HOST_DEVICE INLINE Ray transformRay(const float4x4& inversedTransform, const Ray& ray)
{
    float4 o = inversedTransform * make_float4(ray.origin, 1.0f);
    float4 d = inversedTransform * make_float4(ray.direction, 0.0f);

    return Ray(
        make_float3(o),
        make_float3(d)
    );
}

HOST_DEVICE INLINE float3 transformNormal(const float4x4& inversedTransform, const float3& normal)
{
    return make_float3(transpose(inversedTransform) * make_float4(normal, 0.0f));
}

}
}