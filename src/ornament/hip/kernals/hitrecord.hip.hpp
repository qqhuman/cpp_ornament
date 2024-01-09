#pragma once

#include <hip/hip_runtime.h>
#include "common.hip.hpp"
#include "ray.hip.hpp"
#include "vec_math.hip.hpp"

namespace ornament {
namespace kernals {

struct HitRecord
{
    float3 p;
    uint32_t materialId;
    float3 normal;
    float t;
    float2 uv;
    bool frontFace;

    HOST_DEVICE void setFaceNormal(const Ray& r, const float3& outwardNormal)
    {
        if (dot(r.direction, outwardNormal) > 0.0f)
        {
            normal = -outwardNormal;
            frontFace = false;
        } 
        else
        {
            normal = outwardNormal;
            frontFace = true;
        }
    }
};

}
}