#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_math_constants.h>
#include "common.hip.hpp"
#include "vec_math.hip.hpp"

namespace ornament {
namespace kernals {
struct RndGen
{
    uint32_t state;

    HOST_DEVICE RndGen(uint32_t seed) : state(seed) {}

    HOST_DEVICE INLINE uint32_t genUint32()
    {
        // PCG random number generator
        // Based on https://www.shadertoy.com/view/XlGcRh
        
        // rng_state = (word >> 22u) ^ word;
        // return rng_state;
        uint32_t oldState = state + 747796405 + 2891336453;
        uint32_t word = ((oldState >> ((oldState >> 28) + 4)) ^ oldState) * 277803737;
        state = (word >> 22) ^ word;
        return state;
    }

    HOST_DEVICE INLINE float genFloat()
    {
        return (float)genUint32() * (1.0f / 4294967296.0f);
    }

    HOST_DEVICE INLINE float genFloat(float min, float max)
    {
        return min + (max - min) * genFloat();
    }

    HOST_DEVICE INLINE float3 genFloat3()
    {
        return make_float3(genFloat(), genFloat(), genFloat());
    }

    HOST_DEVICE INLINE float3 genFloat3(float min, float max)
    {
        return make_float3(genFloat(min, max), genFloat(min, max), genFloat(min, max));
    }

    HOST_DEVICE INLINE float3 genInUnitSphere()
    {
        float r = pow(genFloat(), 0.33333f);
        float theta = HIP_PI_F * genFloat();
        float phi = 2.0f * HIP_PI_F * genFloat();

        float sinTheta, cosTheta;
        sincosf(theta, &sinTheta, &cosTheta);
        float sinPhi, cosPhi;
        sincosf(phi, &sinPhi, &cosPhi);

        float x = r * sinTheta * cosPhi;
        float y = r * sinTheta * sinPhi;
        float z = r * cosTheta;

        return make_float3(x, y, z);
    }

    HOST_DEVICE INLINE float3 genUnitVector()
    {
        return normalize(genInUnitSphere());
    }

    HOST_DEVICE INLINE float3 genOnHemisphere(const float3& normal)
    {
        float3 onUnitSphere = genUnitVector();
        if (dot(onUnitSphere, normal) > 0.0f)
        {
            return onUnitSphere;
        } 
        else
        {
            return -onUnitSphere;
        }
    }

    HOST_DEVICE INLINE float3 genInUnitDisk()
    {
        // r^2 is distributed as U(0, 1).
        float r = sqrtf(genFloat());
        float alpha = 2.0f * HIP_PI_F * genFloat();

        float sinAlpha, cosAlpha;
        sincosf(alpha, &sinAlpha, &cosAlpha);
        float x = r * cosAlpha;
        float y = r * sinAlpha;

        return make_float3(x, y, 0.0f);
    }
};

}
}