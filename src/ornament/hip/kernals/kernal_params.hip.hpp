#pragma once

#include <hip/hip_runtime.h>
#include "common.hip.hpp"
#include "bvh.hip.hpp"
#include "material.hip.hpp"
#include "random.hip.hpp"
#include "array.hip.hpp"
#include "bvh.hip.hpp"

struct KernalGlobals
{
    Bvh bvh;
    Array<Material> materials;
    Array<hipTextureObject_t> textures;
    float4* framebuffer;
    float4* accumulation_buffer;
    uint32_t* rng_seed_buffer;
    uint32_t pixel_count;
};

struct KernalLocalState
{
    KernalGlobals kg;
    uint2 xy;
    uint32_t global_invocation_id;
    RndGen rnd;

    HOST_DEVICE KernalLocalState(const KernalGlobals& kg, uint2 resolution, uint32_t global_invocation_id) : kg(kg),
        xy(make_uint2(global_invocation_id % resolution.x, global_invocation_id / resolution.x)),
        global_invocation_id(global_invocation_id), 
        rnd(kg.rng_seed_buffer[global_invocation_id])
    {}

    HOST_DEVICE INLINE void save_rng_seed()
    {
        kg.rng_seed_buffer[global_invocation_id] = rnd.state;
    }
};