#include <hip/hip_runtime.h>
#include <hip/hip_math_constants.h>
#include "vec_math.hip.hpp"
#include "global_structs.hip.hpp"
#include "random.hip.hpp"
#include "camera.hip.hpp"
#include "bvh.hip.hpp"
#include "material.hip.hpp"
#include "hitrecord.hip.hpp"
#include "transform.hip.hpp"

using namespace ornament::kernals;

__constant__ ConstantParams constantParams;

extern "C" __global__ void pathTracingKernal(KernalBuffers kbuffs) {
    uint32_t globalId = blockDim.x * blockIdx.x + threadIdx.x;
    if (globalId >= kbuffs.frameBuffer.len) {
        return;
    }

    uint2 globalXY = make_uint2(globalId % constantParams.width, globalId / constantParams.width);
    RndGen rnd(kbuffs.rngSeedBuffer[globalId]);

    float u = ((float)globalXY.x + rnd.genFloat()) / (constantParams.width - 1);
    float v = ((float)globalXY.y + rnd.genFloat()) / (constantParams.height - 1);

    Ray ray = cameraGetRay(constantParams.camera, rnd, u, v);
    float3 finalColor = make_float3(1.0f);

    for (int i = 0; i < constantParams.depth; i += 1)
    {
        BvhHitResult bvhHitResult;
        if (!bvhHit(kbuffs.bvh, ray, constantParams.rayCastEpsilon, &bvhHitResult)) {
            float3 unitDirection = normalize(ray.direction);
            float tt = 0.5f * (unitDirection.y + 1.0f);
            finalColor = finalColor * ((1.0f - tt) * make_float3(1.0f) + tt * make_float3(0.5f, 0.7f, 1.0f));
            //finalColor = make_float3(0.0f);
            break;
        }

        uint32_t transformId = bvhHitResult.invertedTransformId + 1;
        HitRecord hit;
        hit.t = bvhHitResult.t;
        hit.p = ray.at(bvhHitResult.t);
        hit.materialId = bvhHitResult.materialId;
        switch (bvhHitResult.nodeType)
        {
            case SphereType: 
            {
                float3 center = transformPoint(kbuffs.bvh.transforms[transformId], make_float3(0.0f));
                float3 outwardNormal = normalize(hit.p - center);
                float theta = acos(-outwardNormal.y);
                float phi = atan2(-outwardNormal.z, outwardNormal.x) + HIP_PI_F;
                hit.uv = make_float2(phi / (2.0f * HIP_PI_F), theta / HIP_PI_F);
                hit.setFaceNormal(ray, outwardNormal);
                break;
            }
            case MeshType: 
            {
                float4 n0 = kbuffs.bvh.normals[kbuffs.bvh.normalIndices[bvhHitResult.triangleId]];
                float4 n1 = kbuffs.bvh.normals[kbuffs.bvh.normalIndices[bvhHitResult.triangleId + 1]];
                float4 n2 = kbuffs.bvh.normals[kbuffs.bvh.normalIndices[bvhHitResult.triangleId + 2]];

                float2 uv0 = kbuffs.bvh.uvs[kbuffs.bvh.uvIndices[bvhHitResult.triangleId]];
                float2 uv1 = kbuffs.bvh.uvs[kbuffs.bvh.uvIndices[bvhHitResult.triangleId + 1]];
                float2 uv2 = kbuffs.bvh.uvs[kbuffs.bvh.uvIndices[bvhHitResult.triangleId + 2]];

                float w = 1.0f - bvhHitResult.triangleBarycentricUV.x - bvhHitResult.triangleBarycentricUV.y;
                float4 normal = w * n0 + bvhHitResult.triangleBarycentricUV.x * n1 + bvhHitResult.triangleBarycentricUV.y * n2;
                hit.uv = w * uv0 + bvhHitResult.triangleBarycentricUV.x * uv1 + bvhHitResult.triangleBarycentricUV.y * uv2;
                float3 outwardNormal = normalize(transformNormal(
                    kbuffs.bvh.transforms[bvhHitResult.invertedTransformId],
                    make_float3(normal)
                ));
                hit.setFaceNormal(ray, outwardNormal);
                break;
            }
            default: { break; }
        }

        float3 attenuation;
        Ray scattered;
        Material material = kbuffs.materials[hit.materialId];
        if (materialScatter(material, ray, hit, rnd, kbuffs.textures, &attenuation, &scattered)) {
            ray = scattered;
            finalColor = finalColor * attenuation;
        } else {
            finalColor = finalColor * materialEmit(material, hit, kbuffs.textures);
            break;
        }
    }
    
    float4 accumulatedRgba = make_float4(finalColor, 1.0f);
    if (constantParams.currentIteration > 1.0f) {
        accumulatedRgba = kbuffs.accumulationBuffer[globalId] + accumulatedRgba;
    }

    kbuffs.accumulationBuffer[globalId] = accumulatedRgba;
    kbuffs.rngSeedBuffer[globalId] = rnd.state;
}

extern "C" __global__ void postProcessingKernal(KernalBuffers kbuffs) {
    uint32_t globalId = blockDim.x * blockIdx.x + threadIdx.x;
    if (globalId >= kbuffs.frameBuffer.len) {
        return;
    }
    
    float4 rgba = kbuffs.accumulationBuffer[globalId] / constantParams.currentIteration;
    rgba.x = pow(rgba.x, constantParams.invertedGamma);
    rgba.y = pow(rgba.y, constantParams.invertedGamma);
    rgba.z = pow(rgba.z, constantParams.invertedGamma);
    rgba = clamp(rgba, 0.0f, 1.0f);

    uint32_t fbIndex = globalId;
    if (constantParams.flipY != 0) {
        uint2 globalXY = make_uint2(globalId % constantParams.width, globalId / constantParams.width);
        uint32_t flippedY = constantParams.height - globalXY.y - 1;
        fbIndex = constantParams.width * flippedY + globalXY.x;
    }

    kbuffs.frameBuffer[fbIndex] = rgba;
}