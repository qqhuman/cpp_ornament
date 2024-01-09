#pragma once

#include <hip/hip_runtime.h>
#include "global_structs.hip.hpp"
#include "ray.hip.hpp"
#include "vec_math.hip.hpp"
#include "transform.hip.hpp"

namespace ornament {
namespace kernals {

HOST_DEVICE float3 safeInvdir(float3 d) 
{
    #define MYCOPYSIGN(a, b) b < 0.0f ? -a : a
    const float eps = 1e-5f;
    float x = abs(d.x) > eps ? d.x : MYCOPYSIGN(eps, d.x);
    float y = abs(d.y) > eps ? d.y : MYCOPYSIGN(eps, d.y);
    float z = abs(d.z) > eps ? d.z : MYCOPYSIGN(eps, d.z);

    return make_float3(1.0f / x, 1.0f / y, 1.0f / z);
}

HOST_DEVICE float2 aabbHit(
    const float3& aabbMin, 
    const float3& aabbMax,
    const float3& invdir,
    const float3& oxinvdir,
    float tmin,
    float tmax) 
{
    float3 f = aabbMax * invdir + oxinvdir;
    float3 n = aabbMin * invdir + oxinvdir;
    float3 tmaxf3 = max(f, n);
    float3 tminf3 = min(f, n);
    float t1 = min(min(tmaxf3), tmax);
    float t0 = max(max(tminf3), tmin);
    return make_float2(t0, t1);
}

HOST_DEVICE float triangleHit(
    const Ray& r, 
    const Triangle& triangle,
    float tmin,
    float tmax,
    float2* uv) 
{
    float3 e1 = triangle.v1 - triangle.v0;
    float3 e2 = triangle.v2 - triangle.v0;

    float3 s1 = cross(r.direction, e2);
    float determinant = dot(s1, e1);
    float invd = 1.0f / determinant;

    float3 d = r.origin - triangle.v0;
    float u = dot(d, s1) * invd;

    // Barycentric coordinate U is outside range
    if (u < 0.0f || u > 1.0f) 
    {
        return tmax;
    }

    float3 s2 = cross(d, e1);
    float v = dot(r.direction, s2) * invd;

    // Barycentric coordinate V is outside range
    if (v < 0.0f || u + v > 1.0f) 
    {
        return tmax;
    }

    // t
    float t = dot(e2, s2) * invd;
    if (t < tmin || t > tmax) 
    {
        return tmax;
    } 
    else
    {
        *uv = make_float2(u, v);
        return t;
    }
}

HOST_DEVICE float sphereHit(const Ray& ray, float tmin, float tmax)
{
    #define SPHERE_CENTER make_float3(0.0f)
    #define SPHERE_RADIUS 1.0f

    float3 oc = ray.origin - SPHERE_CENTER;
    float a = length_squared(ray.direction);
    float halfB = dot(oc, ray.direction);
    float c = length_squared(oc) - SPHERE_RADIUS * SPHERE_RADIUS;
    float discriminant = halfB * halfB - a * c;
    if (discriminant < 0.0f) { return tmax; }
    float sqrtd = sqrtf(discriminant);

    float t = (-halfB - sqrtd) / a;
    if (t < tmin || tmax < t)
    {
        t = (-halfB + sqrtd) / a;
        if (t < tmin || tmax < t)
        {
            return tmax;
        }
    }

    return t;
}

struct BvhHitResult {
    float t;
    uint32_t materialId;
    BvhNodeType nodeType;
    uint32_t invertedTransformId;
    uint32_t triangleId;
    float2 triangleBarycentricUV;
};

HOST_DEVICE bool bvhHit(const Bvh& bvh,
    const Ray& notTransformedRay,
    float rayCastEpsilon,
    BvhHitResult* result)
{
    #define FINISH_TRAVERSE_BLAS 0xffffffff
    float tmin = rayCastEpsilon;
    float tmax = 3.40282e+38;

    int stackTop = 0;
    // here push top of tlas tree to the stack
    uint32_t addr = bvh.tlasNodes.len - 1;
    uint32_t nodeStack[64];
    nodeStack[stackTop] = addr;
    bool traverseTlas = true;

    bool hitAnything = false;

    Ray ray = notTransformedRay;
    float3 invdir = safeInvdir(ray.direction);
    float3 oxinvdir = -ray.origin * invdir;

    float3 notTransformedInvdir = invdir;
    float3 notTransformedOxinvdir = oxinvdir;
    uint32_t materialId;
    uint32_t invertedTransformId;
    while (stackTop >= 0)
    {
        BvhNode node = traverseTlas ? bvh.tlasNodes[addr] : bvh.blasNodes[addr];
        switch (node.type)
        {
            case InternalNodeType: 
            {
                float2 left = aabbHit(node.internalNode.leftAabbMin, node.internalNode.leftAabbMax, invdir, oxinvdir, tmin, tmax);
                float2 right = aabbHit(node.internalNode.rightAabbMin, node.internalNode.rightAabbMax, invdir, oxinvdir, tmin, tmax);
                
                if (left.x <= left.y) 
                {
                    stackTop++;
                    nodeStack[stackTop] = node.internalNode.leftNodeId;
                }

                if (right.x <= right.y) 
                {
                    stackTop++;
                    nodeStack[stackTop] = node.internalNode.rightNodeId;
                }
                break;
            }
            case SphereType: 
            {
                invertedTransformId = node.sphereNode.transformId * 2;
                float t = sphereHit(
                    transformRay(bvh.transforms[invertedTransformId], ray), 
                    tmin, 
                    tmax);
                if (t < tmax) 
                {
                    hitAnything = true;
                    tmax = t;
                    result->t = t;
                    result->materialId = node.sphereNode.materialId;
                    result->nodeType = SphereType;
                    result->invertedTransformId = invertedTransformId;
                }
                break;
            }
            case MeshType: 
            {
                // push signal to restore transformation after finshing mesh bvh
                traverseTlas = false;
                stackTop++;
                nodeStack[stackTop] = FINISH_TRAVERSE_BLAS;

                // push mesh bvh
                stackTop++;
                nodeStack[stackTop] = node.meshNode.blasNodeId;

                invertedTransformId = node.meshNode.transformId * 2;
                materialId = node.meshNode.materialId;
                ray = transformRay(bvh.transforms[invertedTransformId], ray);
                invdir = safeInvdir(ray.direction);
                oxinvdir = -ray.origin * invdir;
                break;
            }
            case TriangleType: 
            {
                float2 uv;
                float t = triangleHit(
                    ray, 
                    node.triangleNode,
                    tmin, 
                    tmax,
                    &uv
                );

                if (t < tmax)
                {
                    hitAnything = true;
                    tmax = t;
                    result->t = t;
                    result->materialId = materialId;
                    result->nodeType = MeshType;
                    result->invertedTransformId = invertedTransformId;
                    result->triangleId = node.triangleNode.triangleId * 3;
                    result->triangleBarycentricUV = uv;
                }
                break;
            }
            default: { break; }
        }

        addr = nodeStack[stackTop];
        stackTop--;

        if (addr == FINISH_TRAVERSE_BLAS)
        {
            traverseTlas = true;
            ray = notTransformedRay;
            invdir = notTransformedInvdir;
            oxinvdir = notTransformedOxinvdir;
            addr = nodeStack[stackTop];
            stackTop--;
        }
    }

    return hitAnything;
}

}
}