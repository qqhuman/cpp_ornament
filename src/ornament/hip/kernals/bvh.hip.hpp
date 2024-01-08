#pragma once

#include <hip/hip_runtime.h>
#include "array.hip.hpp"
#include "common.hip.hpp"
#include "ray.hip.hpp"
#include "vec_math.hip.hpp"
#include "constants.hip.hpp"
#include "transform.hip.hpp"

enum BvhNodeType : uint32_t
{
    InternalNodeType = 0,
    SphereType = 1,
    MeshType = 2,
    TriangleType = 3,
};

#pragma pack(push, 1)
struct InternalNode 
{
    float3 leftAabbMin;
    uint32_t leftNodeId;
    float3 leftAabbMax;
    uint32_t rightNodeId;
    float3 rightAabbMin;
    uint32_t _padding;
    float3 rightAabbMax;
};

struct Sphere
{
    uint32_t materialId;
    uint32_t transformId;
};

struct Mesh
{
    uint32_t materialId;
    uint32_t transformId;
    uint32_t blasNodeId;
};

struct Triangle
{
    float3 v0;
    uint32_t triangleId;
    float3 v1;
    uint32_t _padding;
    float3 v2;
};
#pragma pack(pop)

struct BvhNode
{
    union {
        InternalNode internalNode;
        Sphere sphereNode;
        Mesh meshNode;
        Triangle triangleNode;
    };
    BvhNodeType type;
};

struct Bvh
{
    Array<BvhNode> tlas_nodes;
    Array<BvhNode> blas_nodes;
    Array<float4> normals;
    Array<uint32_t> normal_indices;
    Array<float2> uvs;
    Array<uint32_t> uv_indices;
    Array<float4x4> transforms;

    HOST_DEVICE float3 safe_invdir(float3 d) 
    {
        #define MYCOPYSIGN(a, b) b < 0.0f ? -a : a
        const float eps = 1e-5f;
        float x = abs(d.x) > eps ? d.x : MYCOPYSIGN(eps, d.x);
        float y = abs(d.y) > eps ? d.y : MYCOPYSIGN(eps, d.y);
        float z = abs(d.z) > eps ? d.z : MYCOPYSIGN(eps, d.z);

        return make_float3(1.0f / x, 1.0f / y, 1.0f / z);
    }

    HOST_DEVICE float2 aabb_hit(
        const float3& aabb_min, 
        const float3& aabb_max,
        const float3& invdir,
        const float3& oxinvdir,
        float t_min,
        float t_max) 
    {
        float3 f = aabb_max * invdir + oxinvdir;
        float3 n = aabb_min * invdir + oxinvdir;
        float3 tmax = max(f, n);
        float3 tmin = min(f, n);

        float max_t = min(min(tmax), t_max);
        float min_t = max(max(tmin), t_min);
        return make_float2(min_t, max_t);
    }

    HOST_DEVICE float triangle_hit(
        const Ray& r, 
        const float3& v1,
        const float3& v2,
        const float3& v3,
        float t_min,
        float t_max,
        float2* uv) 
    {
        float3 e1 = v2 - v1;
        float3 e2 = v3 - v1;

        float3 s1 = cross(r.direction, e2);
        float determinant = dot(s1, e1);
        float invd = 1.0f / determinant;

        float3 d = r.origin - v1;
        float u = dot(d, s1) * invd;

        // Barycentric coordinate U is outside range
        if (u < 0.0f || u > 1.0f) 
        {
            return t_max;
        }

        float3 s2 = cross(d, e1);
        float v = dot(r.direction, s2) * invd;

        // Barycentric coordinate V is outside range
        if (v < 0.0f || u + v > 1.0f) 
        {
            return t_max;
        }

        // t
        float t = dot(e2, s2) * invd;
        if (t < t_min || t > t_max) 
        {
            return t_max;
        } 
        else
        {
            *uv = make_float2(u, v);
            return t;
        }
    }

    HOST_DEVICE float sphere_hit(const Ray& ray, float t_min, float t_max)
    {
        #define SPHERE_CENTER make_float3(0.0f)
        #define SPHERE_RADIUS 1.0f

        float3 oc = ray.origin - SPHERE_CENTER;
        float a = length_squared(ray.direction);
        float half_b = dot(oc, ray.direction);
        float c = length_squared(oc) - SPHERE_RADIUS * SPHERE_RADIUS;
        float discriminant = half_b * half_b - a * c;
        if (discriminant < 0.0f) { return t_max; }
        float sqrtd = sqrtf(discriminant);

        float t = (-half_b - sqrtd) / a;
        if (t < t_min || t_max < t)
        {
            t = (-half_b + sqrtd) / a;
            if (t < t_min || t_max < t)
            {
                return t_max;
            }
        }

        return t;
    }
    
    HOST_DEVICE bool hit(
        const Ray& not_transformed_ray,
        float* closest_t, 
        uint32_t* closest_material_id,
        BvhNodeType* closest_bvh_node_type,
        uint32_t* closest_inverted_transform_id,
        uint32_t* closest_tri_id,
        float2* closest_uv) 
    {
        #define finished_traverse_blas 0xffffffff
        float t_min = constant_params.ray_cast_epsilon;
        float t_max = 3.40282e+38;

        int stack_top = 0;
        // here push top of tlas tree to the stack
        uint32_t addr = tlas_nodes.len - 1;
        uint32_t node_stack[64];
        node_stack[stack_top] = addr;
        bool traverse_tlas = true;

        bool hit_anything = false;

        Ray ray = not_transformed_ray;
        float3 invdir = safe_invdir(ray.direction);
        float3 oxinvdir = -ray.origin * invdir;

        float3 not_transformed_invdir = invdir;
        float3 not_transformed_oxinvdir = oxinvdir;
        uint32_t material_id;
        uint32_t inverted_transform_id;
        while (stack_top >= 0)
        {
            BvhNode node = traverse_tlas ? tlas_nodes[addr] : blas_nodes[addr];
            switch (node.type)
            {
                case InternalNodeType: 
                {
                    float2 left = aabb_hit(node.internalNode.leftAabbMin, node.internalNode.leftAabbMax, invdir, oxinvdir, t_min, t_max);
                    float2 right = aabb_hit(node.internalNode.rightAabbMin, node.internalNode.rightAabbMax, invdir, oxinvdir, t_min, t_max);
                    
                    if (left.x <= left.y) 
                    {
                        stack_top++;
                        node_stack[stack_top] = node.internalNode.leftNodeId;
                    }

                    if (right.x <= right.y) 
                    {
                        stack_top++;
                        node_stack[stack_top] = node.internalNode.rightNodeId;
                    }
                    break;
                }
                case SphereType: 
                {
                    inverted_transform_id = node.sphereNode.transformId * 2;
                    Ray transformed_ray = transform_ray(transforms, inverted_transform_id, ray);
                    float t = sphere_hit(transformed_ray, t_min, t_max);
                    if (t < t_max) 
                    {
                        hit_anything = true;
                        t_max = t;
                        *closest_t = t;
                        *closest_material_id = node.sphereNode.materialId;
                        *closest_bvh_node_type = SphereType;
                        *closest_inverted_transform_id = inverted_transform_id;
                    }
                    break;
                }
                case MeshType: 
                {
                    // push signal to restore transformation after finshing mesh bvh
                    traverse_tlas = false;
                    stack_top++;
                    node_stack[stack_top] = finished_traverse_blas;

                    // push mesh bvh
                    stack_top++;
                    node_stack[stack_top] = node.meshNode.blasNodeId;

                    inverted_transform_id = node.meshNode.transformId * 2;
                    material_id = node.meshNode.materialId;
                    ray = transform_ray(transforms, inverted_transform_id, ray);
                    invdir = safe_invdir(ray.direction);
                    oxinvdir = -ray.origin * invdir;
                    break;
                }
                case TriangleType: 
                {
                    float2 uv;
                    float t = triangle_hit(
                        ray, 
                        node.triangleNode.v0,
                        node.triangleNode.v1,
                        node.triangleNode.v2,
                        t_min, 
                        t_max,
                        &uv
                    );

                    if (t < t_max)
                    {
                        hit_anything = true;
                        t_max = t;
                        *closest_t = t;
                        *closest_material_id = material_id;
                        *closest_bvh_node_type = MeshType;
                        *closest_inverted_transform_id = inverted_transform_id;
                        *closest_tri_id = node.triangleNode.triangleId * 3;
                        *closest_uv = uv;
                    }
                    break;
                }
                default: { break; }
            }

            addr = node_stack[stack_top];
            stack_top--;

            if (addr == finished_traverse_blas)
            {
                traverse_tlas = true;
                ray = not_transformed_ray;
                invdir = not_transformed_invdir;
                oxinvdir = not_transformed_oxinvdir;
                addr = node_stack[stack_top];
                stack_top--;
            }
        }

        return hit_anything;
    }
};