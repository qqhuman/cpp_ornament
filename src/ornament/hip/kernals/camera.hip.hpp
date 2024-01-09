#pragma once

#include <hip/hip_runtime.h>
#include "global_structs.hip.hpp"
#include "ray.hip.hpp"
#include "random.hip.hpp"

namespace ornament {
namespace kernals {

HOST_DEVICE INLINE Ray cameraGetRay(const Camera& camera, RndGen& rnd, float s, float t)
{
    float3 rd = camera.lensRadius * rnd.genInUnitDisk();
    float3 offset = camera.u * rd.x + camera.v * rd.y;
    return Ray(
        camera.origin + offset, 
        camera.lowerLeftCorner + s * camera.horizontal + t * camera.vertical - camera.origin - offset
    );
}

}
}