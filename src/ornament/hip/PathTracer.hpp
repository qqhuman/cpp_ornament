#pragma once

#include <hip/hip_runtime.h>

#include "../GpuStructs.hpp"
#include "../Scene.hpp"
#include "buffers.hpp"

namespace ornament::hip {
class PathTracer {
public:
    PathTracer(Scene scene, const char* kernalsDirPath);
    PathTracer(const PathTracer&) = delete;
    PathTracer& operator=(const PathTracer&) = delete;
    ~PathTracer();
    Scene& getScene() noexcept;
    void getFrameBuffer(uint8_t* dst, size_t size, size_t* retSize);
    void render();

private:
    ornament::Scene m_scene;
    hipModule_t m_module;
    hipFunction_t m_pathTracingKernal;
    hipFunction_t m_postProcessingKernal;
    buffers::Target m_targetBuffer;
    buffers::Textures m_textures;
    buffers::Global<gpu_structs::ConstantParams> m_constantParams;
    buffers::Array<gpu_structs::Material> m_materials;
    buffers::Array<gpu_structs::Normal> m_normals;
    buffers::Array<uint32_t> m_normalIndices;
    buffers::Array<gpu_structs::Uv> m_uvs;
    buffers::Array<uint32_t> m_uvIndices;
    buffers::Array<gpu_structs::Transform> m_transforms;
    buffers::Array<gpu_structs::BvhNode> m_tlasNodes;
    buffers::Array<gpu_structs::BvhNode> m_blasNodes;
    void update();
    void launchKernal(hipFunction_t kernal);
};
}