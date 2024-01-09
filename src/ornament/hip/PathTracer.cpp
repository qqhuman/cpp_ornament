#include <filesystem>
#include <hip/hip_runtime.h>

#include "Bvh.hpp"
#include "PathTracer.hpp"
#include "buffers.hpp"
#include "hip_helper.hpp"
#include "../global_structs_helper.hpp"

namespace ornament::hip {
PathTracer::PathTracer(Scene scene, const char* kernalsDirPath)
    : m_scene(std::move(scene))
{
    Bvh bvh(m_scene);

    int deviceCount = 0;
    checkHipErrors(hipGetDeviceCount(&deviceCount));
    hipDeviceProp_t prop;
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        checkHipErrors(hipGetDeviceProperties(&prop, deviceId));
        printf("Device id =  %d\n", deviceId);
        printf("      name = %s\n", prop.name);
        printf("      warpSize = %d\n", prop.warpSize);
        printf("      totalGlobalMem = %fGB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("      sharedMemPerBlock = %fKB\n", prop.sharedMemPerBlock / 1024.0f);
        printf("      regsPerBlock = %d\n", prop.regsPerBlock);
        printf("      maxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock);
        printf("      integrated = %d\n", prop.integrated);
        printf("      gcnArchName = %s\n", prop.gcnArchName);
    }

    int wantedDeviceId = deviceCount - 1;
    checkHipErrors(hipSetDevice(wantedDeviceId));
    checkHipErrors(hipGetDeviceProperties(&prop, wantedDeviceId));

    auto kernalsPath = std::filesystem::path(kernalsDirPath) / std::filesystem::path("ornament_kernals.co");
    printf("      kernals path = %s\n", kernalsPath.string().c_str());
    checkHipErrors(hipModuleLoad(&m_module, kernalsPath.string().c_str()));
    checkHipErrors(hipModuleGetFunction(&m_pathTracingKernal, m_module, "pathTracingKernal"));
    checkHipErrors(hipModuleGetFunction(&m_postProcessingKernal, m_module, "postProcessingKernal"));

    uint2 resolution = {m_scene.getState().getResolution().x, m_scene.getState().getResolution().y};
    m_targetBuffer = buffers::Target(resolution);
    m_textures = buffers::Textures(bvh.getTextures(), prop.texturePitchAlignment);
    m_constantParams = buffers::Global<kernals::ConstantParams>("constantParams", m_module);
    m_materials = buffers::Array(bvh.getMaterials());
    m_normals = buffers::Array(bvh.getNormals());
    m_normalIndices = buffers::Array(bvh.getNormalIndices());
    m_uvs = buffers::Array(bvh.getUvs());
    m_uvIndices = buffers::Array(bvh.getUvIndices());
    m_transforms = buffers::Array(bvh.getTransforms());
    m_tlasNodes = buffers::Array(bvh.getTlasNodes());
    m_blasNodes = buffers::Array(bvh.getBlasNodes());
}

PathTracer::~PathTracer()
{
    checkHipErrors(hipModuleUnload(m_module));
}

void PathTracer::update()
{
    bool dirty = false;
    Camera& camera = m_scene.getCamera();
    State& state = m_scene.getState();

    if (camera.getDirty()) {
        dirty = true;
    }

    if (state.getDirty()) {
        dirty = true;
    }

    if (dirty) {
        state.resetIterations();
    }

    state.nextIteration();
    buffers::Global<kernals::ConstantParams>::copyHToD(
        m_constantParams,
        kernals::toKernalConstantParams(camera, state, m_textures.getCount()));

    camera.setDirty(false);
    state.setDirty(false);
}

void PathTracer::launchKernal(hipFunction_t kernal)
{
    struct KernalArgs {
        kernals::KernalBuffers kbuffs;
    };

    KernalArgs args = {
        .kbuffs = {
            .bvh = {
                .tlasNodes = m_tlasNodes.getHipArray(),
                .blasNodes = m_blasNodes.getHipArray(),
                .normals = m_normals.getHipArray(),
                .normalIndices = m_normalIndices.getHipArray(),
                .uvs = m_uvs.getHipArray(),
                .uvIndices = m_uvIndices.getHipArray(),
                .transforms = m_transforms.getHipArray(),
            },
            .materials = m_materials.getHipArray(),
            .textures = m_textures.getHipArray(),
            .frameBuffer = m_targetBuffer.getBuffer().getHipArray(),
            .accumulationBuffer = m_targetBuffer.getAccumelationBuffer().getHipArray(),
            .rngSeedBuffer = m_targetBuffer.getRngStateBuffer().getHipArray(),
        },
    };

    size_t argSize = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &argSize,
        HIP_LAUNCH_PARAM_END
    };

    checkHipErrors(hipModuleLaunchKernel(
        kernal,
        m_targetBuffer.workgroups(), // gridDimX
        1, // gridDimY
        1, // gridDimZ
        buffers::workgroupSize, // blockDimX
        1, // blockDimY
        1, // blockDimZ
        0,
        nullptr,
        nullptr,
        (void**)&config));
}

Scene& PathTracer::getScene() noexcept
{
    return m_scene;
}

void PathTracer::getFrameBuffer(uint8_t* dst, size_t size, size_t* retSize)
{
    auto src = m_targetBuffer.getBuffer().getHipArray();
    if (dst == nullptr || size == 0) {
        if (retSize != nullptr) {
            *retSize = src.sizeInBytes();
        }
        return;
    }

    checkHipErrors(hipMemcpy(
        dst,
        src.ptr,
        size,
        hipMemcpyDeviceToHost));
}

void PathTracer::render()
{
    uint32_t iterations = m_scene.getState().getIterations();
    for (size_t i = 0; i < iterations; i++) {
        update();
        launchKernal(m_pathTracingKernal);
    }
    launchKernal(m_postProcessingKernal);
}
}