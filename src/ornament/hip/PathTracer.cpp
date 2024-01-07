#include <hip/hip_runtime.h>
#include <filesystem>

#include "Bvh.hpp"
#include "PathTracer.hpp"
#include "buffers.hpp"
#include "hip_helper.hpp"

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
    checkHipErrors(hipModuleGetFunction(&m_pathTracingKernal, m_module, "path_tracing_kernal"));
    checkHipErrors(hipModuleGetFunction(&m_postProcessingKernal, m_module, "post_processing_kernal"));
    checkHipErrors(hipModuleGetFunction(&m_pathTracingAndPostProcessingKernal, m_module, "path_tracing_and_post_processing_kernal"));

    m_targetBuffer = buffers::Target(m_scene.getState().getResolution());
    m_textures = buffers::Textures(bvh.getTextures(), prop.texturePitchAlignment);
    m_constantParams = buffers::Global<gpu_structs::ConstantParams>("constant_params", m_module);
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
    buffers::Global<gpu_structs::ConstantParams>::copyHToD(
        m_constantParams,
        gpu_structs::ConstantParams(camera, state, m_textures.getCount()));

    camera.setDirty(false);
    state.setDirty(false);
}

void PathTracer::launchKernal(hipFunction_t kernal)
{
    struct KernalArgs {
        struct {
            struct {
                buffers::HipArray<gpu_structs::BvhNode> tlas_nodes;
                buffers::HipArray<gpu_structs::BvhNode> blas_nodes;
                buffers::HipArray<gpu_structs::Normal> normals;
                buffers::HipArray<uint32_t> normal_indices;
                buffers::HipArray<gpu_structs::Uv> uvs;
                buffers::HipArray<uint32_t> uv_indices;
                buffers::HipArray<gpu_structs::Transform> transforms;
            } bvh;
            buffers::HipArray<gpu_structs::Material> materials;
            buffers::HipArray<hipTextureObject_t> textures;
            hipDeviceptr_t framebuffer;
            hipDeviceptr_t accumulation_buffer;
            hipDeviceptr_t rng_seed_buffer;
            uint32_t pixel_count;
        } kg;
    };

    KernalArgs args = {
        .kg = {
            .bvh = {
                .tlas_nodes = m_tlasNodes.getHipArray(),
                .blas_nodes = m_blasNodes.getHipArray(),
                .normals = m_normals.getHipArray(),
                .normal_indices = m_normalIndices.getHipArray(),
                .uvs = m_uvs.getHipArray(),
                .uv_indices = m_uvIndices.getHipArray(),
                .transforms = m_transforms.getHipArray(),
            },
            .materials = m_materials.getHipArray(),
            .textures = m_textures.getHipArray(),
            .framebuffer = m_targetBuffer.getBuffer().getHipArray().dptr,
            .accumulation_buffer = m_targetBuffer.getAccumelationBuffer().getHipArray().dptr,
            .rng_seed_buffer = m_targetBuffer.getRngStateBuffer().getHipArray().dptr,
            .pixel_count = m_targetBuffer.pixelCount(),
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
    if (dst == nullptr || size == 0)
    {
        if (retSize != nullptr)
        {
            *retSize = src.sizeInBytes();
        }
        return;
    } 

    checkHipErrors(hipMemcpy(
        dst,
        src.dptr,
        size,
        hipMemcpyDeviceToHost));
}

void PathTracer::render()
{
    uint32_t iterations = m_scene.getState().getIterations();
    if (iterations > 1) {
        for (size_t i = 0; i < iterations; i++) {
            update();
            launchKernal(m_pathTracingKernal);
        }
        launchKernal(m_postProcessingKernal);
    } else {
        update();
        launchKernal(m_pathTracingAndPostProcessingKernal);
    }
}
}