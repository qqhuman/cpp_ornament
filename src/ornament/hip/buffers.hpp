#pragma once

#include "hip_helper.hpp"
#include <algorithm>
#include <glm/glm.hpp>
#include <hip/hip_runtime.h>
#include <stdexcept>

namespace ornament::hip::buffers {

template <typename T>
struct HipArray {
    hipDeviceptr_t dptr = nullptr;
    uint32_t length = 0;

    size_t sizeInBytes()
    {
        return length * sizeof(T);
    }
};

template <typename T>
class Array {
public:
    Array() = default;
    Array(size_t length)
    {
        size_t sizeInBytes = length * sizeof(T);
        checkHipErrors(hipMalloc(&m_dptr, sizeInBytes));
        m_length = length;
    }

    Array(const std::vector<T>& hostArray)
        : Array(hostArray.size())
    {
        memcpyHToD(m_dptr, hostArray);
    }

    Array(Array&& other) noexcept
        : m_dptr(std::exchange(other.m_dptr, nullptr))
        , m_length(std::exchange(other.m_length, 0))
    {
    }

    ~Array()
    {
        if (m_dptr) {
            checkHipErrors(hipFree(m_dptr));
        }
    }

    Array& operator=(Array&& other)
    {
        if (this != &other) {
            if (m_dptr) {
                checkHipErrors(hipFree(m_dptr));
            }

            m_dptr = other.m_dptr;
            m_length = other.m_length;

            other.m_dptr = nullptr;
            other.m_length = 0;
        }

        return *this;
    }

    HipArray<T> getHipArray() const noexcept
    {
        return { .dptr = m_dptr, .length = m_length };
    }

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;

private:
    hipDeviceptr_t m_dptr = nullptr;
    uint32_t m_length = 0;
};

template <typename T>
class Global {
public:
    Global() = default;
    Global(const char* globalMemName, hipModule_t module)
    {
        size_t bytes;
        checkHipErrors(hipModuleGetGlobal(&m_dptr, &bytes, module, globalMemName));
        if (sizeof(T) != bytes) {
            throw std::runtime_error("[ornament]" + std::string(globalMemName) + " has wrong size..");
        }
    }

    Global(Global&& other) noexcept
        : m_dptr(std::exchange(other.m_dptr, nullptr))
    {
    }

    Global& operator=(Global&& other) noexcept
    {
        if (this != &other) {
            m_dptr = other.m_dptr;
            other.m_dptr = nullptr;
        }

        return *this;
    }

    template <typename T>
    inline static void copyHToD(const Global<T>& dst, T src)
    {
        checkHipErrors(hipMemcpy(
            dst.m_dptr,
            &src,
            sizeof(T),
            hipMemcpyHostToDevice));
    }

    Global(const Global&) = delete;
    Global& operator=(const Global&) = delete;

private:
    hipDeviceptr_t m_dptr = nullptr;
};

const uint32_t workgroupSize = 256;

static std::vector<uint32_t> rngSeed(int size)
{
    uint32_t n = 0;
    std::vector<uint32_t> v(size);
    std::generate(v.begin(), v.end(), [&n] { return n++; });
    return v;
}

class Target {
public:
    Target() = default;
    Target(const glm::uvec2& resolution)
        : m_resolution(resolution)
        , m_pixelCount(resolution.x * resolution.y)
    {
        m_buffer = Array<glm::vec4>(m_pixelCount);
        m_accumulationBuffer = Array<glm::vec4>(m_pixelCount);
        m_rngStateBuffer = Array(rngSeed(m_pixelCount));

        m_workgroups = m_pixelCount / workgroupSize;
        if (m_pixelCount % workgroupSize > 0) {
            m_workgroups += 1;
        }
    }

    Target(Target&& other) = default;
    Target& operator=(Target&& other) = default;

    uint32_t pixelCount() const noexcept
    {
        return m_pixelCount;
    }

    uint32_t workgroups() const noexcept
    {
        return m_workgroups;
    }

    const Array<uint32_t>& getRngStateBuffer() const noexcept
    {
        return m_rngStateBuffer;
    }

    const Array<glm::vec4>& getAccumelationBuffer() const noexcept
    {
        return m_accumulationBuffer;
    }

    const Array<glm::vec4>& getBuffer() const noexcept
    {
        return m_buffer;
    }

    Target(const Target&) = delete;
    Target& operator=(const Target&) = delete;

private:
    Array<glm::vec4> m_buffer;
    Array<glm::vec4> m_accumulationBuffer;
    Array<uint32_t> m_rngStateBuffer;
    glm::uvec2 m_resolution;
    uint32_t m_workgroups;
    uint32_t m_pixelCount;
};

class Textures {
public:
    Textures() = default;
    Textures(const std::vector<Texture*>& textures, size_t pitchAlignment)
    {
        m_textureObjects.reserve(textures.size());
        m_textureData.reserve(textures.size());

        for (auto txt : textures) {
            hipArray_Format format = txt->isHdr ? HIP_AD_FORMAT_FLOAT : HIP_AD_FORMAT_UNSIGNED_INT8;
            HIPfilter_mode filterMode = HIP_TR_FILTER_MODE_POINT;
            size_t srcPitch = txt->bytesPerRow;
            size_t dstPitch = alignUp(srcPitch, pitchAlignment);

            hipDeviceptr_t dptr;
            checkHipErrors(hipMalloc(&dptr, dstPitch * txt->height));
            m_textureData.push_back(dptr);
            hip_Memcpy2D param;
            memset(&param, 0, sizeof(param));
            param.dstMemoryType = hipMemoryTypeDevice;
            param.dstDevice = dptr;
            param.dstPitch = dstPitch;
            param.srcMemoryType = hipMemoryTypeHost;
            param.srcHost = txt->data.data();
            param.srcPitch = srcPitch;
            param.WidthInBytes = srcPitch;
            param.Height = txt->height;
            checkHipErrors(hipDrvMemcpy2DUnaligned(&param));

            HIP_RESOURCE_DESC resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = HIP_RESOURCE_TYPE_PITCH2D;
            resDesc.res.pitch2D.devPtr = dptr;
            resDesc.res.pitch2D.format = format;
            resDesc.res.pitch2D.numChannels = txt->numComponents;
            resDesc.res.pitch2D.height = txt->height;
            resDesc.res.pitch2D.width = txt->width;
            resDesc.res.pitch2D.pitchInBytes = dstPitch;

            HIP_TEXTURE_DESC texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.addressMode[0] = HIP_TR_ADDRESS_MODE_WRAP;
            texDesc.addressMode[1] = HIP_TR_ADDRESS_MODE_WRAP;
            texDesc.addressMode[2] = HIP_TR_ADDRESS_MODE_WRAP;
            texDesc.filterMode = filterMode;
            texDesc.flags = HIP_TRSF_NORMALIZED_COORDINATES;

            hipTextureObject_t texObj;
            hipTexObjectCreate(&texObj, &resDesc, &texDesc, nullptr);
            m_textureObjects.push_back(texObj);
        }

        m_deviceTextureObjects = Array(m_textureObjects);
    }

    Textures(Textures&& other) = default;
    Textures& operator=(Textures&& other) = default;

    HipArray<hipTextureObject_t> getHipArray() const noexcept
    {
        return m_deviceTextureObjects.getHipArray();
    }

    ~Textures()
    {
        for (auto to : m_textureObjects) {
            checkHipErrors(hipTexObjectDestroy(to));
        }

        for (auto td : m_textureData) {
            checkHipErrors(hipFree(td));
        }
    }

    uint32_t getCount() const noexcept
    {
        return m_textureObjects.size();
    }

    Textures(const Textures&) = delete;
    Textures& operator=(const Textures&) = delete;

private:
    std::vector<hipTextureObject_t> m_textureObjects;
    std::vector<hipDeviceptr_t> m_textureData;
    Array<hipTextureObject_t> m_deviceTextureObjects;

    static size_t alignUp(size_t offset, size_t alignment)
    {
        return (offset + alignment - 1) & ~(alignment - 1);
    }
};

}