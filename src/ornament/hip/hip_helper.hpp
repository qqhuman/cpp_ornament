#pragma once

#include <hip/hip_runtime.h>

#define checkHipErrors(err) __checkHipErrors(err, __FILE__, __LINE__)

inline void __checkHipErrors(hipError_t err, const char* file, const int line)
{
    if (HIP_SUCCESS != err) {
        const char* errorStr = hipGetErrorString(err);
        fprintf(stderr,
            "checkHipErrors() HIP API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
        exit(EXIT_FAILURE);
    }
}

template <typename T>
inline void memcpyHToD(hipDeviceptr_t dst, const std::vector<T>& src)
{
    size_t sizeInBytes = src.size() * sizeof(T);
    checkHipErrors(hipMemcpy(
        dst,
        src.data(),
        sizeInBytes,
        hipMemcpyHostToDevice));
}