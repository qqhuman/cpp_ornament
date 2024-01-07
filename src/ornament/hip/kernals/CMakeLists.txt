if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

find_file(HIP_HIPCC_EXECUTABLE hipcc HINTS ${HIP_PATH} PATH_SUFFIXES bin)
if(NOT HIP_HIPCC_EXECUTABLE)
    message(FATAL_ERROR "hipcc is missing")
endif()

set(HEADERS 
    array.hip.h
    bvh.hip.h
    camera.hip.h
    common.hip.h
    constant_params.hip.h
    hitrecord.hip.h
    kernal_params.hip.h
    material.hip.h
    random.hip.h
    ray.hip.h
    transform.hip.h
    vec_math.hip.h
)

SET(SOURCES 
    pathtracer.hip.cpp
)

# set(ORNAMENT_HIP_BINARIES_ARCH gfx900 gfx90c gfx902 gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1034 gfx1035 gfx1100 gfx1101 gfx1102 CACHE STRING "AMD HIP architectures to build binaries for")
# gfx1030 RX 6800XT
# gfx1031 RX 6800M
# gfx90c APU 5900HX
set(ORNAMENT_HIP_BINARIES_ARCH gfx90c gfx1030 gfx1031 CACHE STRING "AMD HIP architectures to build binaries for")

set(ORNAMENT_TARGET_GPUS)
foreach(arch ${ORNAMENT_HIP_BINARIES_ARCH})
    list(APPEND ORNAMENT_TARGET_GPUS "--offload-arch=${arch}")
endforeach()

set(SOURCES_WITH_FULL_PATH)
foreach(source ${SOURCES})
    list(APPEND SOURCES_WITH_FULL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${source})
endforeach()

add_custom_target(
    ornament_kernals
    COMMAND  ${HIP_HIPCC_EXECUTABLE} --genco ${ORNAMENT_TARGET_GPUS} ${SOURCES_WITH_FULL_PATH} -o ornament_kernals.co
    COMMENT "ornament_kernals generated"
)