include(FetchContent)
FetchContent_Declare(glm GIT_REPOSITORY https://github.com/g-truc/glm)
set(GLM_BUILD_TESTS OFF CACHE INTERNAL "Glm build tests")
set(GLM_BUILD_INSTALL OFF CACHE INTERNAL "Glm build install")
FetchContent_MakeAvailable(glm)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

find_path(HIP_INCLUDE_DIR NAME hip HINTS ${HIP_PATH} PATH_SUFFIXES include)
if(NOT HIP_INCLUDE_DIR)
    message(FATAL_ERROR "HIP include is missing")
endif()
find_library(HIP_LIBRARY amdhip64 HINTS ${HIP_PATH} PATH_SUFFIXES lib)
if(NOT HIP_LIBRARY)
    message(FATAL_ERROR "HIP amdhip64 is missing")
endif()

# for some reasons it's not working
# set(HIP_CXX_COMPILER "${HIP_PATH}/bin/clang++")
# list(APPEND CMAKE_MODULE_PATH  "${HIP_PATH}/cmake")
# find_package(HIP)
# if(HIP_FOUND)
#     message(STATUS "Found HIP: " ${HIP_VERSION})
#     find_library(HIP_LIBRARY amdhip64 HINTS ${HIP_PATH}/lib)
#     set(HIP_INCLUDE_DIR, ${HIP_PATH}/include)
# else()
#     message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
# endif()

set(HEADERS 
    hip/buffers.hpp
    hip/hip_helper.hpp
    hip/PathTracer.hpp
    math/Aabb.hpp
    math/math.hpp
    math/transform.hpp
    Bvh.hpp
    GpuStructs.hpp
    Camera.hpp
    ornament.hpp
    Scene.hpp
    State.hpp
)

SET(SOURCES 
    hip/PathTracer.cpp
    math/Aabb.cpp
    math/math.cpp
    math/transform.cpp
    Bvh.cpp
    Camera.cpp
    Scene.cpp
    State.cpp
)

add_subdirectory(hip/kernals)

add_library(ornament STATIC ${SOURCES} ${HEADERS})

add_dependencies(ornament ornament_kernals)

target_compile_definitions(ornament PUBLIC __HIP_PLATFORM_AMD__) 

target_include_directories(ornament 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/.
    PUBLIC ${HIP_INCLUDE_DIR})

target_link_libraries(ornament 
    PUBLIC glm::glm 
    PUBLIC ${HIP_LIBRARY})
