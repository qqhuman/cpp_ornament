macro(_copy_files_to_target target thefiles)
    foreach (FFF ${thefiles} )
        add_custom_command(
            TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${FFF}
            $<TARGET_FILE_DIR:${target}>
            VERBATIM
        )
    endforeach()
endmacro()

include(FetchContent)

FetchContent_Declare(assimp GIT_REPOSITORY https://github.com/assimp/assimp)
set(ASSIMP_BUILD_SAMPLES OFF CACHE INTERNAL "Assimp build samples")
set(ASSIMP_BUILD_TESTS OFF CACHE INTERNAL "Assimp build tests")
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE INTERNAL "Assimp build assimp tools")
set(ASSIMP_BUILD_ZLIB ON CACHE INTERNAL "Assimp build zlib")
set(ASSIMP_NO_EXPORT ON CACHE INTERNAL "Assimp no export")
set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF CACHE INTERNAL "Assimp build all importers by default")
set(ASSIMP_BUILD_OBJ_IMPORTER TRUE CACHE INTERNAL "Assimp build obj importer")
set(ASSIMP_BUILD_FBX_IMPORTER TRUE CACHE INTERNAL "Assimp build fbx importer")
FetchContent_MakeAvailable(assimp)

FetchContent_Declare(stb GIT_REPOSITORY https://github.com/nothings/stb.git)
FetchContent_MakeAvailable(stb)

FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw)
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Glfw build examples")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Glfw build tests")
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Glfw build docs")
set(GLFW_INSTALL OFF CACHE INTERNAL "Glfw install")
FetchContent_MakeAvailable(glfw)

find_package(Vulkan REQUIRED)
get_filename_component(Vulkan_LIBRARY_DIR ${Vulkan_LIBRARY} DIRECTORY)
find_file(Vulkan_SHADERC_LIB NAMES shaderc_shared.lib HINTS ${Vulkan_LIBRARY_DIR})
find_file(Vulkan_SHADERC_DLL NAMES shaderc_shared.dll HINTS ${Vulkan_LIBRARY_DIR}/../Bin)
if(NOT Vulkan_SHADERC_DLL)
    message(FATAL_ERROR "Windows platform requires VulkanSDK with shaderc_shared.lib/dll (since SDK 1.2.135.0)")
endif()

add_subdirectory(consoleapp)
add_subdirectory(glfwapp)