set(HEADERS
    ../common/examples.hpp
    ../common/utils.hpp
    App.hpp
)
set(SOURCES
    ../common/examples.cpp
    ../common/utils.cpp
    App.cpp
)
add_executable(glfwapp ${SOURCES} ${HEADERS} main.cpp)

get_target_property(ORNAMENT_KERNALS_BINARY_DIR ornament_kernals BINARY_DIR)
_copy_files_to_target(glfwapp "${ORNAMENT_KERNALS_BINARY_DIR}/ornament_kernals.co")
_copy_files_to_target(glfwapp "${Vulkan_SHADERC_DLL}")


target_include_directories(glfwapp 
    PRIVATE ${Vulkan_INCLUDE_DIR}
    PRIVATE ${stb_SOURCE_DIR}
)

target_link_libraries(glfwapp 
    glfw
    ${Vulkan_LIBRARY}
    ${Vulkan_SHADERC_LIB}
    assimp::assimp
    ornament
)