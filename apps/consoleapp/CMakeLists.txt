set(HEADERS 
    ../common/examples.hpp
    ../common/utils.hpp
)

set(SOURCES 
    ../common/examples.cpp
    ../common/utils.cpp
)

add_executable(consoleapp ${SOURCES} ${HEADERS} main.cpp)

get_target_property(ORNAMENT_KERNALS_BINARY_DIR ornament_kernals BINARY_DIR)
_copy_files_to_target(consoleapp "${ORNAMENT_KERNALS_BINARY_DIR}/ornament_kernals.co")

target_include_directories(consoleapp 
    PRIVATE ${stb_SOURCE_DIR}
)

target_link_libraries(consoleapp 
    assimp::assimp
    ornament
)