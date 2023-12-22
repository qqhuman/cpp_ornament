set(HEADERS 
    ornament.h
)

SET(SOURCES 
    Renderer.cpp
)

add_library(ornament STATIC ${SOURCES} ${HEADERS})
target_include_directories(ornament PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/.)