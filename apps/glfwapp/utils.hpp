#pragma once

#include <cstdint>

namespace utils {

struct StbImage {
    uint8_t* data;
    uint32_t width;
    uint32_t height;
    uint32_t numComponents;
    uint32_t bytesPerComponent;
    uint32_t bytesPerRow;
    bool isHdr;
};

StbImage loadImageFromFile(const char* filename, uint32_t forcedNumComponents);
void freeImage(StbImage img);
void savePngImage(const char* filename, uint8_t* img, uint32_t width, uint32_t height, uint32_t numComponents, bool isHdr);

}