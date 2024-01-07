#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "utils.hpp"
#include <stb_image.h>
#include <stb_image_write.h>
#include <stdexcept>

namespace utils {

StbImage loadImageFromFile(const char* filename, uint32_t forcedNumComponents)
{
    stbi_set_flip_vertically_on_load(true);
    if (stbi_is_hdr(filename)) {
        throw std::runtime_error("hdr loading is not implemented");
    } else if (stbi_is_16_bit(filename)) {
        throw std::runtime_error("16bit image loading is not implemented");
    } else {
        int x, y, ch;
        uint8_t* ptr = stbi_load(filename, &x, &y, &ch, forcedNumComponents);
        if (ptr == nullptr) {
            throw std::runtime_error("stbi_load failed");
        }

        uint32_t numComponents = forcedNumComponents == 0 ? ch : forcedNumComponents;
        uint32_t bytesPerComponent = 1;
        return StbImage {
            .data = ptr,
            .width = (uint32_t)x,
            .height = (uint32_t)y,
            .numComponents = numComponents,
            .bytesPerComponent = bytesPerComponent,
            .bytesPerRow = (uint32_t)x * numComponents * bytesPerComponent,
            .isHdr = false,
        };
    }
}

void freeImage(StbImage img)
{
    stbi_image_free(img.data);
}

void savePngImage(const char* filename, uint8_t* img, uint32_t width, uint32_t height, uint32_t numComponents, bool isHdr)
{
    uint8_t* dst = img;
    if (isHdr) {
        dst = new uint8_t[width * height * numComponents];
        float* hdrImg = (float*)img;
        for (size_t i = 0; i < width * height * numComponents; i++) {
            float val = floor(hdrImg[i] * 255.0f);
            dst[i] = std::max((uint8_t)0, std::min((uint8_t)255, (uint8_t)val));
        }
    }

    stbi_write_png(filename, width, height, numComponents, dst, width * numComponents);

    if (isHdr) {
        delete[] dst;
    }
}

}