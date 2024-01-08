#include <iostream>
#include <filesystem>
#include <ornament.hpp>

#include "../common/examples.hpp"
#include "../common/utils.hpp"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

int main(int argc, const char* argv[])
{
    std::cout << "Console App started..." << std::endl;
    ornament::Scene scene = examples::lucy_and_spheres_with_textures((float)WIDTH / HEIGHT);
    scene.getState().setResolution({ WIDTH, HEIGHT });
    scene.getState().setDepth(10);
    scene.getState().setIterations(250);
    scene.getState().setGamma(2.2f);
    scene.getState().setFlipY(true);
    std::filesystem::path exeDirPath = std::filesystem::path(argv[0]).parent_path();
    ornament::hip::PathTracer pathTracer(
        std::move(scene),
        exeDirPath.string().c_str());
    pathTracer.render();
    pathTracer.render();
    pathTracer.render();
    pathTracer.render();

    {
        size_t size;
        pathTracer.getFrameBuffer(nullptr, 0, &size);
        uint8_t* img = new uint8_t[size];
        pathTracer.getFrameBuffer(img, size, nullptr);
        auto resultPath = exeDirPath / "result.png";
        std::filesystem::remove(resultPath);
        utils::savePngImage(resultPath.string().c_str(), img, WIDTH, HEIGHT, 4, true);
        delete[] img;
    }
    std::cout << "Console App finished..." << std::endl;
    return 0;
}