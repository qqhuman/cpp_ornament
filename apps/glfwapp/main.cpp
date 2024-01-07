#include <iostream>
#include <memory>
#include <ornament.hpp>
#include <filesystem>

#include "App.hpp"
#include "examples.hpp"
#include "utils.hpp"

int main(int argc, const char * argv[])
{
    //App app;
    try {
        auto s = sizeof(glm::mat4);
        std::cout << "Glfw App started..." << std::endl;
        ornament::Scene scene = examples::spheres((float)WIDTH / HEIGHT);
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
        // app.run();
        std::cout << "Glfw App finished..." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}