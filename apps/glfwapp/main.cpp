#include <iostream>
#include "App.hpp"

int main(int argc, const char* argv[])
{
    App app;
    try {
        std::cout << "Glfw App started..." << std::endl;
        app.run();
        std::cout << "Glfw App finished..." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}