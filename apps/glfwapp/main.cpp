#include <iostream>
#include <GLFW/glfw3.h>
#include <ornament.h>

int main()
{
    Renderer r(std::cout, 1000, 1000);
    r.render();
    /* Init GLFW */
    if (!glfwInit())
        exit(EXIT_FAILURE);

    GLFWwindow *window;
    window = glfwCreateWindow(400, 400, "Glfw App", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    /* Main loop */
    for (;;)
    {
        /* Swap buffers */
        glfwSwapBuffers(window);
        glfwPollEvents();

        /* Check if we are still running */
        if (glfwWindowShouldClose(window))
            break;
    }
    return EXIT_SUCCESS;
}