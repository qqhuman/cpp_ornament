#pragma once

#include <iostream>

class Renderer
{
private:
    std::ostream& stream;
    /* data */
public:
    Renderer(std::ostream& stream, int width, int height);
    ~Renderer();

    void render();
    void getFrameBuffer(float *);
};
