#include "Renderer.h"

Renderer::Renderer(std::ostream& s, int width, int height) : stream(s)
{
    stream << "Renderer::constructor" << std::endl;
}

Renderer::~Renderer()
{
    stream << "Renderer::destructor" << std::endl;
}

void Renderer::render()
{
    stream << "Renderer::render" << std::endl;
}

void Renderer::getFrameBuffer(float *dst)
{
    stream << "Renderer::getFrameBuffer" << std::endl;
}
