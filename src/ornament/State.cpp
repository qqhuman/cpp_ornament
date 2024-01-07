#include "State.hpp"

namespace ornament {
void State::setFlipY(bool flipY) noexcept
{
    m_flipY = flipY;
    setDirty(true);
}

bool State::getFlipY() const noexcept
{
    return m_flipY;
}

void State::setGamma(float gamma) noexcept
{
    m_invertedGamma = 1.0f / gamma;
    setDirty(true);
}

float State::getGamma() const noexcept
{
    return 1.0f / m_invertedGamma;
}

float State::getInvertedGamma() const noexcept
{
    return m_invertedGamma;
}

void State::setDepth(uint32_t depth) noexcept
{
    m_depth = depth;
    setDirty(true);
}

uint32_t State::getDepth() const noexcept
{
    return m_depth;
}

void State::setIterations(uint32_t iterations) noexcept
{
    m_iterations = iterations;
    setDirty(true);
}

uint32_t State::getIterations() const noexcept
{
    return m_iterations;
}

void State::setResolution(const glm::uvec2& resolution) noexcept
{
    m_resolution = resolution;
    setDirty(true);
}

glm::uvec2 State::getResolution() const noexcept
{
    return m_resolution;
}

void State::setRayCastEpsilon(float rayCastEpsilon) noexcept
{
    m_rayCastEpsilon = rayCastEpsilon;
    setDirty(true);
}

float State::getRayCastEpsilon() const noexcept
{
    return m_rayCastEpsilon;
}

void State::nextIteration() noexcept
{
    m_currentIteration += 1.0f;
}

void State::resetIterations() noexcept
{
    m_currentIteration = 0.0f;
}

float State::getCurrentIteration() const noexcept
{
    return m_currentIteration;
}

void State::setDirty(bool dirty) noexcept
{
    m_dirty = dirty;
}

bool State::getDirty() const noexcept
{
    return m_dirty;
}

}