#pragma once

#include <glm/glm.hpp>

namespace ornament {

class State {
public:
    void setFlipY(bool flipY) noexcept;
    bool getFlipY() const noexcept;
    void setGamma(float gamma) noexcept;
    float getGamma() const noexcept;
    float getInvertedGamma() const noexcept;
    void setDepth(uint32_t depth) noexcept;
    uint32_t getDepth() const noexcept;
    void setIterations(uint32_t iterations) noexcept;
    uint32_t getIterations() const noexcept;
    void setResolution(const glm::uvec2& resolution) noexcept;
    glm::uvec2 getResolution() const noexcept;
    void setRayCastEpsilon(float rayCastEpsilon) noexcept;
    float getRayCastEpsilon() const noexcept;
    void nextIteration() noexcept;
    void resetIterations() noexcept;
    float getCurrentIteration() const noexcept;
    void setDirty(bool dirty) noexcept;
    bool getDirty() const noexcept;

private:
    bool m_dirty = true;
    glm::uvec2 m_resolution = glm::uvec2(500, 500);
    uint32_t m_depth = 10;
    bool m_flipY = false;
    float m_invertedGamma = 1.0f;
    uint32_t m_iterations = 1;
    float m_rayCastEpsilon = 0.001;
    float m_currentIteration = 0.0f;
};

}