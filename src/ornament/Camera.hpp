#pragma once

#include <glm/glm.hpp>

namespace ornament {

class Camera {
public:
    Camera(const glm::vec3& lookFrom, const glm::vec3& lookAt, const glm::vec3& vup, float aspectRatio, float vfov, float aperture, float focusDist) noexcept;
    void setAsoectRatio(float aspectRatio) noexcept;
    float getAspectRatio() const noexcept;
    void setLookAt(const glm::vec3& lookFrom, const glm::vec3& lookAt, const glm::vec3& vup) noexcept;
    glm::vec3 getLookFrom() const noexcept;
    glm::vec3 getLookAt() const noexcept;
    glm::vec3 getVUp() const noexcept;
    glm::vec3 getU() const noexcept;
    glm::vec3 getV() const noexcept;
    glm::vec3 getW() const noexcept;
    glm::vec3 getLowerLeftCorner() const noexcept;
    glm::vec3 getHorizontal() const noexcept;
    glm::vec3 getVertical() const noexcept;
    float getLensRadius() const noexcept;
    void setDirty(bool dirty) noexcept;
    bool getDirty() const noexcept;

private:
    glm::vec3 m_origin;
    glm::vec3 m_lowerLeftCorner;
    glm::vec3 m_horizontal;
    glm::vec3 m_vertical;
    glm::vec3 m_u;
    glm::vec3 m_v;
    glm::vec3 m_w;
    float m_lensRadius;
    float m_vfov;
    float m_focusDist;
    float m_aspectRatio;
    glm::vec3 m_lookFrom;
    glm::vec3 m_lookAt;
    glm::vec3 m_vup;
    bool m_dirty;
};

}