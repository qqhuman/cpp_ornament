#include "Camera.hpp"

namespace ornament {

Camera::Camera(const glm::vec3& lookFrom, const glm::vec3& lookAt, const glm::vec3& vup, float aspectRatio, float vfov, float aperture, float focusDist) noexcept
{
    float theta = glm::radians(vfov);
    float h = std::tanf(theta / 2.0f);
    float viewportHeight = 2.0f * h;
    float viewportWidth = aspectRatio * viewportHeight;

    glm::vec3 w = glm::normalize(lookFrom - lookAt);
    glm::vec3 u = glm::normalize(glm::cross(vup, w));
    glm::vec3 v = glm::cross(w, u);

    glm::vec3 origin = lookFrom;
    glm::vec3 horizontal = focusDist * viewportWidth * u;
    glm::vec3 vertical = focusDist * viewportHeight * v;
    glm::vec3 lowerLeftCorner = origin - horizontal / 2.0f - vertical / 2.0f - focusDist * w;

    float lensRadius = aperture / 2.0f;

    m_origin = origin;
    m_lowerLeftCorner = lowerLeftCorner;
    m_horizontal = horizontal;
    m_vertical = vertical;
    m_u = u;
    m_v = v;
    m_w = w;
    m_lensRadius = lensRadius;
    m_focusDist = focusDist;
    m_vfov = vfov;
    m_aspectRatio = aspectRatio;
    m_lookFrom = lookFrom;
    m_lookAt = lookAt;
    m_vup = vup;
    m_dirty = true;
}

void Camera::setAsoectRatio(float aspectRatio) noexcept
{
    *this = Camera(m_lookFrom, m_lookAt, m_vup, aspectRatio, m_vfov, 2.0f * m_lensRadius, m_focusDist);
}

float Camera::getAspectRatio() const noexcept
{
    return m_aspectRatio;
}

void Camera::setLookAt(const glm::vec3& lookFrom, const glm::vec3& lookAt, const glm::vec3& vup) noexcept
{
    *this = Camera(lookFrom, lookAt, vup, m_aspectRatio, m_vfov, 2.0f * m_lensRadius, m_focusDist);
}

glm::vec3 Camera::getLookFrom() const noexcept
{
    return m_lookFrom;
}

glm::vec3 Camera::getLookAt() const noexcept
{
    return m_lookAt;
}

glm::vec3 Camera::getVUp() const noexcept
{
    return m_vup;
}

glm::vec3 Camera::getU() const noexcept
{
    return m_u;
}

glm::vec3 Camera::getV() const noexcept
{
    return m_v;
}

glm::vec3 Camera::getW() const noexcept
{
    return m_w;
}

glm::vec3 Camera::getLowerLeftCorner() const noexcept
{
    return m_lowerLeftCorner;
}

glm::vec3 Camera::getHorizontal() const noexcept
{
    return m_horizontal;
}

glm::vec3 Camera::getVertical() const noexcept
{
    return m_vertical;
}

float Camera::getLensRadius() const noexcept
{
    return m_lensRadius;
}

void Camera::setDirty(bool dirty) noexcept
{
    m_dirty = dirty;
}

bool Camera::getDirty() const noexcept
{
    return m_dirty;
}

}