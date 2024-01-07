#include "Aabb.hpp"
#include "transform.hpp"

namespace ornament::math {

Aabb::Aabb() noexcept
    : m_min(std::numeric_limits<float>::infinity())
    , m_max(-std::numeric_limits<float>::infinity())
{
}

Aabb::Aabb(const glm::vec3& min, const glm::vec3& max) noexcept
    : m_min(min)
    , m_max(max)
{
}

glm::vec3 Aabb::min() const noexcept
{
    return m_min;
}

glm::vec3 Aabb::max() const noexcept
{
    return m_max;
}

void Aabb::grow(const glm::vec3& p) noexcept
{
    m_min = glm::min(m_min, p);
    m_max = glm::max(m_max, p);
}

Aabb transform(const glm::mat4& m, const Aabb& aabb)
{
    glm::vec3 p0 = aabb.min();
    glm::vec3 p1(aabb.max()[0], aabb.min()[1], aabb.min()[2]);
    glm::vec3 p2(aabb.min()[0], aabb.max()[1], aabb.min()[2]);
    glm::vec3 p3(aabb.min()[0], aabb.min()[1], aabb.max()[2]);
    glm::vec3 p4(aabb.min()[0], aabb.max()[1], aabb.max()[2]);
    glm::vec3 p5(aabb.max()[0], aabb.max()[1], aabb.min()[2]);
    glm::vec3 p6(aabb.max()[0], aabb.min()[1], aabb.max()[2]);
    glm::vec3 p7 = aabb.max();

    p0 = transformPoint(m, p0);
    p1 = transformPoint(m, p1);
    p2 = transformPoint(m, p2);
    p3 = transformPoint(m, p3);
    p4 = transformPoint(m, p4);
    p5 = transformPoint(m, p5);
    p6 = transformPoint(m, p6);
    p7 = transformPoint(m, p7);

    Aabb result(p0, p0);
    result.grow(p1);
    result.grow(p2);
    result.grow(p3);
    result.grow(p4);
    result.grow(p5);
    result.grow(p6);
    result.grow(p7);
    return result;
}

}
