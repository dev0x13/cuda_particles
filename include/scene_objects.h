#pragma once

#include <device_vec3.h>

#include <memory>

#include <GL/glew.h>
#include <vector_types.h>

enum SceneObjectType {
    SPHERE,
    PLANE,
    PARTICLE_SYSTEM
};

class SceneObject {
public:
    virtual void draw() const = 0;

    virtual SceneObjectType getType() const = 0;

    virtual void update() {}

    virtual ~SceneObject() {};
};

class Sphere : public SceneObject {
public:
    Sphere(
        const DeviceVec3& pos,
        float radius,
        int slices,
        int stacks,
        const GLfloat color_[4]
    ) :
        type_(SPHERE),
        pos(pos),
        radius(radius),
        radius2(radius * radius),
        slices(slices),
        stacks(stacks)
    {
        color[0] = color_[0];
        color[1] = color_[1];
        color[2] = color_[2];
        color[3] = color_[3];
    }

    void draw() const;

    /* Was intended to be used both on host and device

    __device__ bool collides(const Vec3& point) const {
        return (point.x - pos.x) * (point.x - pos.x) +
               (point.y - pos.y) * (point.y - pos.y) +
               (point.z - pos.z) * (point.z - pos.z) <= radius2;
    }

    __device__ Vec3 normal(const Vec3& point) const {
        Vec3 tmp = point - pos;

        return tmp / tmp.length();
    }
    */

    SceneObjectType getType() const {
        return type_;
    }

    ~Sphere() {};

    DeviceVec3 pos;

    const SceneObjectType type_;
    GLfloat color[4] = {1, 1, 1};
    const float radius;
    const float radius2;
    const int slices;
    const int stacks;
};

class Plane : public SceneObject {
public:
    Plane(
            const DeviceVec3& pos,
            float width,
            float length,
            const GLfloat color_[4]
    ) :
            type_(PLANE),
            pos(pos),
            width(width),
            length(length)
    {
        vertexes[0].x = pos.x - width / 2;
        vertexes[0].y = pos.y;
        vertexes[0].z = pos.z - length / 2;

        vertexes[1].x = pos.x + width / 2;
        vertexes[1].y = pos.y;
        vertexes[1].z = pos.z - length / 2;

        vertexes[2].x = pos.x + width / 2;
        vertexes[2].y = pos.y;
        vertexes[2].z = pos.z + length / 2;

        vertexes[3].x = pos.x - width / 2;
        vertexes[3].y = pos.y;
        vertexes[3].z = pos.z + length / 2;

        color[0] = color_[0];
        color[1] = color_[1];
        color[2] = color_[2];
        color[3] = color_[3];
    }

    void draw() const;

    /*
    __device__ bool collides(const Vec3& point) const {
        return false;
    }

    __device__ Vec3 normal(const Vec3& point) const {
        //return (vertexes[2] - vertexes[0], vertexes[1] - vertexes[0]).norm();
        return Vec3(0, 0, 0);
    }
    */

    SceneObjectType getType() const {
        return type_;
    }

    ~Plane() {};

    DeviceVec3 pos;

    const SceneObjectType type_;
    GLfloat color[4] = {1, 1, 1, 1};
    float3 vertexes[4];
    const float width;
    const float length;
};