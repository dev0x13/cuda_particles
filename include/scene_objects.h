#pragma once

#include "vec3.h"
#include <memory>

enum SceneObjectType {
    SPHERE
};

class SceneObject {
public:
    virtual void draw() const = 0;

    __device__ virtual bool collides(const Vec3& point) const = 0;

    __device__ virtual Vec3 normal(const Vec3& point) const = 0;

    virtual SceneObjectType type() const = 0;

    virtual ~SceneObject() {};
};

class Sphere : public SceneObject {
public:
    Sphere(
        const Vec3& pos,
        float radius,
        int slices,
        int stacks
    ) :
        type_(SPHERE),
        pos(pos),
        radius(radius),
        radius2(radius * radius),
        slices(slices),
        stacks(stacks) {}

    void draw() const;

    __device__ bool collides(const Vec3& point) const {
        return (point.x - pos.x) * (point.x - pos.x) +
               (point.y - pos.y) * (point.y - pos.y) +
               (point.z - pos.z) * (point.z - pos.z) <= radius2;
    }

    __device__ Vec3 normal(const Vec3& point) const {
        Vec3 tmp = point - pos;

        return tmp / tmp.length();
    }

    SceneObjectType type() const {
        return type_;
    }

    ~Sphere() {};

    Vec3 pos;

    const SceneObjectType type_;
    const float radius;
    const float radius2;
    const int slices;
    const int stacks;
};