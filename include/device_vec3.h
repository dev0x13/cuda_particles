#pragma once

#include <cmath>
#include <host_defines.h>

class DeviceVec3 {

public:
    float x;
    float y;
    float z;

public:
    __device__ DeviceVec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __device__ DeviceVec3() : x(0), y(0), z(0) {}

    __device__ float length() const {
        return sqrt(x * x + y * y + z * z);
    }

    __device__ float dot(DeviceVec3 *other) const {
        return x * other->x + y * other->y + z * other->z;
    }

    __device__ DeviceVec3 operator+(const DeviceVec3& other) const {
        DeviceVec3 newVec(x + other.x, y + other.y, z + other.z);

        return newVec;
    }

    __device__ DeviceVec3& operator+=(const DeviceVec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;

        return *this;
    }

    __device__ DeviceVec3 operator-(const DeviceVec3& other) const {
        DeviceVec3 newVec(x - other.x, y - other.y, z - other.z);

        return newVec;
    }

    __device__ DeviceVec3 operator*(const DeviceVec3& other) const {
        DeviceVec3 newVec(x * other.x, y * other.y, z * other.z);

        return newVec;
    }

    __device__ DeviceVec3 operator*(const float scalar) const {
        DeviceVec3 newVec(x * scalar, y * scalar, z * scalar);

        return newVec;
    }
};