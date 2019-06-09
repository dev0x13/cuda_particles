#pragma once

#include <cmath>
#include <host_defines.h>

class DeviceVec3 {

public:
    float x;
    float y;
    float z;

public:
    __host__ __device__ DeviceVec3(float x, float y, float z) :
            x(x), y(y), z(z) {
    }

    __host__ __device__ DeviceVec3() :
            x(0), y(0), z(0) {
    }

    __device__ float lengthSquared() const {
        float sum = 0;
        sum += x * x;
        sum += y * y;
        sum += z * z;

        return sum;
    }

    __device__ float dot(DeviceVec3 *other) const {
        return x * other->x + y * other->y + z * other->z;
    }

    __device__ DeviceVec3 operator+(const DeviceVec3& other) const {
        DeviceVec3 newVec(x + other.x, y + other.y, z + other.z);

        return newVec;
    }

    __device__ DeviceVec3 operator-(const DeviceVec3& other) const {
        DeviceVec3 newVec(x - other.x, y - other.y, z - other.z);

        return newVec;
    }

    __device__ DeviceVec3 operator-(const float scalar) const {
        DeviceVec3 newVec(x - scalar, y - scalar, z - scalar);

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