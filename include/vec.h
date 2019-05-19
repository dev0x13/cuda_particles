#pragma once

#include <cmath>
#include <host_defines.h>

struct Vec3
{
	float x;
	float y;
	float z;

	__device__ Vec3(float x, float y, float z): x(x), y(y), z(z) {}

	__device__ Vec3() {};
	
	float length() {
        float sum = 0;
        sum += x * x;
        sum += y * y;
        sum += z * z;

        return sqrt(sum);
	}

    __device__ Vec3 operator+(const Vec3& other)
	{
		Vec3 newVec(x + other.x, y + other.y, z + other.z);
		return newVec;
	}

    __device__ Vec3 operator-(const Vec3& other)
	{
		Vec3 newVec(x - other.x, y - other.y, z - other.z);
		return newVec;
	}

    __device__ Vec3 operator*(const Vec3& other)
	{
		Vec3 newVec(x * other.x, y * other.y, z * other.z);
		return newVec;
	}
};
