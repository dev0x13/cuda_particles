#include "particles.cuh"
#include "vec3.h"

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <curand_kernel.h>

#include <particle.h>
#include <scene_objects.h>

static int gridSize;
static int blockSize;

static Particle *hostParticles;
static Particle *gpuParticles;

struct DeviceSphere;
static DeviceSphere *gpuSceneSpheres;
static size_t numSpheresOnScene;

static size_t numParticles;

struct Scene {
    DeviceSphere *spheres;
    size_t numSpheres;

    Scene(DeviceSphere *spheres, size_t numSpheres) : spheres(spheres), numSpheres(numSpheres) {}
};

class DeviceVec {

public:
	// Member variables
	float x;
	float y;
	float z;

public:
	// Methods
	__host__ __device__ DeviceVec(float x, float y, float z) :
			x(x), y(y), z(z) {
	}

	__host__ __device__ DeviceVec(DeviceVec *v) :
			x(v->x), y(v->y), z(v->z) {
	}

	__host__ __device__ DeviceVec(Vec3 *v) :
			x(v->x), y(v->y), z(v->z) {
	}

	__device__ float lengthSquared() const {
		float sum = 0;
		sum += x * x;
		sum += y * y;
		sum += z * z;

		return sum;
	}

	__device__ float length() const {
		return sqrt(lengthSquared());
	}

	__device__ void toVec3(Vec3 *toFill) {
		toFill->x = x;
		toFill->y = y;
		toFill->z = z;
	}

	__device__ float dot(DeviceVec *other) const {
		return x * other->x + y * other->y + z * other->z;
	}

	__device__ DeviceVec normalised() const {
		float len = length();

		DeviceVec normalised(x / len, y / len, z / len);
		return normalised;
	}

	__device__ void reflect(DeviceVec *normal) {
		DeviceVec tmp = *normal * 2 * this->dot(normal);
		x -= tmp.x;
		y -= tmp.y;
		z -= tmp.z;
	}

	// Operators
	__device__ DeviceVec operator+(const DeviceVec& other) const {
		DeviceVec newVec(x + other.x, y + other.y, z + other.z);
		return newVec;
	}

	__device__ DeviceVec operator-(const DeviceVec& other) const {
		DeviceVec newVec(x - other.x, y - other.y, z - other.z);
		return newVec;
	}

	__device__ DeviceVec operator-(const float scalar) const {
		DeviceVec newVec(x - scalar, y - scalar, z - scalar);
		return newVec;
	}

	__device__ DeviceVec operator*(const DeviceVec& other) const {
		DeviceVec newVec(x * other.x, y * other.y, z * other.z);
		return newVec;
	}

	__device__ DeviceVec operator*(const float scalar) const {
		DeviceVec newVec(x * scalar, y * scalar, z * scalar);
		return newVec;
	}
};

struct DeviceSphere {

    DeviceSphere() : pos(0, 0, 0), radius(0), radius2(0) {}

    DeviceSphere(Sphere *sphere): pos(&sphere->pos), radius(sphere->radius), radius2(radius * radius) {}

    __device__ bool collides(const DeviceVec& point) const {
        return (point.x - pos.x) * (point.x - pos.x) +
               (point.y - pos.y) * (point.y - pos.y) +
               (point.z - pos.z) * (point.z - pos.z) <= radius2;
    }

    __device__ DeviceVec normal(const DeviceVec& point) const {
        return (point - pos).normalised();
    }

    DeviceVec pos;
    float radius;
    float radius2;
};

__device__ void emitParticle(Particle& particle, unsigned int seed) {
    curandState s;

    curand_init(clock64() + 100 * seed, 0, 0, &s);

    float radius = 0.05;

    float xPos, yPos, zPos;
    float xVel, yVel, zVel;

    yPos = 0.5f;
    float z, x;

    float sourceRadius2 = 0.1;

    x = curand_uniform(&s) - 0.5;
    if (curand_uniform(&s) > 0.5) {
        z = curand_uniform(&s) * sqrt(sourceRadius2 - x * x);
    } else {
        z = (curand_uniform(&s) - 1) * sqrt(sourceRadius2 - x * x);
    }

    xPos = x;
    zPos = z;

    xVel = 0;//(curand_uniform(&s) - 0.5) / 10000;
    yVel = curand_uniform(&s) / 10000;
    zVel = 0;//(curand_uniform(&s) - 0.5) / 10000;

    particle.position = Vec3(xPos, yPos, zPos);
    particle.velocity = Vec3(xVel, yVel, zVel);
    particle.radius = radius;
    particle.lifetime = curand_uniform(&s) + 0.9;
    particle.age = 0;
}

__global__ void firstEmitParticles(Particle *particles, size_t numParticles_) {
    int t_x = threadIdx.x;
    int b_x = blockIdx.x;
    int in_x = b_x * blockDim.x + t_x;

    if (in_x < numParticles_) {
        emitParticle(particles[in_x], in_x);
    }
}

// Device function for checking whether two given particles collide
__device__ bool particlesCollide(Particle *p1, Particle *p2) {
    // Find the vector between the two particles
    DeviceVec collideVec = DeviceVec(&(p2->position)) - DeviceVec(&(p1->position));
    // Find the combined radius of the two particles

    float radiuses = p1->radius + p2->radius;
    float collideDistSq = radiuses * radiuses;

    // Particles collide if the distance between them is less
    // than their combined radiuses
    return collideVec.lengthSquared() <= collideDistSq;
}

// Resolve a collision between two particles. Adapted from CUDA samples
__device__ void collide(Particle *p1, Particle *p2, DeviceVec *force) {
    DeviceVec posA(&p1->position);
    DeviceVec posB(&p2->position);
    DeviceVec velA(&p1->velocity);
    DeviceVec velB(&p2->velocity);

    // relative Position and velocity
    DeviceVec relPos = posB - posA;
    DeviceVec relVel = velB - velA;

    // Distance between the two particles
    float dist = relPos.length();
    // Minimum distance for these particles to be colliding
    float collideDist = p1->radius + p2->radius;

    DeviceVec norm = relPos.normalised();

    // New force is accumalated in the force parameter
    // spring force
    *force = *force - norm * 0.5f * (collideDist - dist);
    // damping force
    *force = *force + relVel * 0.02f;
}

__global__ void moveParticles(Particle * particles, size_t numParticles_, Scene scene) {
	int t_x = threadIdx.x;
	int b_x = blockIdx.x;
	int in_x = b_x * blockDim.x + t_x;

	float delta = 0.01;

    curandState s;

    curand_init(clock64() + 1000 * t_x, 0, 0, &s);

	if (in_x < numParticles_) {
		Particle thisParticle = particles[in_x];

		DeviceVec newPosD(&thisParticle.position);
		DeviceVec velD(&thisParticle.velocity);

        		 DeviceVec force(0, 0, 0);

		 for (int i = 0; i < numParticles_; i++) {
		 	if (i != in_x) { // Don't consider ourselves
		 		Particle other = particles[i];

		 		if (particlesCollide(&thisParticle, &other)) {
		 			collide(&thisParticle, &other, &force);
		 		}
		 	}
		 }

		 velD = velD + force;

		// Calculate our new desired position
		newPosD = newPosD + velD;

        DeviceVec normalD(0, 0, 0);

        for (size_t i = 0; i < scene.numSpheres; ++i) {
            if (scene.spheres[i].collides(newPosD)) {
                normalD = scene.spheres[i].normal(newPosD);
                velD.reflect(&normalD);
            }
        }

        velD.y -= delta * 0.1;

        //velD = velD + DeviceVec((curand_uniform(&s) - 0.5) / 1000, (curand_uniform(&s) - 0.5) / 1000, (curand_uniform(&s) - 0.5) / 1000);

		// Calculate the position after movement
		newPosD = DeviceVec(&thisParticle.position) + velD;

		newPosD.toVec3(&thisParticle.position);
		velD.toVec3(&thisParticle.velocity);

		// Move this particle
		thisParticle.age += delta;

		if (thisParticle.age >= thisParticle.lifetime) {
		    emitParticle(thisParticle, t_x);
		}

        particles[in_x] = thisParticle;
	}
}


/**
 * Compute the ideal grid size for the number of particles that we have
 */
void computeGridSize(int n, int blockSize, int &numBlocks, int &numThreads) {
	numThreads = min(blockSize, n);
	numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
}

void cuda_scene_add_spheres(Sphere* spheres, size_t numSpheres) {
    cudaMalloc((void**) &gpuSceneSpheres, numSpheres * sizeof(Sphere));

    numSpheresOnScene = numSpheres;

    DeviceSphere *deviceSpheres = new DeviceSphere[numSpheres];

    for (size_t i = 0; i < numSpheres; ++i) {
        deviceSpheres[i] = DeviceSphere(&spheres[i]);
    }

    cudaMemcpy(gpuSceneSpheres, deviceSpheres, numSpheres * sizeof(DeviceSphere), cudaMemcpyHostToDevice);
}

void cuda_init(Particle *particles, size_t numParticles_) {
    numParticles = numParticles_;

	const size_t particlesArraySize = sizeof(Particle) * numParticles;

    hostParticles = particles;

	cudaMalloc((void**) &gpuParticles, particlesArraySize);

    cudaMemcpy(gpuParticles, particles, particlesArraySize, cudaMemcpyHostToDevice);

	computeGridSize(numParticles, 256, gridSize, blockSize);

    firstEmitParticles<<<gridSize, blockSize>>>(gpuParticles, numParticles);
}

void cuda_particles_update() {
	moveParticles<<<gridSize, blockSize>>>(gpuParticles, numParticles, Scene(gpuSceneSpheres, numSpheresOnScene));

	cudaMemcpy(hostParticles, gpuParticles, sizeof(Particle) * numParticles, cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
}
