#include <cuda/particle_manager.cuh>
#include <cuda/noise.cuh>

#include <vec3.h>
#include <config.h>

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand_kernel.h>

#include <particle.h>
#include <scene_objects.h>

static int numThreads;
static int numBlocks;

static Particle *hostParticles;
static Particle *gpuParticles;

struct DeviceSphere;
static DeviceSphere *gpuSceneSpheres;
static size_t numSpheresOnScene;

static Config conf;

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

    __host__ __device__ DeviceVec() :
            x(0), y(0), z(0) {
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

	__device__ DeviceVec operator+(const DeviceVec& other) const {
		DeviceVec newVec(x + other.x, y + other.y, z + other.z);
		return newVec;
	}

    __device__ DeviceVec operator+(const Vec3& other) const {
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
               (point.z - pos.z) * (point.z - pos.z) <= radius2 ;
    }

    DeviceVec pos;
    float radius;
    float radius2;
};

__device__ void emitParticle(Particle& particle, Config conf, unsigned int seed) {
    curandState s;

    curand_init(clock64() + 1000 * seed, 0, 0, &s);

    particle.position.x = 2 * (curand_uniform(&s) - 0.5) * conf.emitterRadius + conf.emitterPos.x;
    particle.position.z = 2 * (curand_uniform(&s) - 0.5) *
                          sqrt(conf.emitterRadius * conf.emitterRadius - particle.position.x * particle.position.x);
    particle.position.y = conf.emitterPos.y;

    particle.velocity.x = conf.particleInitialVelocity.x + (curand_uniform(&s) - 0.5) * conf.particleInitialVelocityNoiseFactor.x;
    particle.velocity.y = conf.particleInitialVelocity.y + (curand_uniform(&s) - 0.5) * conf.particleInitialVelocityNoiseFactor.y;
    particle.velocity.z = conf.particleInitialVelocity.z + (curand_uniform(&s) - 0.5) * conf.particleInitialVelocityNoiseFactor.z;

    particle.radius = conf.particleRadius;
    particle.lifetime = max(curand_uniform(&s) + conf.particleMinLifetime, conf.particleMaxLifetime);
    particle.age = 0;
}

__global__ void firstEmitParticles(Particle *particles, Config conf) {
    int t_x = threadIdx.x;
    int b_x = blockIdx.x;
    int in_x = b_x * blockDim.x + t_x;

    if (in_x < conf.numParticles) {
        emitParticle(particles[in_x], conf, in_x);
    }
}


__global__ void moveParticles(Particle * particles, Config conf, Scene scene, int64_t seed) {
	int t_x = threadIdx.x;
	int b_x = blockIdx.x;
	int in_x = b_x * blockDim.x + t_x;

    curandState s;

    curand_init(clock64() + 1000 * t_x, 0, 0, &s);

	if (in_x < conf.numParticles) {
		Particle thisParticle = particles[in_x];

		DeviceVec newPosD(&thisParticle.position);
		DeviceVec velD(&thisParticle.velocity);

		newPosD = newPosD + velD;

        DeviceVec force(0, 0, 0);
        DeviceVec normalD(0, 0, 0);

        for (size_t i = 0; i < scene.numSpheres; ++i) {

            DeviceVec relPos = scene.spheres[i].pos - DeviceVec(&thisParticle.position);

            float dist = relPos.length();
            float collideDist = thisParticle.radius + scene.spheres[i].radius + 0.1;

            if (dist < collideDist) {

                normalD = relPos * (1.0 / dist);

                DeviceVec tanVel = velD - normalD * velD.dot(&normalD);

                force = normalD * (collideDist - dist) * -conf.collisionSpring;
                force = force + velD * conf.collisionDamping;
                force = force + tanVel * conf.collisionShear;
            }

            velD = velD + force;
        }

        velD = velD + conf.gravity * conf.timestep;

        velD.z += NoiseCuda::simplexNoise(make_float3(newPosD.x, newPosD.y, -newPosD.z), 1, seed) * conf.particleVelocityLowFreqNoiseFactor.z;
        velD.y += NoiseCuda::simplexNoise(make_float3(newPosD.x, -newPosD.y, newPosD.z), 1, seed) * conf.particleVelocityLowFreqNoiseFactor.y;
        velD.x += NoiseCuda::simplexNoise(make_float3(-newPosD.x, newPosD.y, newPosD.z), 1, seed) * conf.particleVelocityLowFreqNoiseFactor.x;

        velD.z += NoiseCuda::simplexNoise(make_float3(newPosD.x, newPosD.y, -newPosD.z), 100, seed) * conf.particleVelocityHighFreqNoiseFactor.z;
        velD.y += NoiseCuda::simplexNoise(make_float3(newPosD.x, -newPosD.y, newPosD.z), 100, seed) * conf.particleVelocityHighFreqNoiseFactor.y;
        velD.x += NoiseCuda::simplexNoise(make_float3(-newPosD.x, newPosD.y, newPosD.z), 100, seed) * conf.particleVelocityHighFreqNoiseFactor.x;

		newPosD = DeviceVec(&thisParticle.position) + velD;

		newPosD.toVec3(&thisParticle.position);

		velD.toVec3(&thisParticle.velocity);

		thisParticle.age += conf.timestep;

		if (thisParticle.age >= thisParticle.lifetime) {
		    emitParticle(thisParticle, conf, in_x);
		}

        particles[in_x] = thisParticle;
	}
}


void ParticleManagerCuda::AddSpheresToScene(Sphere** spheres, const size_t numSpheres) {
    cudaMalloc((void**) &gpuSceneSpheres, numSpheres * sizeof(Sphere));

    numSpheresOnScene = numSpheres;

    DeviceSphere *deviceSpheres = new DeviceSphere[numSpheres];

    for (size_t i = 0; i < numSpheres; ++i) {
        deviceSpheres[i] = DeviceSphere(spheres[i]);
    }

    cudaMemcpy(gpuSceneSpheres, deviceSpheres, numSpheres * sizeof(DeviceSphere), cudaMemcpyHostToDevice);
}

void ParticleManagerCuda::Init(Particle *particles, const Config& conf_) {
    conf = conf_;

	const size_t particlesArraySize = sizeof(Particle) * conf.numParticles;

    hostParticles = particles;

	cudaMalloc((void**) &gpuParticles, particlesArraySize);

    cudaMemcpy(gpuParticles, particles, particlesArraySize, cudaMemcpyHostToDevice);

    numThreads = min((size_t) 256, conf.numParticles);
    numBlocks = (conf.numParticles % numThreads != 0) ? (conf.numParticles / numThreads + 1) : (conf.numParticles / numThreads);

    firstEmitParticles<<<numThreads, numBlocks>>>(gpuParticles, conf);
}

#include <chrono>

void ParticleManagerCuda::Update() {
	moveParticles<<<numThreads, numBlocks>>>(gpuParticles, conf, Scene(gpuSceneSpheres, numSpheresOnScene), std::chrono::system_clock::now().time_since_epoch().count());

	cudaMemcpy(hostParticles, gpuParticles, sizeof(Particle) * conf.numParticles, cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
}
