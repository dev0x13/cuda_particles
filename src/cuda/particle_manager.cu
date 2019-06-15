#include <cuda/particle_manager.cuh>

#include <device_vec3.h>
#include <config.h>

#include <math.h>
#include <chrono>

#include <cuda.h>
#include <curand_kernel.h>

#include <particle.h>
#include <scene_objects.h>

/********************
 * Noise Generation *
 ********************/

// CUDA Simplex noise implementation taken from https://github.com/covexp/cuda-noise

__device__ __constant__ float gradMap[12][3] = { { 1.0f, 1.0f, 0.0f },{ -1.0f, 1.0f, 0.0f },{ 1.0f, -1.0f, 0.0f },{ -1.0f, -1.0f, 0.0f },
                                                 { 1.0f, 0.0f, 1.0f },{ -1.0f, 0.0f, 1.0f },{ 1.0f, 0.0f, -1.0f },{ -1.0f, 0.0f, -1.0f },
                                                 { 0.0f, 1.0f, 1.0f },{ 0.0f, -1.0f, 1.0f },{ 0.0f, 1.0f, -1.0f },{ 0.0f, -1.0f, -1.0f } };

// Hashing function (used for fast on-device pseudorandom numbers for randomness in noise)
__device__ unsigned int hash(unsigned int seed) {
    seed = (seed + 0x7ed55d16) + (seed << 12);
    seed = (seed ^ 0xc761c23c) ^ (seed >> 19);
    seed = (seed + 0x165667b1) + (seed << 5);
    seed = (seed + 0xd3a2646c) ^ (seed << 9);
    seed = (seed + 0xfd7046c5) + (seed << 3);
    seed = (seed ^ 0xb55a4f09) ^ (seed >> 16);

    return seed;
}

// Random value for simplex noise [0, 255]
__device__ unsigned char calcPerm(int p) {
    return (unsigned char)(hash(p));
}

// Random value for simplex noise [0, 11]
__device__ unsigned char calcPerm12(int p) {
    return (unsigned char)(hash(p) % 12);
}

__device__ float dot(float g[3], float x, float y, float z) {
    return g[0] * x + g[1] * y + g[2] * z;
}

// Simplex noise adapted from Java code by Stefan Gustafson and Peter Eastman
__device__ float simplexNoise(float3 pos, float scale, int seed) {
    float xin = pos.x * scale;
    float yin = pos.y * scale;
    float zin = pos.z * scale;

    // Skewing and unskewing factors for 3 dimensions
    float F3 = 1.0f / 3.0f;
    float G3 = 1.0f / 6.0f;

    // Noise contributions from the four corners
    float n0, n1, n2, n3;

    // Skew the input space to determine which simplex cell we're in
    float s = (xin + yin + zin) * F3;

    int i = floorf(xin + s);
    int j = floorf(yin + s);
    int k = floorf(zin + s);

    float t = (i + j + k) * G3;

    // Unskew the cell origin back to (x,y,z) space
    float X0 = i - t;
    float Y0 = j - t;
    float Z0 = k - t;

     // The x,y,z distances from the cell origin
    float x0 = xin - X0;
    float y0 = yin - Y0;
    float z0 = zin - Z0;

    // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    // Determine which simplex we are in.

    // Offsets for second corner of simplex in (i,j,k) coords
    int i1, j1, k1;

     // Offsets for third corner of simplex in (i,j,k) coords
    int i2, j2, k2;

    if (x0 >= y0) {
        if (y0 >= z0)      { i1 = 1.0f; j1 = 0.0f; k1 = 0.0f; i2 = 1.0f; j2 = 1.0f; k2 = 0.0f; } // X Y Z order
        else if (x0 >= z0) { i1 = 1.0f; j1 = 0.0f; k1 = 0.0f; i2 = 1.0f; j2 = 0.0f; k2 = 1.0f; } // X Z Y order
        else               { i1 = 0.0f; j1 = 0.0f; k1 = 1.0f; i2 = 1.0f; j2 = 0.0f; k2 = 1.0f; } // Z X Y order
    }
    else { // x0<y0
        if (y0 < z0)       { i1 = 0.0f; j1 = 0.0f; k1 = 1.0f; i2 = 0.0f; j2 = 1; k2 = 1.0f;    } // Z Y X order
        else if (x0 < z0)  { i1 = 0.0f; j1 = 1.0f; k1 = 0.0f; i2 = 0.0f; j2 = 1.0f; k2 = 1.0f; } // Y Z X order
        else               { i1 = 0.0f; j1 = 1.0f; k1 = 0.0f; i2 = 1.0f; j2 = 1.0f; k2 = 0.0f; } // Y X Z order
    }

    // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
    // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
    // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
    // c = 1/6.

    // Offsets for second corner in (x,y,z) coords
    float x1 = x0 - i1 + G3;
    float y1 = y0 - j1 + G3;
    float z1 = z0 - k1 + G3;

    // Offsets for third corner in (x,y,z) coords
    float x2 = x0 - i2 + 2.0f * G3;
    float y2 = y0 - j2 + 2.0f * G3;
    float z2 = z0 - k2 + 2.0f * G3;

    // Offsets for last corner in (x,y,z) coords
    float x3 = x0 - 1.0f + 3.0f * G3;
    float y3 = y0 - 1.0f + 3.0f * G3;
    float z3 = z0 - 1.0f + 3.0f * G3;

    // Work out the hashed gradient indices of the four simplex corners
    int ii = i & 255;
    int jj = j & 255;
    int kk = k & 255;

    int gi0 = calcPerm12(seed + ii + calcPerm(seed + jj + calcPerm(seed + kk)));
    int gi1 = calcPerm12(seed + ii + i1 + calcPerm(seed + jj + j1 + calcPerm(seed + kk + k1)));
    int gi2 = calcPerm12(seed + ii + i2 + calcPerm(seed + jj + j2 + calcPerm(seed + kk + k2)));
    int gi3 = calcPerm12(seed + ii + 1 + calcPerm(seed + jj + 1 + calcPerm(seed + kk + 1)));

    // Calculate the contribution from the four corners
    float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;

    if (t0 < 0.0f) {
        n0 = 0.0f;
    } else {
        t0 *= t0;
        n0 = t0 * t0 * dot(gradMap[gi0], x0, y0, z0);
    }

    float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;

    if (t1 < 0.0f) {
        n1 = 0.0f;
    } else {
        t1 *= t1;
        n1 = t1 * t1 * dot(gradMap[gi1], x1, y1, z1);
    }

    float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
    if (t2 < 0.0f) {
        n2 = 0.0f;
    } else {
        t2 *= t2;
        n2 = t2 * t2 * dot(gradMap[gi2], x2, y2, z2);
    }

    float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
    if (t3 < 0.0f) {
        n3 = 0.0f;
    } else {
        t3 *= t3;
        n3 = t3 * t3 * dot(gradMap[gi3], x3, y3, z3);
    }

    // Add contributions from each corner to get the final noise value.
    // The result is scaled to stay just inside [-1,1]
    return 32.0f * (n0 + n1 + n2 + n3);
}

/********************
 * Particle Manager *
 ********************/

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

struct DeviceSphere {

    DeviceSphere() : pos(0, 0, 0), radius(0), radius2(0) {}

    DeviceSphere(Sphere *sphere): pos(sphere->pos), radius(sphere->radius), radius2(radius * radius) {}

    __device__ bool collides(const DeviceVec3& point) const {
        return (point.x - pos.x) * (point.x - pos.x) +
               (point.y - pos.y) * (point.y - pos.y) +
               (point.z - pos.z) * (point.z - pos.z) <= radius2 ;
    }

    DeviceVec3 pos;
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

__global__ void __launch_bounds__(128, 8) moveParticles(Particle * particles, Config conf, Scene scene, int64_t seed) {
    int t_x = threadIdx.x;
    int b_x = blockIdx.x;
    int in_x = b_x * blockDim.x + t_x;

    curandState s;

    curand_init(clock64() + 1000 * t_x, 0, 0, &s);

    if (in_x < conf.numParticles) {
        Particle thisParticle = particles[in_x];

        DeviceVec3 newPosD(thisParticle.position);

        newPosD = newPosD + thisParticle.velocity;

        DeviceVec3 force(0, 0, 0);

        for (size_t i = 0; i < scene.numSpheres; ++i) {

            DeviceVec3 relPos = scene.spheres[i].pos - thisParticle.position;

            float dist = relPos.length();
            float collideDist = thisParticle.radius + scene.spheres[i].radius + 0.1;

            if (dist < collideDist) {
                DeviceVec3 normalD = relPos * (1.0 / dist);

                force = normalD * (collideDist - dist) * -conf.collisionSpring;
                force = force + thisParticle.velocity * conf.collisionDamping;
                force = force + (thisParticle.velocity - normalD * thisParticle.velocity.dot(&normalD)) * conf.collisionShear;

                thisParticle.velocity += force;
            }
        }

        thisParticle.velocity += conf.gravity * conf.timestep;

        // Low frequency noise
        thisParticle.velocity.z += simplexNoise(make_float3(newPosD.x, newPosD.y, -newPosD.z), 1, seed) * conf.particleVelocityLowFreqNoiseFactor.z;
        thisParticle.velocity.y += simplexNoise(make_float3(newPosD.x, -newPosD.y, newPosD.z), 1, seed) * conf.particleVelocityLowFreqNoiseFactor.y;
        thisParticle.velocity.x += simplexNoise(make_float3(-newPosD.x, newPosD.y, newPosD.z), 1, seed) * conf.particleVelocityLowFreqNoiseFactor.x;

        // High frequency noise
        thisParticle.velocity.z += simplexNoise(make_float3(newPosD.x, newPosD.y, -newPosD.z), 100, seed) * conf.particleVelocityHighFreqNoiseFactor.z;
        thisParticle.velocity.y += simplexNoise(make_float3(newPosD.x, -newPosD.y, newPosD.z), 100, seed) * conf.particleVelocityHighFreqNoiseFactor.y;
        thisParticle.velocity.x += simplexNoise(make_float3(-newPosD.x, newPosD.y, newPosD.z), 100, seed) * conf.particleVelocityHighFreqNoiseFactor.x;

        thisParticle.position += thisParticle.velocity;

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

    numBlocks = min((size_t) 128, conf.numParticles);
    numThreads = (conf.numParticles % numBlocks != 0) ? (conf.numParticles / numBlocks + 1) : (conf.numParticles / numBlocks);

    firstEmitParticles<<<numThreads, numBlocks>>>(gpuParticles, conf);
}

void ParticleManagerCuda::Update() {
    moveParticles<<<numThreads, numBlocks>>>(gpuParticles, conf, Scene(gpuSceneSpheres, numSpheresOnScene), std::chrono::system_clock::now().time_since_epoch().count());

    cudaMemcpy(hostParticles, gpuParticles, sizeof(Particle) * conf.numParticles, cudaMemcpyDeviceToHost);

    cudaThreadSynchronize();
}
