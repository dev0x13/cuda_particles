#pragma once

#include <device_vec3.h>

struct Config {
    size_t numParticles = 20000;

    // Single particle params
    float particleRadius = 0.1f;
    float particleMaxLifetime = 1;
    float particleMinLifetime = 1;

    DeviceVec3 particleColor = {1, 1, 1};

    DeviceVec3 particleVelocityHighFreqNoiseFactor = {0.001f, 0, 0.001f};
    DeviceVec3 particleVelocityLowFreqNoiseFactor = {0.002f, 0, 0.002f};
    DeviceVec3 particleInitialVelocity = {0, 0.03f, 0};
    DeviceVec3 particleInitialVelocityNoiseFactor = {0.003f, 0, 0.003f};

    // Emitter params
    float emitterRadius = 0.1f;
    DeviceVec3 emitterPos = {0, 0, 0};

    // Environment parameters
    float collisionSpring = 0.1f;
    float collisionDamping = 0.002f;
    float collisionShear = 0.001f;

    float timestep = 0.01f;
    DeviceVec3 gravity = {0, 0.03f, 0};
};