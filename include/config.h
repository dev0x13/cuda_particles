#pragma once

struct Config {
    size_t numParticles = 10000;

    // Single particle params
    float particleRadius = 0.1;
    float particleMaxLifetime = 1;
    float particleMinLifetime = 1;

    Vec3 particleColor = {1, 0.5, 0.5};

    Vec3 particleVelocityNoiseFactor = {0.001, 0, 0.001};
    Vec3 particleInitialVelocity = {0, 0.03, 0};
    Vec3 particleInitialVelocityNoiseFactor = {0.0003, 0, 0.0003};

    // Emitter params
    float emitterRadius = 0.01;
    Vec3 emitterPos = {0, 0, 0};

    // Environment parameters
    float collisionSpring = 0.1;
    float collisionDamping = 0.002;
    float collisionShear = 0.01;

    float timestep = 0.01;
    Vec3 gravity = {0, 0.03, 0};
};