#pragma once

#include <device_vec3.h>

struct Particle {
    DeviceVec3 position;
    DeviceVec3 velocity;

    float radius;
    float age;
    float lifetime;

    Particle(DeviceVec3 pos, DeviceVec3 vel, float radius, float age, float lifeTime) :
            position(pos),
            velocity(vel),
            radius(radius),
            age(age),
            lifetime(lifeTime) {}

    Particle() {
        radius = 0;
        age = 0;
        lifetime = 0;
    };
};


