#pragma once

#include <vec3.h>

struct Particle {
	Vec3 position;
	Vec3 velocity;
	
	float radius;
	float age;
	float lifetime;

    Particle(Vec3 pos, Vec3 vel, float radius, float age, float lifeTime) :
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


