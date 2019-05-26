#ifndef __particle_h_
#define __particle_h_

#include "vec3.h"

struct Particle {
	Vec3 position;
	Vec3 velocity;
	
	float radius;
	float age;
	float lifetime;
	
	Particle(Vec3 pos, Vec3 vel, float radius, float age, float lifeTime);

	Particle() {
	    radius = 0;
	    age = 0;
	    lifetime = 0;
	};

	void move();

	bool collidesWith(Particle *other);
};

#endif
