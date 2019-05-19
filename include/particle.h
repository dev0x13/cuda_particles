#ifndef __particle_h_
#define __particle_h_

#include "vec.h"

struct Particle {
	Vec3 position;
	Vec3 velocity;
	
	float radius;
	float lifeTime;
	
	Particle(Vec3 pos, Vec3 vel, float radius, float lifeTime);

	Particle() {
	    radius = 0;
	    lifeTime = 0;
	};

	void move();

	bool collidesWith(Particle *other);
};

#endif
