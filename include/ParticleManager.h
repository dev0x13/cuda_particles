#pragma once

#include <vector>
#include "particle.h"
#include "vec.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

class ParticleManager {
protected:
	Particle *particles;
	float *particlesFloatArray;

	GLuint mainVBO;
	GLuint colorVBO;
	GLuint glslProgram;

    void updateParticlesFloatArray();

public:
	size_t numParticles;
	size_t particlesFloatArraySize;

    float windowHeight = 0;
    float fieldOfView = 0;

    ParticleManager(size_t numParticles, Vec3 boxDimensions);

	void update();
	void render();
};
