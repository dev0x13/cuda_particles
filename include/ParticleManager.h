#pragma once

#include <vector>
#include "particle.h"
#include "vec3.h"
#include "scene_objects.h"
#include "scene_objects_factory.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

class ParticleManager {
protected:
	Particle *particles;

	float *particlesPosFloatArray;
    float *particlesVelFloatArray;

    GLuint mainVBO;
	GLuint colorVBO;
	GLuint velVBO;
	GLuint glslProgram;

    void updateParticlesFloatArrays();

public:
	size_t numParticles;
	size_t particlesFloatArraySize;

    float windowHeight = 0;
    float fieldOfView = 0;

    ParticleManager(size_t numParticles, const std::vector<SceneObjectsFactory::Ptr>& scene);

	void update();
	void render();
};
