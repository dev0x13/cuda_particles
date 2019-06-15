#pragma once

#include <vector>

#include <GL/glew.h>

#include <scene_objects_factory.h>
#include <config.h>

struct Particle;

class ParticleSystem: public SceneObject {

    Particle *particles;

    float *particlesPosFloatArray;
    float *particlesVelFloatArray;

    GLuint mainVBO;
    GLuint posVBO;

    GLuint glslProgram;

    void updateParticlesFloatArrays();

    const Config conf;
    const size_t particlesFloatArraySize;

    const SceneObjectType type_ = PARTICLE_SYSTEM;
public:

    ParticleSystem(const Config& config, const std::vector<SceneObjectsFactory::Ptr>& scene);

    void update() override;
    void draw() const override;

    SceneObjectType getType() const {
        return type_;
    }
};
