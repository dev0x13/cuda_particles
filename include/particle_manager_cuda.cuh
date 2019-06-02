#pragma once

class Particle;
class Sphere;
class Config;

namespace ParticleManagerCuda {
    void AddSpheresToScene(Sphere **spheres, size_t numSpheres);

    void Init(Particle *particles, const Config &conf);

    void Update();
}
