#pragma once

struct Particle;
class Sphere;
struct Config;

namespace ParticleManagerCuda {
    void AddSpheresToScene(Sphere **spheres, size_t numSpheres);

    void Init(Particle *particles, const Config &conf);

    void Update();
}
