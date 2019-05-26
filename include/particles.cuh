#pragma once

class Particle;
class Sphere;

void cuda_scene_add_spheres(Sphere* spheres, size_t numSpheres);
void cuda_init(Particle *particles, size_t numParticles);
void cuda_particles_update();
