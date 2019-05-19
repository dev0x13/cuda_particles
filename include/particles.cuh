#pragma once

#include "particle.h"

void cuda_init(Particle *particles, int numParticles);
void cuda_particles_update();
