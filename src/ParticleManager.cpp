#include "ParticleManager.h"
#include "particles.cuh"
#include "shaders.h"

extern const char *particleVS;
extern const char *particlePS;

GLuint createVBO(GLuint size) {
	GLuint vbo;

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	return vbo;
}

GLuint compileProgram(const char *vsource, const char *fsource) {
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(vertexShader, 1, &vsource, nullptr);
	glShaderSource(fragmentShader, 1, &fsource, nullptr);

	glCompileShader(vertexShader);
	glCompileShader(fragmentShader);

	GLuint program = glCreateProgram();

	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);

	glLinkProgram(program);

	return program;
}

void ParticleManager::updateParticlesFloatArray() {
    for (int i = 0; i < numParticles; i++) {
        int idx = i * 4;

        particlesFloatArray[idx] = particles[i].position.x;
        particlesFloatArray[idx + 1] = particles[i].position.y;
        particlesFloatArray[idx + 2] = particles[i].position.z;
        particlesFloatArray[idx + 3] = particles[i].lifeTime;
    }
}

ParticleManager::ParticleManager(size_t numParticles, Vec3 /*boxDimensions*/):
    numParticles(numParticles),
    particlesFloatArraySize(numParticles * 4)
{
	particles = new Particle[numParticles];
	particlesFloatArray = new float[particlesFloatArraySize];

	cuda_init(particles, numParticles);

	glslProgram = compileProgram(particleVS, particlePS);
	mainVBO = createVBO(sizeof(float) * particlesFloatArraySize);
	colorVBO = createVBO(sizeof(float) * particlesFloatArraySize);

	glBindBuffer(GL_ARRAY_BUFFER, colorVBO);

	auto data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	for (int i = 0; i < numParticles; i++) {
		*data++ = 0.9;
		*data++ = 0.9;
		*data++ = 0.9;
		*data++ = 1.0f;
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
}

// Update method, called every time the particles move
void ParticleManager::update() {
    cuda_particles_update();

    updateParticlesFloatArray();

	glBindBuffer(GL_ARRAY_BUFFER, mainVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * particlesFloatArraySize, particlesFloatArray, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Method for drawing the particles on screen, using OpenGL
void ParticleManager::render() {
	glEnable(GL_POINT_SPRITE);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	glUseProgram(glslProgram);
	glUniform1f(glGetUniformLocation(glslProgram, "pointScale"),
			windowHeight / tanf(fieldOfView * 0.5f * (float) M_PI / 180.0f));
	glUniform1f(glGetUniformLocation(glslProgram, "pointRadius"),
			particles[0].radius);

	// Set particle rendering size
	glPointSize(1.5f);

	// Bind the vertex VBO
	glBindBuffer(GL_ARRAY_BUFFER, mainVBO);
	glVertexPointer(4, GL_FLOAT, 0, nullptr);
	glEnableClientState(GL_VERTEX_ARRAY);

	// Bind color VBO
	glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
	glColorPointer(4, GL_FLOAT, 0, nullptr);
	glEnableClientState(GL_COLOR_ARRAY);

	glDrawArrays(GL_POINTS, 0, numParticles);

	// Clean up
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glUseProgram(0);
	glDisable(GL_POINT_SPRITE);
}
