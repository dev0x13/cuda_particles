#include "particle_manager.h"
#include "cuda/particle_manager.cuh"

#include <shaders.h>
#include <particle.h>

GLuint createVBO(GLuint size) {
	GLuint vbo;

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	return vbo;
}

GLuint compileProgram(
        const char* vsource,
        const char* fsource,
        const char* gsource)
{
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint geometryShader = glCreateShader(GL_GEOMETRY_SHADER);

	glShaderSource(vertexShader, 1, &vsource, nullptr);
	glShaderSource(fragmentShader, 1, &fsource, nullptr);
    glShaderSource(geometryShader, 1, &gsource, nullptr);

    glCompileShader(vertexShader);
	glCompileShader(fragmentShader);
    glCompileShader(geometryShader);

	GLuint program = glCreateProgram();

	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
    glAttachShader(program, geometryShader);

    glProgramParameteriEXT(program, GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS);
    glProgramParameteriEXT(program, GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
    glProgramParameteriEXT(program, GL_GEOMETRY_VERTICES_OUT_EXT, 4);

	glLinkProgram(program);

	return program;
}

void ParticleSystem::updateParticlesFloatArrays() {
    for (int i = 0; i < conf.numParticles; ++i) {
        int idx = i * 4;

        particlesPosFloatArray[idx] = particles[i].position.x;
        particlesPosFloatArray[idx + 1] = particles[i].position.y;
        particlesPosFloatArray[idx + 2] = particles[i].position.z;
        particlesPosFloatArray[idx + 3] = particles[i].age;

        particlesVelFloatArray[idx] = particles[i].velocity.x;
        particlesVelFloatArray[idx + 1] = particles[i].velocity.y;
        particlesVelFloatArray[idx + 2] = particles[i].velocity.z;
        particlesVelFloatArray[idx + 3] = particles[i].lifetime;
    }
}

ParticleSystem::ParticleSystem(const Config& conf_, const std::vector<SceneObjectsFactory::Ptr>& scene):
    conf(conf_),
    particlesFloatArraySize(conf.numParticles * 4)
{
	particles = new Particle[conf.numParticles];

	particlesPosFloatArray = new float[particlesFloatArraySize];
    particlesVelFloatArray = new float[particlesFloatArraySize];

	ParticleManagerCuda::Init(particles, conf);

	if (!scene.empty()) {
        std::vector<Sphere *> spheres;

        for (auto &s: scene) {
            if (s->type() == SPHERE) {
                spheres.push_back(static_cast<Sphere *>(s.get()));
            }
        }

        if (!spheres.empty()) {
            ParticleManagerCuda::AddSpheresToScene(spheres.data(), spheres.size());
        }
    }

	static const constexpr Shaders shaders;

	glslProgram = compileProgram(shaders.mblurVS, shaders.particlePS, shaders.mblurGS);

	mainVBO = createVBO(sizeof(float) * particlesFloatArraySize);
    posVBO = createVBO(sizeof(float) * particlesFloatArraySize);
}

void ParticleSystem::update() {

    ParticleManagerCuda::Update();

    updateParticlesFloatArrays();

	glBindBuffer(GL_ARRAY_BUFFER, mainVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * particlesFloatArraySize, particlesPosFloatArray, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, posVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * particlesFloatArraySize, particlesVelFloatArray, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ParticleSystem::draw() const {
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);

    glColor4f(conf.particleColor.x, conf.particleColor.y, conf.particleColor.z, 0.8f);

    glUseProgram(glslProgram);

	glUniform1f(glGetUniformLocation(glslProgram, "particleRadius"), conf.particleRadius);

    glBindBuffer(GL_ARRAY_BUFFER, mainVBO);
    glVertexPointer(4, GL_FLOAT, 0, nullptr);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, posVBO);

    glClientActiveTexture(GL_TEXTURE0);
    glTexCoordPointer(4, GL_FLOAT, 0, nullptr);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glDrawArrays(GL_POINTS, 0, conf.numParticles);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    glDisableClientState(GL_VERTEX_ARRAY);

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);
    glDisable(GL_POINT_SPRITE);
}
