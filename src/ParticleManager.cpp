#include "ParticleManager.h"
#include "particles.cuh"
#include "shaders.h"
#include <iostream>

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

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success) {
        char temp[256];
        glGetProgramInfoLog(program, 256, nullptr, temp);
        std::cout << temp;
        glDeleteProgram(program);
        program = 0;
    }

	return program;
}

void ParticleManager::updateParticlesFloatArrays() {
    for (int i = 0; i < numParticles; i++) {
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

ParticleManager::ParticleManager(size_t numParticles, const std::vector<SceneObjectsFactory::Ptr>& scene):
    numParticles(numParticles),
    particlesFloatArraySize(numParticles * 4)
{
	particles = new Particle[numParticles];

	particlesPosFloatArray = new float[particlesFloatArraySize];
    particlesVelFloatArray = new float[particlesFloatArraySize];

	cuda_init(particles, numParticles);

	std::vector<Sphere*> spheres;

	for (auto& s: scene) {
	    if (s->type() == SPHERE) {
	        spheres.push_back(static_cast<Sphere*>(s.get()));
	    }
	}

	cuda_scene_add_spheres(*spheres.data(), spheres.size());

	static const constexpr Shaders shaders;

	glslProgram = compileProgram(shaders.mblurVS, shaders.particlePS, shaders.mblurGS);

	mainVBO = createVBO(sizeof(float) * particlesFloatArraySize);
	colorVBO = createVBO(sizeof(float) * particlesFloatArraySize);
    velVBO = createVBO(sizeof(float) * particlesFloatArraySize);

//	glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
//
//	auto data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
//
//	for (int i = 0; i < numParticles; i++) {
//		*data++ = 0.9;
//		*data++ = 0.9;
//		*data++ = 0.9;
//		*data++ = 1.0f;
//	}
//
//	glUnmapBuffer(GL_ARRAY_BUFFER);
}

// Update method, called every time the particles move
void ParticleManager::update() {
    //GLfloat matrix[16];
//    glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
//
//    float DOF[3];
//    DOF[0] = matrix[2];
//    DOF[1] = matrix[6];
//    DOF[2] = matrix[10];

    cuda_particles_update();

    updateParticlesFloatArrays();

	glBindBuffer(GL_ARRAY_BUFFER, mainVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * particlesFloatArraySize, particlesPosFloatArray, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, velVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * particlesFloatArraySize, particlesVelFloatArray, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ParticleManager::render() {
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);  // don't write depth
    glEnable(GL_BLEND);

    glColor4f(0.0f, 0.0f, 1.0f, 0.8f);

    glUseProgram(glslProgram);

    glUniform1f(glGetUniformLocation(glslProgram, "timestep"), 5);
	glUniform1f(glGetUniformLocation(glslProgram, "pointRadius"),
			particles[0].radius);

    // Bind the vertex VBO
    glBindBuffer(GL_ARRAY_BUFFER, mainVBO);
    glVertexPointer(4, GL_FLOAT, 0, nullptr);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, velVBO);
    glClientActiveTexture(GL_TEXTURE0);
    glTexCoordPointer(4, GL_FLOAT, 0, nullptr);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glDrawArrays(GL_POINTS, 0, numParticles);


//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER_ARB, mIndexBuffer);
//    glDrawElements(GL_POINTS, numParticles, GL_UNSIGNED_INT, nullptr);
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    glDisableClientState(GL_VERTEX_ARRAY);

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);
    glDisable(GL_POINT_SPRITE);
}
