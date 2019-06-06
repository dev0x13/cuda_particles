#include <GL/glew.h>
#include <GL/freeglut.h>

#include <scene_objects.h>
#include <particle.h>
#include <particle_manager.h>

// Camera parameters
int ox, oy;
uint8_t buttonState = 0;

enum UserAction {
    IDLE = 0,
    ROTATE = 1,
};

static constexpr const GLfloat redColor[4] = {1, 0, 0, 1};
static constexpr const GLfloat blueColor[4] = {0, 0, 1, 1};
static constexpr const GLfloat greenColor[4] = {0, 1, 0, 1};
static constexpr const GLfloat grayColor[4] = {0.5, 0.5, 0.5, 1};

float camera_trans[] = { 0, -2, -5};
float camera_rot[] = { 0, 0, 0 };

// Window dimensions
int windowWidth = 640;
int windowHeight = 640;

std::vector<SceneObjectsFactory::Ptr> scene;

void calculateFrameRate() {
	static float framesPerSecond = 0.0f; // This will store our fps
	static float lastTime = 0.0f; // This will hold the time from the last frame
	float currentTime = glutGet(GLUT_ELAPSED_TIME);
	++framesPerSecond;
	if (currentTime - lastTime > 1000.0f) {
		lastTime = currentTime;
		char fps[10];
		sprintf(fps, "FPS: %d",
				(int) framesPerSecond);
		glutSetWindowTitle(fps);
		framesPerSecond = 0;
	}
}

static void idle() {
	calculateFrameRate();

    for (const auto& e: scene) {
        e->update();
    }

	glutPostRedisplay();
}

static void render() {
    static constexpr const GLfloat ambientLight[]  = {0.2, 0.2, 0.2, 1.0};
    static constexpr const GLfloat diffuseLight[]  = {0.8, 0.8, 0.8, 1.0};
    static constexpr const GLfloat specularLight[] = {1.0, 1.0, 1.0, 1.0};
    static constexpr const GLfloat lightPosition[] = {1.0, 4.0, 1.0, 1.0};

	glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);

    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Perform translation and rotation based on view parameters
    glTranslatef(camera_trans[0], camera_trans[1], camera_trans[2]);
    glRotatef(camera_rot[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot[1], 0.0, 1.0, 0.0);

    for (const auto& e: scene) {
        e->draw();
    }

	glutSwapBuffers();
}

void reshape(int w, int h) {
	float fov = 60.0f;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fov, (float) w / (float) h, 0.1, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, w, h);
}

void mouseHandler(int, int state, int x, int y) {
	if (state == GLUT_DOWN) {
		buttonState = ROTATE;
	} else if (state == GLUT_UP) {
		buttonState = IDLE;
	}

	ox = x;
	oy = y;

	glutPostRedisplay();
}

void keyboardHandler(unsigned char key, int, int) {

    switch (key) {
        case 'w':
            camera_trans[2] += 1 / 10.0f;
            break;
        case 's':
            camera_trans[2] -= 1 / 10.0f;
            break;
        case 'a':
            camera_trans[0] += 1 / 10.0f;
            break;
        case 'd':
            camera_trans[0] -= 1 / 10.0f;
            break;
        default:
            break;
    }

    glutPostRedisplay();
}


void motionHandler(int x, int y) {
	float dx, dy;

	dx = (float) (x - ox);
	dy = (float) (y - oy);

    if (buttonState & ROTATE) {
        camera_rot[0] += dy / 5.0f;
        camera_rot[1] += dx / 5.0f;
    }

	ox = x;
	oy = y;

	glutPostRedisplay();
}

void initGL(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(windowWidth, windowHeight);

	glutCreateWindow("Particle System");

	glutIdleFunc(&idle);
	glutDisplayFunc(&render);
	glutReshapeFunc(&reshape);

	glutMotionFunc(&motionHandler);
	glutMouseFunc(&mouseHandler);
	glutKeyboardFunc(&keyboardHandler);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glewInit();

	glutReportErrors();
}

/* Demo with smoke */
inline void demo0() {
    scene.push_back(SceneObjectsFactory::create<Plane>(Vec3(0, -0.1, 0), 10, 10, grayColor));

    Config config;

    config.particleInitialVelocity = {0, 0.01, 0};
    config.numParticles = 20000;
    config.particleRadius = 0.15;
    config.emitterRadius = 0.01;
    config.particleColor = {1, 0.1, 0.1};

    scene.push_back(SceneObjectsFactory::create<ParticleSystem>(config, scene));
}

/* Demo with water and spherical obstacle */
inline void demo1() {
    scene.push_back(SceneObjectsFactory::create<Sphere>(Vec3(-0.3, 1.5, 0), 0.5, 100, 100, greenColor));

    Config config;

    config.numParticles = 1000;
    config.particleColor = {0.001, 0.001, 1};
    config.gravity = {0, -0.2, 0};
    config.particleInitialVelocity = {0, 0, 0};
    config.emitterPos = {0, 3.5, 0};
    config.collisionShear = 0.1;
    config.collisionDamping = 0.002;
    config.collisionSpring = 0.09;

    scene.push_back(SceneObjectsFactory::create<ParticleSystem>(config, scene));
}

/* Demo with water and 3 spherical obstacles */
inline void demo2() {
    scene.push_back(SceneObjectsFactory::create<Sphere>(Vec3(-0.35, 1.5, 0), 0.2, 100, 100, greenColor));
    scene.push_back(SceneObjectsFactory::create<Sphere>(Vec3(0.35, 1.5, 0), 0.2, 100, 100, redColor));
    scene.push_back(SceneObjectsFactory::create<Sphere>(Vec3(0, 0, 0), 0.5, 100, 100, redColor));

    Config config;

    config.numParticles = 10000;
    config.particleColor = {0.001, 0.001, 1};
    config.gravity = {0, -0.1, 0};
    config.particleInitialVelocity = {0, 0, 0};
    config.emitterPos = {0, 3.5, 0};
    config.collisionShear = 0.01;
    config.collisionDamping = 0.002;
    config.collisionSpring = 0.5;
    config.particleRadius = 0.07;

    scene.push_back(SceneObjectsFactory::create<ParticleSystem>(config, scene));
}

int main(int argc, char **argv) {
	initGL(argc, argv);

	demo2();

	glutMainLoop();

	return 0;
}
