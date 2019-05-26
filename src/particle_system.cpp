#include <scene_objects.h>

#include <stdio.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cmath>
#include "particle.h"
#include "ParticleManager.h"

// Camera parameters
int ox, oy;
int buttonState = 0;
float camera_trans[] = { 0, 0, -3 };
float camera_rot[] = { 0, 0, 0 };
float camera_trans_lag[] = { 0, 0, -3 };
float camera_rot_lag[] = { 0, 0, 0 };
const float inertia = 0.1f;

// Window dimensions
int windowWidth = 640;
int windowHeight = 640;

std::vector<SceneObjectsFactory::Ptr> scene;

ParticleManager *pManager;

/**
 * Calculate and display the current frame rate for benchmarking
 */
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

/**
 * Update particle system here. This is called by the GLUT
 * loop when nothing else is happening.
 */
static void idle(void) {
	calculateFrameRate();

	pManager->update();

	glutPostRedisplay();
}

/**
 * Renders to the screen.
 */
static void render() {
    static constexpr const GLfloat ambientLight[]  = {0.2, 0.2, 0.2, 1.0};
    static constexpr const GLfloat diffuseLight[]  = {0.8, 0.8, 0.8, 1.0};
    static constexpr const GLfloat specularLight[] = {1.0, 1.0, 1.0, 1.0};
    static constexpr const GLfloat whiteColor[]    = {1.0, 1.0, 1.0, 1.0};
    static constexpr const GLfloat redColor[]      = {1.0, 0.0, 0.0, 1.0};
    static constexpr const GLfloat lightPosition[] = {1.0, 0.0, 1.0, 1.0};

	// Clear the screen to begin with
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Set up light
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);

    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);

    // Move the camera
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Update for inertia
    for (int i = 0; i < 3; ++i) {
        camera_trans_lag[i] += (camera_trans[i] - camera_trans_lag[i]) * inertia;
        camera_rot_lag[i] += (camera_rot[i] - camera_rot_lag[i]) * inertia;
    }

    // Perform translation and rotation based on view parameters
    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    glColor3fv(redColor);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, redColor);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, redColor);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, redColor);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 20);

    for (const auto& e: scene) {
        e->draw();
    }

    pManager->render();

	glutSwapBuffers();
}

/**
 * Called when the window resizes. Reload the camera projection,
 * updating for the new aspect ratio.
 */
void reshape(int w, int h) {
	float fov = 60.0f;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fov, (float) w / (float) h, 0.1, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, w, h);

	pManager->windowHeight = h;
	pManager->fieldOfView = fov;
}

/**
 * Called when the mouse changes state.
 */
void mouse(int button, int state, int x, int y) {
	int mods;

	if (state == GLUT_DOWN) {
		buttonState |= 1 << button;
	} else if (state == GLUT_UP) {
		buttonState = 0;
	}

	mods = glutGetModifiers();

	if (mods & GLUT_ACTIVE_SHIFT) {
		buttonState = 2;
	} else if (mods & GLUT_ACTIVE_CTRL) {
		buttonState = 3;
	}

	ox = x;
	oy = y;

	glutPostRedisplay();
}

/**
 * Called when the mouse is dragged. Updates the camera
 * parameters depending on the button state. (Taken from
 * the CUDA sample)
 */
void motion(int x, int y) {
	float dx, dy;
	dx = (float) (x - ox);
	dy = (float) (y - oy);

	if (buttonState == 3) {
		// left+middle = zoom
		camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
	} else if (buttonState & 2) {
		// middle = translate
		camera_trans[0] += dx / 100.0f;
		camera_trans[1] -= dy / 100.0f;
	} else if (buttonState & 1) {
		// left = rotate
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
	glutMotionFunc(&motion);
	glutMouseFunc(&mouse);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glewInit();

	glutReportErrors();
}

int main(int argc, char **argv) {
	initGL(argc, argv);

	// Set up params
	size_t numParticles = 650;

    auto sphere = SceneObjectsFactory::create<Sphere>(Vec3(0.1, -1, 0), 0.1, 100, 100);

    scene.push_back(sphere);
	
	// Generate the required number of particles
	pManager = new ParticleManager(numParticles, scene);

	glutMainLoop();

	// Free the memory used to store the particles
	delete pManager;

	return 0;
}
