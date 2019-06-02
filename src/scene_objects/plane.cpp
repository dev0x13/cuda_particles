#include <scene_objects.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

void Plane::draw() const {
    glColor3fv(color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 20);

    glTranslatef(pos.x, pos.y, pos.z);

    glBegin(GL_QUADS);

    for (size_t i = 0; i < 4; ++i) {
        glVertex3f(vertexes[i].x, vertexes[i].y, vertexes[i].z);
    }

    glEnd();

    glTranslatef(-pos.x, -pos.y, -pos.z);
}
