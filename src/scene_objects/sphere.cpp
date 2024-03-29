#include <scene_objects.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

void Sphere::draw() const {
    glColor3fv(color);

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 10);

    glTranslatef(pos.x, pos.y, pos.z);
    glutSolidSphere(radius, slices, stacks);
    glTranslatef(-pos.x, -pos.y, -pos.z);
}
