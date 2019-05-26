#include <scene_objects.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

void Sphere::draw() const {
    glTranslatef(pos.x, pos.y, pos.z);
    glutSolidSphere(radius, slices, stacks);
    glTranslatef(-pos.x, -pos.y, -pos.z);
}
