cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(particles LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

file(GLOB sources *.cpp *.cu)

add_executable(
    particles
        ${sources}
)

target_include_directories(
    particles
    PUBLIC
        common/inc
        ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(
    particles
        ${OPENGL_gl_LIBRARY}
        ${OPENGL_glu_LIBRARY}
        ${GLUT_LIBRARIES}
)

set_target_properties(
    particles
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
