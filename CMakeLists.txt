cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

project(particles LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

file(GLOB sources
    src/*.cpp
    src/*.cu
    src/scene_objects/*.cpp)

add_executable(
    particles
        ${sources}
)

target_include_directories(
    particles
    PUBLIC
        include/
        ${CUDA_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
)

target_link_libraries(
    particles
        ${OPENGL_gl_LIBRARY}
        ${OPENGL_glu_LIBRARY}
        ${GLUT_LIBRARIES}
        ${GLEW_LIBRARIES}
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

set_target_properties(
    particles
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 11
)
