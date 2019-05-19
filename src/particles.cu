#include "particles.cuh"
#include "vec.h"

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <curand_kernel.h>

class DeviceVec;
class BVHNode;

static int gridSize;
static int blockSize;
static Particle *hostParticles;
static Particle *gpuParticles;
static DeviceVec *d_forces;

static size_t numParticles;

/**
 *  3D vector class for the device.
 */
class DeviceVec {

public:
	// Member variables
	float x;
	float y;
	float z;

public:
	// Methods
	__host__ __device__ DeviceVec(float x, float y, float z) :
			x(x), y(y), z(z) {
	}

	__host__ __device__ DeviceVec(DeviceVec *v) :
			x(v->x), y(v->y), z(v->z) {
	}

	__host__ __device__ DeviceVec(Vec3 *v) :
			x(v->x), y(v->y), z(v->z) {
	}

	__device__ float lengthSquared() {
		float sum = 0;
		sum += x * x;
		sum += y * y;
		sum += z * z;

		return sum;
	}

	__device__ float length() {
		return sqrt(lengthSquared());
	}

	__device__ void toVec3(Vec3 *toFill) {
		toFill->x = x;
		toFill->y = y;
		toFill->z = z;
	}

	__device__ float dot(DeviceVec *other) {
		return x * other->x + y * other->y + z * other->z;
	}

	__device__ DeviceVec normalised() {
		float len = length();

		DeviceVec normalised(x / len, y / len, z / len);
		return normalised;
	}

	__device__ DeviceVec *reflection(DeviceVec *normal) {
		DeviceVec tmp(x, y, z);
		tmp = tmp - (*normal * 2 * this->dot(normal));

		return new DeviceVec(tmp);
	}

	// Operators
	__device__ DeviceVec operator+(const DeviceVec& other) {
		DeviceVec newVec(x + other.x, y + other.y, z + other.z);
		return newVec;
	}

	__device__ DeviceVec operator-(const DeviceVec& other) {
		DeviceVec newVec(x - other.x, y - other.y, z - other.z);
		return newVec;
	}

	__device__ DeviceVec operator-(const float scalar) {
		DeviceVec newVec(x - scalar, y - scalar, z - scalar);
		return newVec;
	}

	__device__ DeviceVec operator*(const DeviceVec& other) {
		DeviceVec newVec(x * other.x, y * other.y, z * other.z);
		return newVec;
	}

	__device__ DeviceVec operator*(const float scalar) {
		DeviceVec newVec(x * scalar, y * scalar, z * scalar);
		return newVec;
	}
};

/**
 * Axis-Aligned Bounding Box (AABB) class.
 * Used for collision detection within the Bounding Volume
 * Hierarchy structure.
 */
class AABB {
private:
	DeviceVec centre;
	float width;
	float height;
	float depth;

	// Accessor methods (half dimensions are pre-computed
	// to provide a small level of optimisation)
	__host__ __device__ float getLeft(float halfWidth) {
		return centre.x - halfWidth;
	}

	__host__ __device__ float getRight(float halfWidth) {
		return centre.x + halfWidth;
	}

	__host__ __device__ float getTop(float halfHeight) {
		return centre.y + halfHeight;
	}

	__host__ __device__ float getBottom(float halfHeight) {
		return centre.y - halfHeight;
	}

	__host__ __device__ float getFront(float halfDepth) {
		return centre.z + halfDepth;
	}

	__host__ __device__ float getBack(float halfDepth) {
		return centre.z - halfDepth;
	}

public:
	__host__ __device__ AABB() :
			centre(0,0,0), width(0), height(0), depth(0) {
	}

	__host__ __device__ AABB(DeviceVec centre, float width, float height, float depth) :
			centre(centre), width(width), height(height), depth(depth) {
	}

	__host__ __device__  static AABB fromParticle(Particle *p) {
		DeviceVec centre(&p->position);
		float diameter = p->radius * 2; // This is width, height, and depth

		return AABB(centre, diameter, diameter, diameter);
	}

	/**
 	 * Function for checking whether this AABB and another intersect
 	 */
	__device__ bool intersects(AABB *other) {
		float halfWidth = width / 2;
		float oHalfWidth = other->width / 2;
		float halfHeight = height / 2;
		float oHalfHeight = other->height / 2;
		float halfDepth = depth / 2;
		float oHalfDepth = other->depth / 2;

		if (getRight(halfWidth) < other->getLeft(oHalfWidth))
			return false;
		if (getLeft(halfWidth) > other->getRight(oHalfWidth))
			return false;
		if (getBottom(halfHeight) > other->getTop(oHalfHeight))
			return false;
		if (getTop(halfHeight) < other->getBottom(oHalfHeight))
			return false;
		if (getFront(halfDepth) < other->getBack(oHalfDepth))
			return false;
		if (getBack(halfDepth) > other->getFront(oHalfDepth))
			return false;

		return true;
	}

	// Get the AABB that is found by combining two AABBs
	AABB aabbUnion(AABB other) {
		float halfWidth = width / 2;
		float oHalfWidth = other.width / 2;
		float halfHeight = height / 2;
		float oHalfHeight = other.height / 2;
		float halfDepth = depth / 2;
		float oHalfDepth = other.depth / 2;

		// Get the extreme values (leftmost, rightmost, topmost, etc.) from either AABB
		float left = min(getLeft(halfWidth), other.getLeft(oHalfWidth));
		float right = max(getRight(halfWidth), other.getRight(oHalfWidth));
		float top = max(getTop(halfHeight), other.getTop(oHalfHeight));
		float bottom = min(getBottom(halfHeight), other.getBottom(oHalfHeight));
		float front = max(getFront(halfDepth), other.getFront(oHalfDepth));
		float back = min(getBack(halfDepth), other.getBack(oHalfDepth));

		// Calculate new width, height and depth based on above calculation
		float newWidth = right - left;
		float newHeight = top - bottom;
		float newDepth = front - back;

		DeviceVec newCentre(left + newWidth/2, bottom + newHeight/2, back + newDepth/2);

		return AABB(newCentre, newWidth, newHeight, newDepth);
	}
};

/**
 * Represents a node in a Bounding Volume Hierarchy (BVH).
 * The BVH is described by its root node, and is a binary
 * tree of AABBs.
 */
struct BVHNode {
	int particleIdx;
	AABB boundingBox;
	int leftChildIdx, rightChildIdx;

	BVHNode() :
		particleIdx(-1), boundingBox(), leftChildIdx(-1), rightChildIdx(-1)
	{ }

	// Constructor creates an internal (non-leaf) node
	BVHNode(AABB aabb, int l, int r) :
		particleIdx(-1), boundingBox(aabb), leftChildIdx(l), rightChildIdx(r)
	{ }

	// Static function for creating a leaf node which directly represents a particle
	static BVHNode leafNode(Particle *p, int idx) {
		// Find the aabb bounding box for this particle
		// -1 represents there are no child nodes of this element
		BVHNode node(AABB::fromParticle(p), -1, -1);
		// Set the particle index
		node.particleIdx = idx;

		return node;
	}

	// Check whether a node is a leaf
	__device__ bool isLeaf() {
		// A node is leaf if both of it's child indexes are -1
		return leftChildIdx == -1 && rightChildIdx == -1;
	}

	// Functions for checking the presence of individual children
	__device__ bool hasLeftChild() {
		return leftChildIdx != -1;
	}
	__device__ bool hasRightChild() {
		return rightChildIdx != -1;
	}
};

static thrust::host_vector<BVHNode> nodes;
static BVHNode *d_nodes;

// Device function for checking whether two given particles collide
__device__ bool particlesCollide(Particle *p1, Particle *p2) {
	// Find the vector between the two particles
	DeviceVec collideVec = DeviceVec(&(p2->position)) - DeviceVec(&(p1->position));
	// Find the combined radius of the two particles
	
	float radiuses = p1->radius + p2->radius;
	float collideDistSq = radiuses * radiuses;

	// Particles collide if the distance between them is less
	// than their combined radiuses
	return collideVec.lengthSquared() <= collideDistSq;
}

// Resolve a collision between two particles. Adapted from CUDA samples
__device__ void collide(Particle *p1, Particle *p2, DeviceVec *force) {
	DeviceVec posA(&p1->position);
	DeviceVec posB(&p2->position);
	DeviceVec velA(&p1->velocity);
	DeviceVec velB(&p2->velocity);

	// relative Position and velocity
	DeviceVec relPos = posB - posA;
    DeviceVec relVel = velB - velA;

    // Distance between the two particles
	float dist = relPos.length();
	// Minimum distance for these particles to be colliding
	float collideDist = p1->radius + p2->radius;

	DeviceVec norm = relPos.normalised();

	// New force is accumalated in the force parameter
	// spring force
	*force = *force - norm * 0.5f * (collideDist - dist);
	// damping force
    *force = *force + relVel * 0.02f;
}

__device__ void emitParticle(Particle& particle, unsigned int seed) {
    curandState s;

    curand_init(seed, 0, 0, &s);

    float radius = 0.02;

    float xPos, yPos, zPos;
    float xVel, yVel, zVel;

    xPos = -0.98f;
    float y, z;

    y = curand_uniform(&s) - 0.5;
    if (curand_uniform(&s) > 0.5) {
        z = curand_uniform(&s) * sqrt(0.1 - y * y);
    } else {
        z = (curand_uniform(&s) - 1) * sqrt(0.1 - y * y);
    }

    yPos = y;
    zPos = z;

    xVel = 0.005f;//randomFloat(0.005f, maxVel);
    yVel = 0;//randomFloat(-maxVel, maxVel);
    zVel = 0;//randomFloat(-maxVel, maxVel);

    particle.position = Vec3(xPos, yPos, zPos);
    particle.velocity = Vec3(xVel, yVel, zVel);
    particle.radius = radius;
    particle.lifeTime = curand_uniform(&s) + 0.5;
}

__global__ void firstEmitParticles(Particle *particles, size_t numParticles_) {
    int t_x = threadIdx.x;
    int b_x = blockIdx.x;
    int in_x = b_x * blockDim.x + t_x;

    if (in_x < numParticles_) {
        emitParticle(particles[in_x], t_x);
    }
}

__global__ void moveParticles(Particle * particles, size_t numParticles_) {
	int t_x = threadIdx.x;
	int b_x = blockIdx.x;
	int in_x = b_x * blockDim.x + t_x;

	float delta = 0.01;

	if (in_x < numParticles_) {
		Particle thisParticle = particles[in_x];

		DeviceVec newPosD(&thisParticle.position);
		DeviceVec velD(&thisParticle.velocity);

		__syncthreads();

//		velD = velD + forces[in_x];

		// The below is the original (naive) collision detection method
//		 DeviceVec force(0, 0, 0);
//
//		 for (int i = 0; i < size; i++) {
//		 	if (i != in_x) { // Don't consider ourselves
//		 		Particle other = particles[i];
//
//		 		if (particlesCollide(&thisParticle, &other)) {
//		 			collide(&thisParticle, &other, &force);
//		 		}
//		 	}
//		 }
//
//		 velD = velD + force;

		// Calculate our new desired position
		newPosD = newPosD + velD;

		// Declare normal for wall collisions
		DeviceVec normalD(0, 0, 0);

		bool shouldReflect = false;

		// Set the reflection normal to the wall's normal,
		// if we're touching it
		/*
		if ((newPosD.x > 1 && velD.x > 0) || (newPosD.x < -1 && velD.x < 0)) {
			shouldReflect = true;
			normalD.x = 1;
			normalD.y = 0;
			normalD.z = 0;
		}
		if ((newPosD.y > 1 && velD.y > 0) || (newPosD.y < -1 && velD.y < 0)) {
			shouldReflect = true;
			normalD.x = 0;
			normalD.y = 1;
			normalD.z = 0;
		}
		if ((newPosD.z > 1 && velD.z > 0) || (newPosD.z < -1 && velD.z < 0)) {
			shouldReflect = true;
			normalD.x = 0;
			normalD.y = 0;
			normalD.z = 1;
		}
		*/

        if (shouldReflect) {
			// Reflect with respect to the wall's normal
			velD = velD.reflection(&normalD);
		}

		// Calculate the position after movement
		newPosD = DeviceVec(&thisParticle.position) + velD;

		newPosD.toVec3(&thisParticle.position);
		velD.toVec3(&thisParticle.velocity);

		// Move this particle
		thisParticle.lifeTime -= delta;

		if (thisParticle.lifeTime <= 0) {
		    emitParticle(thisParticle, t_x);
		}

        particles[in_x] = thisParticle;
	}
}


/*
 * The following 2 functions taken from the cuda samples
 */
int iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

/**
 * Compute the ideal grid size for the number of particles that we have
 */
void computeGridSize(int n, int blockSize, int &numBlocks, int &numThreads) {
	numThreads = min(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}

/**
 * Initialize CUDA - allocate memory, etc.
 */
void cuda_init(Particle *particles, int numParticles_) {
    numParticles = numParticles_;

	const size_t particlesArraySize = sizeof(Particle) * numParticles;

    hostParticles = particles;

	cudaMalloc((void**) &gpuParticles, particlesArraySize);

    cudaMemcpy(gpuParticles, particles, particlesArraySize, cudaMemcpyHostToDevice);

	computeGridSize(numParticles, 256, gridSize, blockSize);

    firstEmitParticles<<<gridSize, blockSize>>>(gpuParticles, numParticles);
}

void cuda_particles_update() {
	cudaMemcpy(d_nodes, thrust::raw_pointer_cast(&nodes[0]), sizeof(BVHNode) * nodes.size(), cudaMemcpyHostToDevice);

	moveParticles<<<gridSize, blockSize>>>(gpuParticles, numParticles);

	cudaMemcpy(hostParticles, gpuParticles, sizeof(Particle) * numParticles, cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();

	cudaGetLastError();
}
