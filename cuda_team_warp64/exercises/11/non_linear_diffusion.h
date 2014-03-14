


#ifndef NON_LINEAR_DIFFUSION_H
#define NON_LINEAR_DIFFUSION_H

// standard int type
#include <stdint.h>
// size_t
#include <stdlib.h>

// types of diffusivities supported
enum DIFFUSIVITIES {
	DEFAULT,
	IDENTITY,
	EXPONENTIAL
};


// huber diffusivity formula
__host__ __device__ float g_diffusivity(float EPSILON, float s, uint32_t type);

// gradient using forward difference
__global__ void gradient_fd(float *d_imgIn, float *d_imgGradX, float *d_imgGradY, dim3 imgDims, uint32_t nc);

// computes the absolute value of gradient
__global__ void gradient_abs(float *d_imgGradX, float *d_imgGradY, float *d_imgGradNorm, dim3 imgDims, uint32_t nc);

// multiplies the gradient with huber diffusivity value
__global__ void huber_diffuse(float *d_imgGradX, float *d_imgGradY, float * d_imgGradNorm, dim3 imgDims, uint32_t nc, float EPSILON, uint32_t diffType=DEFAULT);

// calculates the divergence from gardient
__global__ void divergence(float *d_imgGradX, float *d_imgGradY, float *d_imgDiv, dim3 imgDims, uint32_t nc);

// image diffusion u(t + 1) = u(t) + t * DIV(g * GRAD(u(t)))
__global__ void diffuse_image(float *d_imgIn, float *d_imgDiv, float *d_imgDiffused, dim3 imgDims, uint32_t nc, float TAU);

// huber diffusion caller
void huber_diffusion_caller(float *h_imgIn, float *imgOut, dim3 imgDims, uint32_t nc, float TAU, float EPSILON, uint32_t steps, uint32_t diffType=DEFAULT);

#endif
