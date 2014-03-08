/********************************************************************************
* 
* 
********************************************************************************/


#ifndef INPAINTING_GRADIENT_DESCENT_H
#define INPAINTING_GRADIENT_DESCENT_H

// standard int type
#include <stdint.h>
// size_t
#include <stdlib.h>

// types of diffusivities supported
enum REGULARIZERS {
	HUBER,
	QUADRATIC,
};


// creates a mask for a given color in an image and sets that color to a given color
__global__ void mask_and_set(float *d_imgIn, bool *d_mask, float3 maskCol, float3 setCol, dim3 imgDims, uint32_t nc);

// diffusivity formula
__host__ __device__ float g_diffusivity(float EPSILON, float s);

// gradient using forward difference
__global__ void gradient_fd(float *d_imgIn, float *d_imgGradX, float *d_imgGradY, dim3 imgDims, uint32_t nc);

// computes the absolute value of gradient
__global__ void gradient_abs(float *d_imgGradX, float *d_imgGradY, float *d_imgGradNorm, dim3 imgDims, uint32_t nc);

// multiplies the gradient with huber diffusivity value
__global__ void huber_diffuse(float *d_imgGradX, float *d_imgGradY, float * d_imgGradNorm, dim3 imgDims, uint32_t nc, float EPSILON, uint32_t diffType=HUBER);

// calculates the divergence from gardient
__global__ void divergence(float *d_imgGradX, float *d_imgGradY, float *d_imgDiv, dim3 imgDims, uint32_t nc);

// image diffusion u(t + 1) = u(t) + t * DIV(g * GRAD(u(t))) with masking
__global__ void inpaint_with_mask(float *d_imgIn, float *d_imgDiv,  bool *d_mask, dim3 imgDims, uint32_t nc, float TAU);

// huber diffusion caller
void inpainting_gradient_descent(float *h_imgIn, float *h_mask, float *h_imgOut, dim3 imgDims, uint32_t nc, float3 maskColor, float3 setColor, float TAU, float EPSILON, uint32_t steps, uint32_t diffType=HUBER);

#endif