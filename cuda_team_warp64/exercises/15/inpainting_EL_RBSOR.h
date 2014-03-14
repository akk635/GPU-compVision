/********************************************************************************
* 
* 
********************************************************************************/


#ifndef INPAINTING_EL_RBSOR_H
#define INPAINTING_EL_RBSOR_H

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

// jacobi update step for image denoising
__global__ void jacobi_inpaint_update(float *d_imgIn, float * d_imgGradAbs, float *d_mask, float *d_imgOut, dim3 imgDims, uint32_t nc, float EPSILON, float LAMBDA, uint32_t diffType, bool notRedBlack=true, int rbGroup=2);

// SOR update of Gauss-Seidel Method for denoising
__global__ void SOR_inpaint_update(float *d_imgOld, float *d_imgJacobied, float *d_mask, float *d_imgOut, dim3 imgDims, uint32_t nc, float THETA);

// huber diffusion caller
void inpainting_EL_RBSOR(float *h_imgIn, float *h_mask, float *h_imgOut, dim3 imgDims, uint32_t nc, float3 maskColor, float3 setColor, float EPSILON, float LAMBDA, float THETA, uint32_t steps, uint32_t diffType=HUBER);

#endif