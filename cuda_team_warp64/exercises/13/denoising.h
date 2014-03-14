#ifndef DENOISING_H
#define DENOISING_H

// standard int type
#include <stdint.h>
// size_t
#include <stdlib.h>

// types of diffusivities supported
enum DIFFUSIVITIES {
	HUBER,
};


// huber diffusivity formula
__host__ __device__ float g_diffusivity(float EPSILON, float s, uint32_t type);

// gradient using forward difference
__global__ void gradient_fd(float *d_imgIn, float *d_imgGradX, float *d_imgGradY, dim3 imgDims, uint32_t nc);

// computes the absolute value of gradient
__global__ void gradient_abs(float *d_imgGradX, float *d_imgGradY, float *d_imgGradAbs, dim3 imgDims, uint32_t nc);

// jacobi update step for image denoising
__global__ void jacobi_update(float *d_imgIn, float * d_IMG_NOISY, float * d_imgGradAbs, float *d_imgOut, dim3 imgDims, uint32_t nc, float EPSILON, float LAMBDA, uint32_t diffType=HUBER, bool notRedBlack=true, int rbGroup=2);

// SOR update of Gauss-Seidel Method for denoising
__global__ void SOR_update(float *d_imgOld, float *d_imgJacobied, float *d_imgOut, dim3 imgDims, uint32_t nc, float THETA);

// denoise caller (using Euler Lagrange)
void denoise_euler_lagrange_caller(float *h_IMG_NOISY, float *h_imgDenoised, dim3 imgDims, uint32_t nc, float EPSILON, float LAMBDA, uint32_t steps, uint32_t diffType=HUBER);

// denoise caller (using Gauss Seidel)
void denoise_gauss_seidel_caller(float *h_IMG_NOISY, float *h_imgDenoised, dim3 imgDims, uint32_t nc, float EPSILON, float LAMBDA, float THETA, uint32_t steps, uint32_t diffType=HUBER);


#endif