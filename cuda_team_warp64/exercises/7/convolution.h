#ifndef CONVOLUTION_H
#define CONVOLUTION_H

// for constant memory kernel (with max radius 20)
# define MAX_RAD 20
__constant__ float constKernel[4 * MAX_RAD * MAX_RAD];

// for standard sizes
#include <stdint.h>

// gaussian (CPU)
void gaussian_kernel(float *h_gaussian, float sigma);

// convolution kernel with dynamic shared memory for image and global memory for kernel
__global__ void convolution_dsm_gk(float *d_imgIn, float *d_kernel, float *d_imgOut, int width, int height, int wGaussian, int hGaussian, int nc);

// convolution kernel with dynamic shared memory for image and constant memory for kernel
__global__ void convolution_dsm_ck(float *d_imgIn, float *d_imgOut, int width, int height, int wGaussian, int hGaussian, int nc);

// alloc, memcopy, kernel calls (gaussian kernel and convolution_dsm_gk) de-alloc 
void gaussian_convolve_dsm_gk_GPU(float *h_imgIn, float *h_gaussian, float *h_imgOut, uint32_t w, uint32_t h, uint32_t nc, uint32_t wKernel, uint32_t hKernel);

// alloc, memcopy, kernel calls (gaussian kernel and convolution_dsm_ck) de-alloc 
void gaussian_convolve_dsm_ck_GPU(float *h_imgIn, float *h_gaussian, float *h_imgOut, uint32_t w, uint32_t h, uint32_t nc, uint32_t wKernel, uint32_t hKernel);

#endif