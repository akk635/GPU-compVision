/*
 * lapl_dfsion.cu
 *
 *  Created on: Mar 6, 2014
 *      Author: p054
 */
#include "lapl_dfsion.h"

lapl_dfsion::lapl_dfsion(float *img_In, int w, int h, int nc) {
	cudaMalloc(&d_imgIn, (size_t) w * h * nc * sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_imgOut, (size_t) w * h * nc * sizeof(float));
	CUDA_CHECK;
	this->w = w;
	this->h = h;
	this->nc = nc;
	cudaMemcpy(d_imgIn, img_In, (size_t) w * h * nc * sizeof(float),
			cudaMemcpyHostToDevice);
	CUDA_CHECK;
}

void lapl_dfsion::lapl_dfsion_caller(dim3 grid, dim3 block, float finalTime,
		float timeStep) {
	for (float t = 0.f; t < finalTime; t += timeStep) {
		gpu_laplace_dfsion_kernel<<<grid, block>>>(d_imgIn, d_imgOut, w, h, nc,
				 timeStep);
		cudaDeviceSynchronize();
		float *swap = d_imgIn;
		d_imgIn = d_imgOut;
		d_imgOut = swap;
	}
}

__global__ void gpu_laplace_dfsion_kernel(float * d_imgIn, float *d_imgOut,
		int w, int h, int nc, float timeStep) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int ind = x + w * y;
	if (x < w && y < h) {
		for (int i = 0; i < nc; i++) {

			d_imgOut[ind + i * w * h] = d_imgIn[ind + i * w * h]
					+ timeStep
							* (((x + 1) < w ? 1 : 0)
									* d_imgIn[ind + i * w * h + 1]
									+ (x > 0 ? 1 : 0)
											* d_imgIn[ind + i * w * h - 1]
									+ ((y + 1) < h ? 1 : 0)
											* d_imgIn[ind + i * w * h + w]
									+ (y > 0 ? 1 : 0)
											* d_imgIn[ind + i * w * h - w]
									- (((x + 1) < w ? 1 : 0) + (x > 0 ? 1 : 0)
											+ ((y + 1) < h ? 1 : 0)
											+ (y > 0 ? 1 : 0))
											* d_imgIn[ind + i * w * h]);
		}
	}
}

void lapl_dfsion::lapl_output(float * imgOut) {
	cudaMemcpy(imgOut, d_imgIn, (size_t) w * h * nc * sizeof(float),
			cudaMemcpyDeviceToHost);
}

lapl_dfsion::~lapl_dfsion() {
	cudaFree(d_imgIn);
	cudaFree(d_imgOut);
}
