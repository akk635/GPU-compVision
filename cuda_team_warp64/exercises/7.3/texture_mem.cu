/*
 * texture_mem.cu
 *
 *  Created on: Mar 5, 2014
 *      Author: p054
 */
#include "texture_mem.h"

texture_mem::texture_mem(float *img_In, float *kernel, int w, int h,
		int wKernel, int hKernel, int nc, int nc_out) {
	cudaMalloc(&d_imgIn, (size_t) w * h * nc * sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_imgOut, (size_t) w * h * nc_out * sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_kernel, (size_t) wKernel * hKernel * sizeof(float));
	CUDA_CHECK;
	this->w = w;
	this->h = h;
	this->nc = nc;
	this->nc_out = nc_out;
	this->wKernel = wKernel;
	this->hKernel = hKernel;

	cudaMemcpy(d_imgIn, img_In, (size_t) w * h * nc * sizeof(float),
			cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMemcpy(d_kernel, kernel, (size_t) wKernel * hKernel * sizeof(float),
			cudaMemcpyHostToDevice);
	CUDA_CHECK;
	// clamp x to border
	texRef.addressMode[0] = cudaAddressModeClamp;
	// clamp y to border
	texRef.addressMode[1] = cudaAddressModeClamp;
	// linear interpolation
	texRef.filterMode = cudaFilterModeLinear;
	// access as (x+0.5f,y+0.5f), not as ((x+0.5f)/w,(y+0.5f)/h)
	texRef.normalized = false;

	// no of bits for each texture channel
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, &texRef, d_imgIn, &desc, w, h * nc,
			 w * sizeof(d_imgIn[0]));
	CUDA_CHECK;
}

void texture_mem::texture_convolute_tm_gk(float *k, dim3 grid, dim3 block) {
	gpu_texture_convolution_tm_gk<<<grid, block>>>(d_imgIn, d_imgOut, d_kernel,
			wKernel, hKernel, w, h, nc);
}

__global__ void gpu_texture_convolution_tm_gk(float *d_imgIn, float *d_imgOut,
		float *d_kernel, int wKernel, int hKernel, int w, int h, int nc) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int ind = x + w * y;
	int domain_size = w * h;
	int kcenter = hKernel / 2 * wKernel + wKernel / 2;
	int offset;
	if (x < w && y < h) {
		for (int i = 0; i < nc; i++) {
			d_imgOut[ind + i * domain_size] = 0;
			for (int p = -wKernel / 2; p <= wKernel / 2; p++) {
				for (int q = -hKernel / 2; q <= hKernel / 2; q++) {
					offset = p + q * wKernel;
/*					if ((x + p) < 0) {
						x = -p;
					} else if ((x + p) >= w) {
						x = w - 1 - p;
					}
					if ((y + q) < 0) {
						y = -q;
					} else if ((y + q) >= h) {
						y = h - 1 - q;
					}*/
					d_imgOut[ind + i * domain_size] += (tex2D(texRef, x + p + 0.5f, (y + q + 0.5f) + i * h)
					*d_kernel[kcenter + offset]);
				}
			}
		}
	}
}

void texture_mem::texture_convolute_tm_ck(float *kernel, dim3 grid, dim3 block) {
	cudaMemcpyToSymbol(constKernel, kernel, (2 * MAXRADIUS + 1) * (2 * MAXRADIUS + 1) * sizeof(float));
	gpu_texture_convolution_tm_ck<<<grid, block>>>(d_imgIn, d_imgOut,
			wKernel, hKernel, w, h, nc);
}

__global__ void gpu_texture_convolution_tm_ck(float *d_imgIn, float *d_imgOut,
	int wKernel, int hKernel, int w, int h, int nc) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int ind = x + w * y;
	int domain_size = w * h;
	int kcenter = hKernel / 2 * wKernel + wKernel / 2;
	int offset;
	if (x < w && y < h) {
		for (int i = 0; i < nc; i++) {
			d_imgOut[ind + i * domain_size] = 0;
			for (int p = -wKernel / 2; p <= wKernel / 2; p++) {
				for (int q = -hKernel / 2; q <= hKernel / 2; q++) {
					offset = p + q * wKernel;
/*					if ((x + p) < 0) {
						x = -p;
					} else if ((x + p) >= w) {
						x = w - 1 - p;
					}
					if ((y + q) < 0) {
						y = -q;
					} else if ((y + q) >= h) {
						y = h - 1 - q;
					}*/
					d_imgOut[ind + i * domain_size] += (tex2D(texRef, x + p + 0.5f, (y + q + 0.5f) + i * h)
					*constKernel[kcenter + offset]);
				}
			}
		}
	}
}

texture_mem::~texture_mem() {
	cudaFree(d_imgIn);
	cudaFree(d_imgOut);
	cudaFree(d_kernel);
	//cudaUnbindTexture(texRef);
}

void texture_mem::output(float *imgOut) {
	cudaMemcpy(imgOut, d_imgOut, (size_t) w * h * nc_out * sizeof(float),
			cudaMemcpyDeviceToHost);
}

