/*
 * gradientclass.cpp
 *
 *  Created on: Mar 4, 2014
 *      Author: p054
 */

#include "gradientclass.h"

gradient_class::gradient_class(float *imgIn, float *imgOut, int w, int h,
		int nc, int nc_out) {
	// TODO Auto-generated constructor stub
	cudaMalloc(&dev_imgIn, (size_t) w * h * nc);
	cudaMalloc(&dev_imgOut, (size_t) w * h * nc_out);
	cudaMalloc(&dx_imgIn, (size_t) w * h * nc);
	cudaMalloc(&dy_imgIn, (size_t) w * h * nc);
	this->w = w;
	this->h = h;
	this->nc = nc;
	this->nc_out = nc_out;
	cudaMemcpy(dev_imgIn, imgIn, (size_t) w * h * nc, cudaMemcpyHostToDevice);
}

gradient_class::~gradient_class() {
	// TODO Auto-generated destructor stub
	cudaFree(dev_imgIn);
	cudaFree(dev_imgOut);
	cudaFree(dx_imgIn);
	cudaFree(dy_imgIn);
}

void gradient_class::create_thread_env(int x, int y, int z) {
	block = dim3(x, y, z);
	grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);
}

void gradient_class::get_gradient_norm( float * img_Out) {
	get_norm<<<grid, block>>>(dev_imgIn,dev_imgOut, dx_imgIn, dy_imgIn, w,h,nc);
	cudaMemcpy(img_Out, dev_imgOut, (size_t) w*h*nc_out, cudaMemcpyDeviceToHost);
}

__global__ friend void get_norm(float *dev_imgIn, float *dev_imgOut,
		float *dx_imgIn, float *dy_imgIn, int w, int h, int nc) {
// COnsidering 2D block and grid
int base_index = (gridDim.x * blockIdx.y * blockDim) + blockIdx.x * blockDim
		+ threadIdx.y * blockDim.x + threadIdx.x;
dev_imgOut[base_index] = 0;
int current_index;
if (base_index < (w * h)) {
	for (int i = 0; i < nc; i++) {
		current_index = base_index + (size_t) w * h * i;
		if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == blockDim.x - 1)) {
			// Making it cyclic
			dx_imgIn[current_index] = dev_imgIn[current_index - w + 1]
					- dev_imgIn[current_index];
		} else {
			dx_imgIn[current_index] = dev_imgIn[current_index + 1]
					- dev_imgIn[current_index];
		}

		if ((blockIdx.y == gridDim.y - 1) && (threadIdx.y == blockDim.y - 1)) {
			dy_imgIn[current_index] = dev_imgIn[current_index - (h - 1) * w]
					- dev_imgIn[current_index];
		} else {
			dy_imgIn[current_index] = dev_imgIn[current_index + w]
					- dev_imgIn[current_index];
		}

		dev_imgOut[base_index] += ((dx_imgIn[current_index]
				* dx_imgIn[current_index])
				+ (dy_imgIn[current_index] * dy_imgIn[current_index]));

	}
}

}

