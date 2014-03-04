/*
 * gradient.cu
 *
 *  Created on: Mar 4, 2014
 *      Author: p054
 */

#include "gradient.cuh"
#include <math.h>

void gradient(float *imgIn, float *imgOut, int pw, int ph, int pnc, int pnc_out) {
	cudaMalloc(&dev_imgIn, (size_t) w * h * nc);
	CUDA_CHECK;
	cudaMalloc(&dev_imgOut, (size_t) w * h * nc_out);
	CUDA_CHECK;
	cudaMalloc(&dx_imgIn, (size_t) w * h * nc);
	CUDA_CHECK;
	cudaMalloc(&dy_imgIn, (size_t) w * h * nc);
	CUDA_CHECK;
	w = pw;
	h = ph;
	nc = pnc;
	nc_out = pnc_out;
	cudaMemcpy(dev_imgIn, imgIn, (size_t) w * h * nc, cudaMemcpyHostToDevice);
	CUDA_CHECK;
}

__global__ void get_gradient_norm(float *dev_imgIn, float *dev_imgOut,
		float *dx_imgIn, float *dy_imgIn, int w, int h, int nc, int nc_out) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int index = x + w * y;
	int curr_index;
	dev_imgOut[index] = 0;
	if( x < w && y < h){
		for (int i = 0; i < nc; i++) {
			curr_index = index + (size_t) i * w * h;
			if (x == w -1){
				// Neuamnn bnd
				dx_imgIn[curr_index ] = 0;
			}else {
				dx_imgIn[curr_index] = dev_imgIn[curr_index+1]- dev_imgIn[curr_index];
			}
			if(y == h-1){
				dy_imgIn[curr_index] = 0;
			}else {
				dy_imgIn[curr_index] = dev_imgIn[curr_index+w] - dev_imgIn[curr_index];
			}
			dev_imgOut[index] += ((dx_imgIn[curr_index] * dx_imgIn[curr_index]) + (dy_imgIn[curr_index] * dy_imgIn[curr_index]));
		}
		dev_imgOut[index] =  sqrt(dev_imgOut[index]);
	}
}

void get_norm_write(float * img_Out, dim3 grid, dim3 block){
	get_gradient_norm<<<grid,block>>> (dev_imgIn, dev_imgOut, dx_imgIn, dy_imgIn, w, h, nc, nc_out);
	cudaMemcpy(img_Out, dev_imgOut, (size_t) w * h * nc_out, cudaMemcpyDeviceToHost);
}
