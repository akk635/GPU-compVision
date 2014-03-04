/*
 * gradient.cu
 *
 *  Created on: Mar 4, 2014
 *      Author: p054
 */

#include <gradient.cuh>

void gradient_norm ( float *imgIn, float *imgOut, int w, int h , int nc, int nc_out){

	// Constructor work
	cudaMalloc(&dev_imgIn, (size_t)w*h*nc); CUDA_CHECK;
	cudaMalloc(&dev_imgOut, (size_t)w*h*nc_out); CUDA_CHECK;
	this.w = w;this.h = h; this.nc = nc; this.nc_out = nc_out;

}
