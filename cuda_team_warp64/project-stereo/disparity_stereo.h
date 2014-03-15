/*
 * disparity_stereo.h
 *
 *  Created on: Mar 14, 2014
 *      Author: p054
 */

#ifndef DISPARITY_STEREO_H_
#define DISPARITY_STEREO_H_
#include <iostream>
#include <aux.h>
#include <math.h>
// types of regularizations supported
enum DIFFUSIVITIES {
	HUBER,
};

void disparity_computation_caller(float *h_imgInleft, float *h_imgInright,
		float *h_imgOut, dim3 imgDims, uint32_t nc, uint32_t ncOut, float sigma,
		float tau,  uint32_t steps, uint32_t diffType = HUBER);

__global__ void dataTerm(float *d_f, float *d_imgInleft, float *d_imgInright,
		uint32_t nc, dim3 imgDims, float *d_imgOutOld, float *d_imgOutFit);

__global__ void initialize_zeta(float *d_imgOutOld, size_t ncOut,
		float *d_zetaX, float *d_zetaY, dim3 imgDims);

__global__ void regularizer_update(float *d_zetaX, float *d_zetaY,
		float *d_imgOutFit, float ncOut, float sigma, dim3 imgDims);

__device__ void gradient_imgFd(float *d_imgGradX, float *d_imgGradY,
		float *d_imgOut, dim3 globalIdx_XY, size_t ch_i, dim3 imgDims);

__global__ void variational_update(float *d_imgOutNew, float *d_imgOutOld,
		float *d_zetaX, float *d_zetaY, float *d_f, float* d_imgOutFit,
		dim3 imgDims, size_t ncOut, float tau);

__device__ void divergence_zeta(float *div_zeta, float *d_zetaX, float *d_zetaY,
		dim3 globalIdx_XY, float ncOut, dim3 imgDims);

#endif /* DISPARITY_STEREO_H_ */
