/*
 * disparity_stereo.h
 *
 *  Created on: Mar 14, 2014
 *      Author: p054
 */

#ifndef DISPARITY_STEREO_H_
#define DISPARITY_STEREO_H_
#include <aux.h>
#include <math.h>
// types of regularizations supported
enum DIFFUSIVITIES {
	HUBER,
};

void disparity_computation_caller(float *h_imgInleft, float *h_imgInright,
		float *h_imgOut, dim3 imgDims, uint32_t nc, uint32_t ncOut, float sigma,
		float tau, uint32_t diffType = HUBER);

__global__ void dataTerm(float *d_f, float *d_imgInleft, float *d_imgInright,
		uint32_t nc, dim3 imgDims);

__global__ void initialize_output(float *d_imgOut, float *d_imgOutFit,
		size_t ncOut);

__global__ void regularizer_update(float *d_zetaX, float *d_zetaY,
		float *d_imgOutFit, float ncOut, float sigma, float imgDims);

__device__ void gradient_imgFit(float *d_imgGradX, float *d_imgGradY,
		float *d_imgOutFit, dim3 globalIdx_XY, size_t ch_i);
#endif /* DISPARITY_STEREO_H_ */
