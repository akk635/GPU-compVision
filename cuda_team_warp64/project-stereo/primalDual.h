/*
 * primalDual.h
 *
 *  Created on: Mar 18, 2014
 *      Author: p054
 */

#ifndef PRIMALDUAL_H_
#define PRIMALDUAL_H_
#include "initialize.h"

__global__ void regularizer_update(float **dptr_phiX, float **dptr_phiY,
		float **dptr_phiZ, float **dptr_imgOutFit, float **dptr_f, float sigma,
		dim3 imgDims, uint32_t disparities);

__global__ void variational_update(float **dptr_imgOutNew,
		float **dptr_imgOutOld, float **dptr_phiX, float **dptr_phiY,
		float **dptr_phiZ, float **dptr_imgOutFit,float tau, dim3 imgDims, uint32_t disparities);

__device__ void divergence_phi(float *div_phi, float **dptr_phiX, float **dptr_phiY, float **dptr_phiZ, uint32_t disparity,
		uint32_t disparities, dim3 globalIdx_XY, dim3 imgDims);

#endif /* PRIMALDUAL_H_ */
