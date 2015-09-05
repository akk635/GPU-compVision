/*
 * initialize.h
 *
 *  Created on: Mar 17, 2014
 *      Author: p054
 */

#ifndef INITIALIZE_H_
#define INITIALIZE_H_
#include <aux.h>
#include <math.h>

texture<float, 2, cudaReadModeElementType> texRefleftImage;
texture<float, 2, cudaReadModeElementType> texRefrightImage;

__global__ void initialize(float **d_f, float *d_imgInleft, float *d_imgInright,
		uint32_t nc, dim3 imgDims, float **d_imgOutOld, float **d_imgOutFit,
		uint32_t disparities, uint32_t mu);

__global__ void initialize_tm(float **d_f, float *d_imgInright, uint32_t nc,
		dim3 imgDims, float **d_imgOutOld, float **d_imgOutFit,
		uint32_t disparities, uint32_t mu);

__global__ void initialize_phi(float **dptr_phiX, float **dptr_phiY,
		float **dptr_phiZ, float **dptr_imgOutOld, float **dptr_f,
		uint32_t disparities, dim3 imgDims);

__device__ void gradient_imgFd(float *dphiX, float *dphiY, float *dphiZ,
		float **dptr_imgOutOld, uint32_t disparity, uint32_t disparities,
		dim3 globalIdx_XY, dim3 imgDims);

__global__ void regularizer_update(float **dptr_phiX, float **dptr_phiY,
		float **dptr_phiZ, float **dptr_imgOutFit, float **dptr_f, float sigma,
		dim3 imgDims, uint32_t disparities);

__global__ void variational_update(float **dptr_imgOutNew,
		float **dptr_imgOutOld, float **dptr_phiX, float **dptr_phiY,
		float **dptr_phiZ, float **dptr_imgOutFit, float tau, dim3 imgDims,
		uint32_t disparities);

__device__ void divergence_phi(float *div_phi, float **dptr_phiX,
		float **dptr_phiY, float **dptr_phiZ, uint32_t disparity,
		uint32_t disparities, dim3 globalIdx_XY, dim3 imgDims);

__global__ void layers_summation(float *d_imgOut, float **dptr_imgOutOld,
		uint32_t disparities, dim3 imgDims);

#endif /* INITIALIZE_H_ */
