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

void
disparity_computation_caller(float *h_imgInleft, float *h_imgInright,
    float *h_imgOut, dim3 imgDims, uint32_t nc, float sigma, float tau,
    uint32_t steps, uint32_t mu, uint32_t disparities);

void
disparity_computation_caller_tm(float *h_imgInleft, float *h_imgInright,
    float *h_imgOut, dim3 imgDims, uint32_t nc, float sigma, float tau,
    uint32_t steps, uint32_t mu, uint32_t disparities);

texture<float, 2, cudaReadModeElementType> texRefleftImage;
texture<float, 2, cudaReadModeElementType> texRefrightImage;
texture<float, 3, cudaReadModeElementType> texRefDataTerm;

__global__ void
initialize(float *d_f, float *d_imgInleft, float *d_imgInright, uint32_t nc,
    dim3 imgDims, float **d_imgOutOld, float **d_imgOutFit,
    uint32_t disparities, uint32_t mu);

__global__ void
initialize_tm(float *d_f, uint32_t nc, dim3 imgDims, float **d_imgOutOld,
    float **d_imgOutFit, uint32_t disparities, uint32_t mu);

__global__ void
initialize_dual(float **dptr_phiX, float **dptr_phiY, float **dptr_phiZ,
    uint32_t disparities, dim3 imgDims);

__device__ void
gradient_imgFd(float *dphiX, float *dphiY, float *dphiZ, float **dptr_imgOutOld,
    uint32_t disparity, uint32_t disparities, dim3 globalIdx_XY, dim3 imgDims);

__global__ void
regularizer_update(float **dptr_phiX, float **dptr_phiY, float **dptr_phiZ,
    float **dptr_imgOutFit, float *d_f, float sigma, dim3 imgDims,
    uint32_t disparities);

__global__ void
regularizer_update_tm(float **dptr_phiX, float **dptr_phiY, float **dptr_phiZ,
    float **dptr_imgOutFit, float sigma, dim3 imgDims, uint32_t disparities);

__global__ void
variational_update(float **dptr_imgOutOld, float **dptr_phiX, float **dptr_phiY,
    float **dptr_phiZ, float **dptr_imgOutFit, float tau, dim3 imgDims,
    uint32_t disparities);

__device__ void
divergence_phi(float *div_phi, float **dptr_phiX, float **dptr_phiY,
    float **dptr_phiZ, uint32_t disparity, uint32_t disparities,
    dim3 globalIdx_XY, dim3 imgDims);

__global__ void
layers_summation(float *d_imgOut, float **dptr_imgOutOld, uint32_t disparities,
    dim3 imgDims);

#endif /* DISPARITY_STEREO_H_ */
