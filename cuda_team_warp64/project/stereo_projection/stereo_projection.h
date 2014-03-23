/******************************************************************************
 * Author: Shiv
 * Date: 16/03/14
 * stereo_projection.h - (header for stereo_projection.cu)
	- finds depth given two images (one shifted relative to other along x-axis)
 ******************************************************************************/

#ifndef STEREO_PROJECTION_H
#define STEREO_PROJECTION_H

// standard int type
#include <stdint.h>
// size_t
#include <stdlib.h>

// calculates the data term for disparity estimation (as defined in pock_convex.pdf)
__global__ void calc_data_term(float *d_imgLeft, float *d_imgRight, float *d_g, dim3 imgDims, uint32_t nc, dim3 convexGridDims, float MU);

// update the dual variable using update step of primal-dual algorithm
__global__ void update_dual(float *d_vCap, float *d_g, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims, float SIGMA);

// updates the primal variable using the update step in primal-dual algorithm
__global__ void update_primal_and_extrapolate(float *d_vn, float *d_phiX, float *d_phiY, float *d_phiZ, float *d_vCap, dim3 convexGridDims, float TAU);

// choses the starting v, v-cap and phi from the C an K set respectively
__global__ void init_primal_dual(float *d_v, float *d_vCap, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims);

// computes summation of all elements along z axis of a particular (x, y) point in v matrix
__global__ void compute_depth_map(float *d_v, float *d_depthMap, dim3 convexGridDims, dim3 imgDims);

// caller - calculates stereo projection depth map using primal-dual algorithm
void stereo_projection_PD(float *h_imgLeft, float *h_imgRight, float  *h_depthMap, dim3 imgDims, uint32_t nc, dim3 convexGridDims, uint32_t steps, float MU, float SIGMA, float TAU);

#endif