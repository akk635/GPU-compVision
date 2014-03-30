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


// texture reference to bind data term
texture<float, 2, cudaReadModeElementType> texRef2D;


// calculates the data term for disparity estimation
__global__ void calc_data_term(float *d_imgLeft, float *d_imgRight, float *d_g, dim3 imgDims, uint32_t nc, dim3 convexGridDims, float MU);

// calculates the data term for disparity estimation using pitched allocation of data term
__global__ void calc_data_term_pitch(float *d_imgLeft, float *d_imgRight, float *d_g, dim3 imgDims, uint32_t nc, dim3 convexGridDims, size_t pitch, float MU);

// calculates the data term for disparity estimation using shared memory for both images
__global__ void calc_data_term_sm(float *d_imgLeft, float *d_imgRight, float *d_g, dim3 imgDims, uint32_t nc, dim3 convexGridDims, float MU);

// update the dual variable using update step of primal-dual algorithm
__global__ void update_dual(float *d_vCap, float *d_g, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims, float SIGMA);

// update the dual variable using update step of primal-dual algorithm using texture to access data term
__global__ void update_dual_tex(float *d_vCap, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims, float SIGMA);

// update the dual variable using update step of primal-dual algorithm using pitched allocations for all
__global__ void update_dual_pitch(float *d_vCap, float *d_g, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims, size_t pitch, float SIGMA);

// update the dual variable using update step of primal-dual algorithm using shared memory to store d_vCap
__global__ void update_dual_sm(float *d_vCap, float *d_g, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims, float SIGMA);

// updates the primal variable using the update step in primal-dual algorithm
__global__ void update_primal_and_extrapolate(float *d_vn, float *d_phiX, float *d_phiY, float *d_phiZ, float *d_vCap, dim3 convexGridDims, float TAU);

// updates the primal variable using the update step in primal-dual algorithm using pitched allocations for all
__global__ void update_primal_and_extrapolate_pitch(float *d_vn, float *d_phiX, float *d_phiY, float *d_phiZ, float *d_vCap, dim3 convexGridDims, size_t pitch, float TAU);

// choses the starting v, v-cap and phi from the C an K set respectively
__global__ void init_primal_dual(float *d_v, float *d_vCap, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims);

// choses the starting v, v-cap and phi from the C an K set respectively using pitched allocations for all
__global__ void init_primal_dual_pitch(float *d_v, float *d_vCap, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims, size_t pitch);

// computes summation of all elements along z axis of a particular (x, y) point in v matrix
__global__ void compute_depth_map(float *d_v, float *d_depthMap, dim3 convexGridDims, dim3 imgDims);

// computes summation of all elements along z axis of a particular (x, y) point in v matrix using pitched allocations for all except depth map
__global__ void compute_depth_map_pitch(float *d_v, float *d_depthMap, dim3 convexGridDims, dim3 imgDims, size_t pitch);

// caller - calculates stereo projection depth map using primal-dual algorithm
void stereo_projection_PD(float *h_imgLeft, float *h_imgRight, float  *h_depthMap, dim3 imgDims, uint32_t nc, dim3 convexGridDims, uint32_t steps, float MU, float SIGMA, float TAU);

// caller - calculates stereo projection depth map using primal-dual algorithm using texture for data term
void stereo_projection_PD_tex(float *h_imgLeft, float *h_imgRight, float  *h_depthMap, dim3 imgDims, uint32_t nc, dim3 convexGridDims, uint32_t steps, float MU, float SIGMA, float TAU);

// caller - calculates stereo projection depth map using primal-dual algorithm using pitched for all (except the images and depth map as accessed only once in whole run)
void stereo_projection_PD_pitch(float *h_imgLeft, float *h_imgRight, float  *h_depthMap, dim3 imgDims, uint32_t nc, dim3 convexGridDims, uint32_t steps, float MU, float SIGMA, float TAU);

// caller - calculates stereo projection depth map using primal-dual algorithm using shared memeory kernels
void stereo_projection_PD_sm(float *h_imgLeft, float *h_imgRight, float  *h_depthMap, dim3 imgDims, uint32_t nc, dim3 convexGridDims, uint32_t steps, float MU, float SIGMA, float TAU);

#endif