/******************************************************************************
 * Author: Shiv
 * Date: 03/03/14
 * global_idx.cu
    - Library file for computing global id of a thread
 	-- in a dimension
 	-- in three dimension
 	-- linear id given the width, height and depth of the workspace
 ******************************************************************************/

# include "global_idx.h"


__device__ size_t globalIdx_X() {
 	return (size_t) blockDim.x * blockIdx.x + threadIdx.x;
}


__device__ size_t globalIdx_Y() {
 	return (size_t) blockDim.y * blockIdx.y + threadIdx.y;
}


__device__ size_t globalIdx_Z() {
 	return (size_t) blockDim.z * blockIdx.z + threadIdx.z;
}


__device__ dim3 globalIdx_Dim3() {
 	return dim3(globalIdx_X(), globalIdx_Y(), globalIdx_Z());
}


__device__ size_t linearize_globalIdx(uint32_t w, uint32_t h) {
 	// find global index of thread
 	dim3 globalIdx = globalIdx_Dim3();

 	return (size_t) (globalIdx.z * w * h) + (size_t) (globalIdx.y * w) + globalIdx.x;
}


__device__ dim3 neighbour_globalIdx(uint32_t xOff, uint32_t yOff, uint32_t zOff) {
 	return dim3(globalIdx_X() + xOff, globalIdx_Y() + yOff, globalIdx_Z() + zOff);
}


__device__ size_t linearize_neighbour_globalIdx(uint32_t w, uint32_t h, uint32_t xOff, uint32_t yOff, uint32_t zOff) {
 	// find global index of thread
 	dim3 globalIdx = neighbour_globalIdx(xOff, yOff, zOff);

 	return (size_t) (globalIdx.z * w * h) + (size_t) (globalIdx.y * w) + globalIdx.x;
} 