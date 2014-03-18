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


__device__ dim3 globalIdx_Dim2() {
 	return dim3(globalIdx_X(), globalIdx_Y(), 0);
}


__device__ size_t localIdx_XY() {
	return threadIdx.x + (size_t) threadIdx.y * blockDim.x;
}


__device__ size_t linearize_globalIdx(uint32_t w, uint32_t h, dim3 globalIdx) {
 	return (size_t) (globalIdx.z * w * h) + (size_t) (globalIdx.y * w) + globalIdx.x;
}


__device__ size_t linearize_globalIdx(dim3 globalIdx, dim3 dims) {
 	return (size_t) (globalIdx.z * dims.x * dims.y) + (size_t) (globalIdx.y * dims.x) + globalIdx.x;
}


__device__ dim3 neighbour_globalIdx(int xOff, int yOff, int zOff, dim3 globalIdx) {
 	return dim3(globalIdx.x + xOff, globalIdx.y + yOff, globalIdx.z + zOff);
}


__device__ dim3 neighbour_globalIdx(dim3 globalIdx, int3 offset, dim3 dims, bounding boundType) {
	// calc neighbour
	int x = globalIdx.x + offset.x, y = globalIdx.y + offset.y, z = globalIdx.z + offset.z;

	// boundary check
	switch(boundType) {
		// clamping
		case CLAMP:
			x = dims.x && (x >= (int) dims.x) ? dims.x - 1 : x; x = (x < 0) ? 0 : x;
			y = dims.y && (y >= (int) dims.y) ? dims.y - 1 : y; y = (y < 0) ? 0 : y;
			z = dims.z && (z >= (int) dims.z) ? dims.z - 1 : z; z = (z < 0) ? 0 : z;
			break;
		// none
		default:
			break;
	}

 	return dim3(x, y, z);
}


__device__ size_t linearize_neighbour_globalIdx(uint32_t w, uint32_t h, int xOff, int yOff, int zOff, dim3 globalIdx) {
 	// find global index of thread
 	dim3 neighbourGlobalIdx = neighbour_globalIdx(xOff, yOff, zOff, globalIdx);

 	return (size_t) (neighbourGlobalIdx.z * w * h) + (size_t) (neighbourGlobalIdx.y * w) + neighbourGlobalIdx.x;
}


__device__ size_t linearize_neighbour_globalIdx(dim3 globalIdx, dim3 dims, int3 offset, bounding boundType) {
 	// find global index of thread
 	dim3 neighbourGlobalIdx = neighbour_globalIdx(globalIdx, offset, dims, boundType);

 	return (size_t) (neighbourGlobalIdx.z * dims.x * dims.y) + (size_t) (neighbourGlobalIdx.y * dims.x) + neighbourGlobalIdx.x;
}