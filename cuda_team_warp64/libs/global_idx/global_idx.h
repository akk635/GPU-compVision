/******************************************************************************
 * Author: Shiv
 * Date: 03/03/14
 * global_idx.h
    - Library file for computing global id of a thread
 	    -- in a dimension
 	    -- in three dimension
 	    -- linear id given the width, height and depth of the workspace
 ******************************************************************************/

#ifndef GLOBAL_IDX_H
#define GLOBAL_IDX_H

// size_t
#include <stdlib.h>
// std types
#include <stdint.h>
// string
#include <string>


using namespace std;

// enum of bounding check kinds
enum bounding {
	CLAMP,
	NONE
};


 // returns the golbal index of a thread along a dimension x
__device__ size_t globalIdx_X();

// returns the golbal index of a thread along a dimension y
__device__ size_t globalIdx_Y();

// returns the golbal index of a thread along a dimension z
__device__ size_t globalIdx_Z();

// returns global indexes in three dimensions
__device__ dim3 globalIdx_Dim3();

// returns global indexes in two dimensions
__device__ dim3 globalIdx_Dim2();

// returns the global id of the thread in the block's X-Y plane
__device__ size_t localIdx_XY();

// returns linear global id depending on width and height of workspace
__device__ size_t linearize_globalIdx(uint32_t w, uint32_t h, dim3 globalIdx=globalIdx_Dim3());

// returns linear global id depending on width and height of workspace (overloaded)
__device__ size_t linearize_globalIdx(dim3 globalIdx, dim3 dims);

// returns a neighbour's (3D default) global id given the offsets
__device__ dim3 neighbour_globalIdx(int xOff=0, int yOff=0, int zOff=0, dim3 globalIdx=globalIdx_Dim3());

// returns a neighbour's (3D default) global id given the offsets (overloaded)
__device__ dim3 neighbour_globalIdx(dim3 globalIdx, int3 offset, dim3 dims, bounding boundType);

// returns linear global id of neighbour given the offset and width and height of workspace
__device__ size_t linearize_neighbour_globalIdx(uint32_t w, uint32_t h, int xOff=0, int yOff=0, int zOff=0, dim3 globalIdx=globalIdx_Dim3());

// returns linear global id of neighbour given the offset and width and height of workspace (overloaded); also can take bounding type
__device__ size_t linearize_neighbour_globalIdx(dim3 globalIdx, dim3 dims, int3 offset, bounding boundType=NONE);


#endif
