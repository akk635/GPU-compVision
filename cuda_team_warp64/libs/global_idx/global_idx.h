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


 // returns the golbal index of a thread along a dimension x
__device__ size_t globalIdx_X();

// returns the golbal index of a thread along a dimension y
__device__ size_t globalIdx_Y();

// returns the golbal index of a thread along a dimension z
__device__ size_t globalIdx_Z();

// returns global indexes in three dimensions
__device__ dim3 globalIdx_Dim3();

// returns linear global id depending on width and height of workspace
__device__ size_t linearize_globalIdx(uint32_t w, uint32_t h);

// returns a neighbour's 3D global id given the offsets
__device__ dim3 neighbour_globalIdx(uint32_t xOff=0, uint32_t yOff=0, uint32_t zOff=0);

// returns linear global id of neighbour given the offset and width and height of workspace
__device__ size_t linearize_neighbour_globalIdx(uint32_t w, uint32_t h, uint32_t xOff=0, uint32_t yOff=0, uint32_t zOff=0);

#endif