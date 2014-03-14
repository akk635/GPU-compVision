/******************************************************************************
 * Author: Shiv
 * Date: 03/03/14
 * invert_image.cu - (kernel with caller)
    - inverts pixels of a normalized image
 ******************************************************************************/

#include "invert_image.h"

// global_idx lib
// TODO fix includes to .h and linking
#include <global_idx.cu>


__device__ float invert_value_at(float *d_img, size_t i) {
    return 1 - d_img[i];
}


__global__ void invert_image(float *d_img, uint32_t w, uint32_t h, uint32_t nc) {
 	// find global index of thread
 	dim3 globalIdx = globalIdx_Dim3();

 	// inverse for threads inside image dimensions - (x, y, z) - (w, h, nc)
 	if(globalIdx.x < w && globalIdx.y < h && globalIdx.z < nc) {
 		// get linear index
	    size_t id = linearize_globalIdx(w, h);

	    // invert
	    d_img[id] = invert_value_at(d_img, id);
 	}
}


void invert_image_caller(float* h_imgIn, float* h_imgOut, uint32_t w, uint32_t h, uint32_t nc) {
	// size of bytes occupied by image
	size_t imgSize = (size_t) (w * h * nc) * sizeof(float);

    // define block and grid sizes
    // NOTE: while running with CUDA capability < 2.0 3D grid not supported
    dim3 block = dim3(8, 8, 8);
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (nc + block.z - 1) / block.z);

    // alloc GPU memory and copy data
    float *d_img;
    cudaMalloc((void **) &d_img, imgSize);
    cudaMemcpy(d_img, h_imgIn, imgSize, cudaMemcpyHostToDevice);

    // call kernel
    invert_image<<<grid, block>>>(d_img, w, h, nc);

	// wait for kernel call to finish
    cudaDeviceSynchronize();
    
    // check for error
    cudaGetLastError();

    // copy back data
    cudaMemcpy(h_imgOut, d_img, imgSize, cudaMemcpyDeviceToHost);

    // free GPU allocation
    cudaFree(d_img);
}