/*
 * texture_mem.h
 *
 *  Created on: Mar 5, 2014
 *      Author: p054
 */

#ifndef TEXTURE_MEM_H_
#define TEXTURE_MEM_H_

#include "aux.h"
#include "main.h"

class texture_mem {
private:
	float *d_imgIn, *d_imgOut, *d_kernel;
	int w, h, wKernel, hKernel, nc, nc_out; //nc_out output channels
public:
//	texture<float, 2, cudaReadModeElementType> texRef;
	texture_mem(float *img_In, float *kernel, int w, int h, int wKernel, int hKernel, int nc,
			int nc_out);
	void texture_convolute(float *kernel, dim3 grid, dim3 block);
	void output(float * imgOut);
	~texture_mem( );
};
__global__ void gpu_texture_convolution(float *d_imgIn, float *d_imgOut,
		float *d_k, int wKernel, int hKernel, int w, int h, int nc);


#endif /* TEXTURE_MEM_H_ */
