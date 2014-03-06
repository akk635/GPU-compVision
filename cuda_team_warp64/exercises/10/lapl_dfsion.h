/*
 * lapl_dfsion.h
 *
 *  Created on: Mar 6, 2014
 *      Author: p054
 */

#ifndef LAPL_DFSION_H_
#define LAPL_DFSION_H_
#include "aux.h"

__global__ void gpu_laplace_dfsion_kernel(float * d_imgIn, float *d_imgOut,
		int w, int h, int nc,  float timeStep);

class lapl_dfsion {
private:
	float *d_imgIn, *d_imgOut;
	int w, h, nc; //nc_out output channels
public:
	lapl_dfsion(float *img_In, int w, int h, int nc);
	void lapl_dfsion_caller(dim3 grid, dim3 block, float finalTime,
			float timeStep);
	void lapl_output(float * imgOut);
	~lapl_dfsion();
};

#endif /* LAPL_DFSION_H_ */
