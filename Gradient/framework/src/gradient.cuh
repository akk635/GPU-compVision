/*
 * gradient.cuh
 *
 *  Created on: Mar 4, 2014
 *      Author: p054
 */

#ifndef GRADIENT_CUH_
#define GRADIENT_CUH_

#include <iostream>
#include <cuda_runtime.h>
#include <ctime>
#include "aux.h"

float *dev_imgIn, *dev_imgOut;
float *dx_imgIn, *dy_imgIn;
int w, h, nc, nc_out;
void gradient(float *imgIn, float *imgOut, int pw, int ph, int pnc, int pnc_out);
void get_norm_write(float * img_Out, dim3 grid, dim3 block);
__global__ void get_gradient_norm(float *dev_imgIn, float *dev_imgOut,
		float *dx_imgIn, float *dy_imgIn, int w, int h, int nc, int nc_out);

#endif /* GRADIENT_CUH_ */
