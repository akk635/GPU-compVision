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

float *dev_imgIn, *dev_imgOut;
int w, h, nc, nc_out;
void gradient_norm(float *imgIn, float *imgOut, int w, int h, int nc, int nc_out);gradient_class
__global__ void get_gradient_norm( );

#endif /* GRADIENT_CUH_ */
