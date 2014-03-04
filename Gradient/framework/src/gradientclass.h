/*
 * gradientclass.h
 *
 *  Created on: Mar 4, 2014
 *      Author: p054
 */

#ifndef GRADIENTCLASS_H_
#define GRADIENTCLASS_H_

#include <iostream>
#include <cuda_runtime.h>
#include <ctime>

class gradient_class {
private:
	float *dev_imgIn, *dev_imgOut;
	float *dx_imgIn, *dy_imgIn;
	int w, h, nc, nc_out;
	dim3 block;
	dim3 grid;

public:
	gradient_class();
	gradient_class(float *imgIn, float *imgOut, int w, int h, int nc,
			int nc_out);
	void create_thread_env(int x, int y, int z);
	void get_gradient_norm(float * imgOut);
	__global__ friend void get_norm(float *dev_imgIn, float *dev_imgOut,
			float *dx_imgIn, float *dy_imgIn, int w, int h, int nc);

	~gradient_class();
};

#endif /* GRADIENTCLASS_H_ */
