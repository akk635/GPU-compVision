/*
 * cpu_tasks.h
 *
 *  Created on: Mar 4, 2014
 *      Author: p054
 */

#ifndef CPU_TASKS_H_
#define CPU_TASKS_H_

#include <math.h>

void cpu_Gaussian_kernel(float *k, int kradius, float variance) {
	float coeff = 1 / (2 * M_PI * variance * variance);
	float sum_weights = 0;
	int kwidth = 2 * kradius + 1;
	int kcenter = kradius * kwidth + kradius;
	for (int i = -kradius; i <= kradius; i++) {
		for (int j = -kradius; j <= kradius; j++) {
			k[kcenter + j * kwidth + i] =
					coeff
							* exp(
									-1
											* ((i * i)
													/ (2 * variance * variance)
													+ (j * j)
															/ (2 * variance
																	* variance)));
			sum_weights += k[kcenter + j * kwidth + i];
		}
	}
	for (int i = 0; i < kwidth; i++) {
		for (int j = 0; j < kwidth; j++) {
			k[j * kwidth + i] /= sum_weights;
		}
	}
}

void cpu_convolution(float *imgIn, float *imgOut, float *k, int kradius, int w,
		int h, int nc) {
	int domain_size = w * h;
	int kwidth = 2 * kradius + 1;
	int kcenter = kwidth * kradius + kradius;
	for (int i = 0; i < nc; i++) {
		for (int ind = 0; ind < domain_size; ind++) {
			imgOut[ind + i * domain_size] = 0;
			for (int p = -kradius; p <= kradius; p++) {
				for (int q = -kradius; q <= kradius; q++) {
					imgOut[ind + i * domain_size] += (imgIn[ind
							+ i * domain_size + q * kwidth + p]
							* k[kcenter + q * kwidth + p]);
				}
			}
		}
	}
}

#endif /* CPU_TASKS_H_ */
