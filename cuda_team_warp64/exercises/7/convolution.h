#ifndef CO_ORDINATES_H
#define CO_ORDINATES_H

// Convolution Kernel
__global__ void convolution_image(float *d_a, float *d_b, float *d_c, int width, int height, int wGaussian, int hGaussian, int nc);


#endif