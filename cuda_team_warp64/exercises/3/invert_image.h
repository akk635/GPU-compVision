/******************************************************************************
 * Author: Shiv
 * Date: 03/03/14
 * invert_image.h - (header for invert_image.cu)
	- inverts pixels of a normalized image
 ******************************************************************************/

#ifndef INVERT_IMAGE_H
#define INVERT_IMAGE_H

// std types
#include <stdint.h>

// inverts the value at a given point (pixel, channel) of image
__device__ float invert_value_at(float *d_img, size_t i);

// inverts image kernel
__global__ void invert_image(float *d_img, uint32_t w, uint32_t h, uint32_t nc);

// responsible for allocation, copy and invert_image kernel call
void invert_image_caller(float* h_imgIn, float* h_imgOut, uint32_t w, uint32_t h, uint32_t nc);

#endif
