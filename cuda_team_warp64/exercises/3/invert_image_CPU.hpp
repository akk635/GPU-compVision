/******************************************************************************
 * Author: Shiv
 * Date: 03/03/14
 * invert_image_CPU.hpp - (header for invert_image_CPU.cpp)
 ******************************************************************************/

#ifndef INVERT_IMAGE_CPU_HPP
#define INVERT_IMAGE_CPU_HPP

// std types 
#include <stdint.h>
// size_t
#include <stdlib.h>


// inverts an image passed as linear array
void invert_image_CPU(float* imgIn, float* imgOut, uint32_t w, uint32_t h, uint32_t nc);

#endif