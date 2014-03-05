/******************************************************************************
 * Author: Shiv
 * Date: 03/03/14
 * invert_image_CPU.cpp
    - inverts pixels of a normalized image
 ******************************************************************************/

#include "invert_image_CPU.hpp"


void invert_image_CPU(float* imgIn, float* imgOut, uint32_t w, uint32_t h, uint32_t nc) {
    // for every channel
    for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
        // along the width
        for(uint32_t x = 0; x < w; x++) {
            // along the height
            for(uint32_t y = 0; y < h; y++) {
                // current value index in the linear array
                size_t idx = (size_t) w * h * ch_i + (size_t) w * y + x;

    	        // invert using u = 1 - u
                imgOut[idx] = 1 - imgIn[idx];
            }
        }
    }
}