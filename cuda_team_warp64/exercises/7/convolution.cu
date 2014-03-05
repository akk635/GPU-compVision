
#include "convolution.h"

#include <aux.h>

// FOR DEBUGGING CALLER
#include <iostream>
using namespace std;

// TODO Fix
// global thread ids lib
#include <global_idx.cu>
// cordinates functions
#include <co_ordinates.cu>

#define PI 3.1412f

// Gaussian Kernel
__global__ void gaussian_kernel(float *d_gaussianKernel, int width, int height, float sigma) {
    // get thread id
    size_t x = threadIdx.x + (size_t) blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + (size_t) blockDim.y * blockIdx.y;
   
    size_t ind = x + width * y;
   
    // only threads inside array range compute
    if (x < width && y < height) {
        d_gaussianKernel[ind] = (1.0f/(2 * PI * sigma * sigma)) * exp(-1.0f *(((x - width/2.0f) * (x - width/2.0f) + (y - height/2.0f) * (y - height/2.0f))/(2 * sigma * sigma)));
    }     
}


// Convolution kernel
__global__ void convolution(float *d_imgIn, float *d_kernel, float *d_imgOut, uint32_t w, uint32_t h, uint32_t nc, uint32_t wKernel, uint32_t hKernel) {
    // get thread global id in 2D
    dim3 globalIdx = globalIdx_Dim3();

    // declare shared memory to store neighbour pixels for the block
    extern __shared__ float imgBlock[];

	// only threads inside image dimensions computes (channels exclusive)
    if(globalIdx.x < w && globalIdx.y < h) {
    	// linearize globalIdx
    	size_t globalId = linearize_globalIdx(w, h);

    	// offset map from thread block to shared memory
    	dim3 offset = dim3(wKernel / 2, hKernel / 2, 0);

    	// theadIdx shifted by offset and linearized
    	dim3 curShMemIdx2D = dim3(offset.x + threadIdx.x, offset.y + threadIdx.y, 0);
    	size_t curShMemIdx = linearize_coords(curShMemIdx2D, dim3(blockDim.x + wKernel, blockDim.y + hKernel, 0));

    	// offset for channel
    	size_t chOffset = w * h;

    	// for each channel
    	for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
    		// current pixel on the image to be copied
    		size_t pixel = globalId;

	    	// copy pixel to shared memory
    		imgBlock[curShMemIdx] = d_imgIn[globalId + chOffset * ch_i];

			// if thread lies on edges of block
    		int onEdge_x = (threadIdx.x == blockDim.x - 1) || (threadIdx.x == 0);
    		int onEdge_y = (threadIdx.y == blockDim.y - 1) || (threadIdx.y == 0);

	    	// border clampings
    		if(onEdge_x || onEdge_y) {
	    		// direction to go depending on which sides being clamped
    			int directn_x = threadIdx.x == blockDim.x - 1 ? 1 : -1;
    			int directn_y = threadIdx.y == blockDim.y - 1 ? 1 : -1;

    			// fill shared memory data that lies outside block dimensions
    			for(uint32_t w_i = 0; w_i <= onEdge_x * wKernel / 2; w_i++) {
    				for(uint32_t h_i = 0; h_i <= onEdge_y * hKernel / 2; h_i++) {
    					curShMemIdx2D = dim3(offset.x + threadIdx.x + directn_x * w_i, offset.y + threadIdx.y + directn_y * h_i, 0);
	 			   		curShMemIdx = linearize_coords(curShMemIdx2D, dim3(blockDim.x + wKernel, blockDim.y + hKernel, 0));

                        // next pixel in 2D
                        dim3 pixel2D = dim3(globalIdx.x + directn_x * w_i, globalIdx.y + directn_y * h_i, 0);
		 				// if current pixel not on border of image then go to next pixel else clamp it
 				   		if(0 <= pixel2D.x && pixel2D.x < w && 0 <= pixel2D.y && pixel2D.y < h)
 				   			pixel = linearize_coords(pixel2D, dim3(w, h, 0));

 				   		// copy pixel to shared memory
					    imgBlock[curShMemIdx] = d_imgIn[pixel + chOffset * ch_i];
    				}
    			}
    		}
            
            // sync threads
            __syncthreads();

            // convolution
            float value = 0.f;
            for(uint32_t kernelIdxX = 0; kernelIdxX < wKernel; kernelIdxX++) {
                for(uint32_t kernelIdxY = 0; kernelIdxY < hKernel; kernelIdxY++) {
                    // compute current shared memory pixel to be convoluted
                    dim3 curShMemIdx2D = dim3(offset.x + threadIdx.x + kernelIdxX - (wKernel / 2), offset.y + threadIdx.y + kernelIdxY - (hKernel / 2), 0);
                    size_t curShMemIdx = linearize_coords(curShMemIdx2D, dim3(blockDim.x + wKernel, blockDim.y + hKernel, 0));

                    // compute current kernel pixel to convoluted
                    dim3 curKernelIdx2D = dim3(kernelIdxX, kernelIdxY, 0);
                    size_t curKernelIdx = linearize_coords(curKernelIdx2D, dim3(wKernel, hKernel, 0));

                    // convolve pixel
                    value += imgBlock[curShMemIdx] * d_kernel[curKernelIdx];
                }
            }
            d_imgOut[globalId + chOffset * ch_i] = value;
    	}
    }
}


// alloc, memcopy, kernel calls (gaussian kernel and convolution) de-alloc 
void gaussian_convolve_GPU(float *h_imgIn, float *h_gaussian, float *h_imgOut, uint32_t w, uint32_t h, uint32_t nc, uint32_t wKernel, uint32_t hKernel, float sigma) {
	// allocate and copy memory to GPU
	size_t imgSize = w * h * nc * sizeof(float);
	size_t kernelSize = wKernel * hKernel * sizeof(float);

    float *d_imgIn, *d_kernel, *d_imgOut;
	cudaMalloc((void **) &d_imgIn, imgSize);
	CUDA_CHECK;
	cudaMalloc((void **) &d_kernel, kernelSize);
	CUDA_CHECK;
	cudaMalloc((void **) &d_imgOut, imgSize);
	CUDA_CHECK;

	cudaMemcpy(d_imgIn, h_imgIn, imgSize, cudaMemcpyHostToDevice);
	CUDA_CHECK;

	// define dimensions - 3D
    // NOTE: CC1.x doesn't support 3D grids
    dim3 block = dim3(16, 16, 1);
    dim3 grid = dim3((wKernel + block.x - 1) / block.x, (hKernel + block.y - 1) / block.y, 1);
    // create gaussian kernel
	gaussian_kernel<<<grid, block>>>(d_kernel, wKernel, hKernel, sigma);

    // copy back data to CPU
    cudaMemcpy(h_gaussian, d_kernel, kernelSize, cudaMemcpyDeviceToHost);
    CUDA_CHECK;
    
    
    float sum = 0.f;
    float maxVal = 0.f;
    for(int i = 0; i < wKernel * hKernel; i++)
        sum += h_gaussian[i];

    for(int i = 0; i < wKernel * hKernel; i++) {
        h_gaussian[i] /= sum;
        if(h_gaussian[i] > maxVal) maxVal = h_gaussian[i];
    }

    /*for(int i = 0; i < wKernel * hKernel; i++)
    {
        h_gaussian[i] /= maxVal;
    }*/


    cudaMemcpy(d_kernel, h_gaussian, kernelSize, cudaMemcpyHostToDevice);
    CUDA_CHECK;
	
	// define dimensions - 3D
    // NOTE: CC1.x doesn't support 3D grids
    block = dim3(16, 16, 1);
    grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

    // dyanmically allocate shared memory bytes - NOTE the size > kernel
    size_t smBytes = (block.x + wKernel) * (block.y + hKernel) * sizeof(float);

    // convolute image
	convolution<<<grid, block, smBytes>>>(d_imgIn, d_kernel, d_imgOut, w, h, nc, wKernel, hKernel);
	
	// copy back data to CPU   
	cudaMemcpy(h_imgOut, d_imgOut, imgSize, cudaMemcpyDeviceToHost);
	CUDA_CHECK;

	// free memory
	cudaFree(d_imgIn);
	CUDA_CHECK;
	cudaFree(d_kernel);
	CUDA_CHECK;
	cudaFree(d_imgOut);
	CUDA_CHECK;
}