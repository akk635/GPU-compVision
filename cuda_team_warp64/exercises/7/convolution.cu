
#include "convolution.h"

#include <aux.h>
#include <math.h>

// TODO Fix
// global thread ids lib source
#include <global_idx.cu>
// cordinates functions lib source
#include <co_ordinates.cu>


// Gaussian Kernel
void gaussian_kernel(float *gaussian, float sigma) {
    // some consts and params
    const float PI = 3.1412f;
    uint32_t radius = ceil(3 * sigma);
    uint32_t width = 2 * radius + 1;
    uint32_t height = 2 * radius + 1;

    // gaussian init
    float sum = 0.f;
    for(uint32_t x = 0; x < width; x++) {
        for(uint32_t y = 0; y < height; y++) {
            size_t id = y * width + x;
            gaussian[id] = (1.0f/(2 * PI * sigma * sigma)) * exp(-1.0f *(((x - width / 2.0f) * (x - width / 2.0f) + (y - height / 2.0f) * (y - height / 2.0f))/(2 * sigma * sigma)));
            sum += gaussian[id];
        }
    }

    // normalise
    for(size_t i = 0; i < width * height; i++) gaussian[i] /= sum;
}



__global__ void convolution_dsm_gk(float *d_imgIn, float *d_kernel, float *d_imgOut, uint32_t w, uint32_t h, uint32_t nc, uint32_t wKernel, uint32_t hKernel) {
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
 				   		if(pixel2D.x < w && pixel2D.y < h)
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


// alloc, memcopy, kernel calls (gaussian kernel and convolution_dsm_gk) de-alloc 
void gaussian_convolve_GPU(float *h_imgIn, float *h_kernel, float *h_imgOut, uint32_t w, uint32_t h, uint32_t nc, uint32_t wKernel, uint32_t hKernel) {
    // allocate memory and copy data to GPU
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
    cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);
    CUDA_CHECK;
	
	// define dimensions - 3D
    // NOTE: CC1.x doesn't support 3D grids
    dim3 block = dim3(16, 16, 1);
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

    // dyanmically allocate shared memory bytes - NOTE the size > kernel
    size_t smBytes = (block.x + wKernel) * (block.y + hKernel) * sizeof(float);

    // convolute image
	convolution_dsm_gk<<<grid, block, smBytes>>>(d_imgIn, d_kernel, d_imgOut, w, h, nc, wKernel, hKernel);
	
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