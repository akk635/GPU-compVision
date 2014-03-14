/********************************************************************************
* 
* 
********************************************************************************/


#include "inpainting_gradient_descent.h"

// cuda helpers by lab instructors
#include <aux.h>

// FIX
#include <global_idx.h>
#include <global_idx.cu>


// creates a mask for a given color in an image and sets that color to a given color
__global__ void mask_and_set(float *d_imgIn, float *d_mask, float3 maskCol, float3 setCol, dim3 imgDims, uint32_t nc) {
    // get global idx in XY (channels exclusive)
    dim3 globalIdx_XY = globalIdx_Dim2();

    // copy mask and set color
    float maskColor[] = {maskCol.x, maskCol.y, maskCol.z};
    float setColor[] = {setCol.x, setCol.y, setCol.z};

    // only threads inside image boundary computes
    if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
        // get linear index
        size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

        // if current pixel has the mask color and set mask as 0
        bool toMask = true; d_mask[id] = 0.f;

        // for all channels
        for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
            // channel offset
            size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

            // check if pixeil value at current channel satisfies mask color
            toMask = toMask && (d_imgIn[id + chOffset] == maskColor[ch_i]);
        }

        // if pixel has mask color
        if(toMask) {
            // set mask to 1 at pixel location
            d_mask[id] = 1.f;

            // for all channels
            for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
                // channel offset
                size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

                // rewrite pixel by set color
                d_imgIn[id + chOffset] = setColor[ch_i];
            }
        }
    }
}


__global__ void gradient_fd(float *d_imgIn, float *d_imgGradX, float *d_imgGradY, dim3 imgDims, uint32_t nc) {
    // get global idx in XY (channels exclusive)
    dim3 globalIdx_XY = globalIdx_Dim2();

    // only threads inside image boundary computes
    if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
    	// get linear index
        size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

        // for all channels
        for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
            // channel offset
            size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

            // get linear ids of neighbours of offset +1 in x and y dir
            size_t neighX = linearize_neighbour_globalIdx(globalIdx_XY, imgDims, make_int3(1, 0, 0));
            size_t neighY = linearize_neighbour_globalIdx(globalIdx_XY, imgDims, make_int3(0, 1, 0));

            // chalculate differentials along x and y
            d_imgGradX[id + chOffset] = (globalIdx_XY.x + 1) < imgDims.x ? (d_imgIn[neighX + chOffset] - d_imgIn[id + chOffset]) : 0;    
            d_imgGradY[id + chOffset] = (globalIdx_XY.y + 1) < imgDims.y ? (d_imgIn[neighY + chOffset] - d_imgIn[id + chOffset]) : 0;            
        }
    }
}


__global__ void gradient_abs(float *d_imgGradX, float *d_imgGradY, float *d_imgGradNorm, dim3 imgDims, uint32_t nc) {
    // get global idx in XY plane
    dim3 globalIdx_XY = globalIdx_Dim2();


    // only threads inside image boundary computes
    if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
        // get linear index
        size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

        // store the square of absolute value of gradient
        float absGradSq = 0;

        // for all channels
        for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
            // channel offset
            size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;
            
            // squared abs value of gradient in the current channel is added to final sum
            absGradSq += pow(d_imgGradX[id + chOffset], 2) + pow(d_imgGradY[id + chOffset], 2);
        }

        // set norm of gradient
        d_imgGradNorm[id] = sqrtf(absGradSq);
    }
}


__host__ __device__ float g_diffusivity(float EPSILON, float s, uint32_t type) {
    switch(type) {
        case 1:
            return 1;
        default: // HUBER
            return 1.f / max(EPSILON, s);

        // implement more here if needed and update enum in header file
    }
}


__global__ void huber_diffuse(float *d_imgGradX, float *d_imgGradY, float * d_imgGradNorm, dim3 imgDims, uint32_t nc, float EPSILON, uint32_t diffType) {
	// get global idx in XY (channels exclusive)
    dim3 globalIdx_XY = globalIdx_Dim2();

     // only threads inside image boundary computes
    if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
    	// get linear index
        size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

        // for all channels
        for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
        	// channel offset
        	size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

        	// diffuse gradient of current pixel at current channel
            float g = g_diffusivity(EPSILON, d_imgGradNorm[id], diffType);
        	d_imgGradX[id + chOffset] = g * d_imgGradX[id + chOffset];    
        	d_imgGradY[id + chOffset] = g * d_imgGradY[id + chOffset];
        }
    }
}


__global__ void divergence(float *d_imgGradX, float *d_imgGradY, float *d_imgDiv, dim3 imgDims, uint32_t nc) {
	// get global idx in XY (channels exclusive)
    dim3 globalIdx_XY = globalIdx_Dim2();

    // only threads inside image boundary computes
    if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
    	// get linear index
        size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

        // for all channels
        for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
        	// channel offset
            size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

            // get linear ids of neighbours of offset -1 in x and y dir
            size_t neighX = linearize_neighbour_globalIdx(globalIdx_XY, imgDims, make_int3(-1, 0, 0));
            size_t neighY = linearize_neighbour_globalIdx(globalIdx_XY, imgDims, make_int3(0, -1, 0));

            // calculate divergence for the current pixel using backward difference
            float dxxU = ((globalIdx_XY.x + 1) < imgDims.x ? d_imgGradX[id + chOffset] : 0) - (globalIdx_XY.x > 0 ? d_imgGradX[neighX + chOffset] : 0);
            float dyyU = ((globalIdx_XY.y + 1) < imgDims.y ? d_imgGradY[id + chOffset] : 0) - (globalIdx_XY.y > 0 ? d_imgGradY[neighY + chOffset] : 0);
            d_imgDiv[id + chOffset] = dxxU + dyyU;
        }
    }
}


__global__ void inpaint_with_mask(float *d_imgIn, float *d_imgDiv, float *d_mask, dim3 imgDims, uint32_t nc, float TAU) {
	// get global idx in XY (channels exclusive)
    dim3 globalIdx_XY = globalIdx_Dim2();

    // only threads inside image boundary computes
    if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
    	// get linear index
        size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

        // for all channels
        for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
        	// channel offset
            size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

            // evolve image only inside mask
            d_imgIn[id + chOffset] =  d_imgIn[id + chOffset] + d_mask[id] * TAU * d_imgDiv[id + chOffset];
        }
    }
}


void inpainting_gradient_descent(float *h_imgIn, float *h_mask, float *h_imgOut, dim3 imgDims, uint32_t nc, float3 maskColor, float3 setColor, float TAU, float EPSILON, uint32_t steps, uint32_t diffType) {
	// size with channels
    size_t imgSizeBytes = (size_t) imgDims.x * imgDims.y * nc * sizeof(float);

    // alloc GPU memory and copy data
    float *d_imgGradX, *d_imgGradY, *d_imgGradNorm, *d_imgDiv, *d_imgOut;
    float *d_mask;
    cudaMalloc((void **) &d_imgGradX, imgSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgGradY, imgSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgGradNorm, imgSizeBytes / nc);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgDiv, imgSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgOut, imgSizeBytes);
    CUDA_CHECK;
    cudaMemcpy(d_imgOut, h_imgIn, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMalloc((void **) &d_mask, imgSizeBytes / nc);
    CUDA_CHECK;

    // define block and grid
    dim3 block = dim3(16, 16, 1);
    dim3 grid = dim3((imgDims.x + block.x - 1) / block.x, (imgDims.y + block.y - 1) / block.y, 1);

    // create mask
    mask_and_set<<<grid, block>>>(d_imgOut, d_mask, maskColor, setColor, imgDims, nc);

    // for each time step
    for(uint32_t tStep = 0; tStep < steps; tStep++) {
    	// find gradient
    	gradient_fd<<<grid, block>>>(d_imgOut, d_imgGradX, d_imgGradY, imgDims, nc);
    	// normalise the gradient
    	gradient_abs<<<grid, block>>>(d_imgGradX, d_imgGradY, d_imgGradNorm, imgDims, nc);
    	// huber_diffusivity := g * GRAD(U)
    	huber_diffuse<<<grid, block>>>(d_imgGradX, d_imgGradY, d_imgGradNorm, imgDims, nc, EPSILON, diffType);
    	// divergence := DIV(huber_diffusivity)
    	divergence<<<grid, block>>>(d_imgGradX, d_imgGradY, d_imgDiv, imgDims, nc);
    	// diffuse image := U(t + 1) = U(t) + t * divergence with mask
    	inpaint_with_mask<<<grid, block>>>(d_imgOut, d_imgDiv, d_mask, imgDims, nc, TAU);
    }

    // copy back data
    cudaMemcpy(h_imgOut, d_imgOut, imgSizeBytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK;
    cudaMemcpy(h_mask, d_mask, imgSizeBytes / nc, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // free allocations
    cudaFree(d_imgGradX);
    CUDA_CHECK;
    cudaFree(d_imgGradY);
    CUDA_CHECK;
    cudaFree(d_imgGradNorm);
    CUDA_CHECK;
    cudaFree(d_imgDiv);
    CUDA_CHECK;
    cudaFree(d_imgOut);
    CUDA_CHECK;
    cudaFree(d_mask);
    CUDA_CHECK;
}