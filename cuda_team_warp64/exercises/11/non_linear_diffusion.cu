


#include "non_linear_diffusion.h"

// cuda helpers by lab instructors
#include <aux.h>

// FIX
#include <global_idx.h>
#include <global_idx.cu>



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
            size_t chOffset = imgDims.x * imgDims.y * ch_i;

            // get linear ids of neighbours of offset +1 in x and y dir
            size_t neighX = linearize_neighbour_globalIdx(globalIdx_XY, imgDims, make_int3(1, 0, 0));
            size_t neighY = linearize_neighbour_globalIdx(globalIdx_XY, imgDims, make_int3(0, 1, 0));

            // chalculate differentials along x and y
            d_imgGradX[id + chOffset] = (globalIdx_XY.x + 1) < imgDims.x ? (d_imgIn[neighX + chOffset] - d_imgIn[id + chOffset]) : 0;    
            d_imgGradY[id + chOffset] = (globalIdx_XY.y + 1) < imgDims.y ? (d_imgIn[neighY + chOffset] - d_imgIn[id + chOffset]) : 0;            
        }
    }
}


__global__ void gradient_normalized(float *d_imgGradX, float *d_imgGradY, float *d_imgGradNorm, dim3 imgDims, uint32_t nc) {
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
            size_t chOffset = imgDims.x * imgDims.y * ch_i;
            
            // squared abs value of gradient in the current channel is added to final sum
            absGradSq += d_imgGradX[id + chOffset] * d_imgGradX[id + chOffset] + d_imgGradY[id + chOffset] * d_imgGradY[id + chOffset];
        }

        // set norm of gradient
        d_imgGradNorm[id] = sqrtf(absGradSq);
    }
}


__host__ __device__ float g_huber(float EPSILON, float s) {
	return 1.f / max(EPSILON, s);
}


__global__ void huber_diffusivity(float *d_imgGradX, float *d_imgGradY, float * d_imgGradNorm, dim3 imgDims, uint32_t nc, float EPSILON) {
	// get global idx in XY (channels exclusive)
    dim3 globalIdx_XY = globalIdx_Dim2();

     // only threads inside image boundary computes
    if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
    	// get linear index
        size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

        // for all channels
        for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
        	// channel offset
        	size_t chOffset = imgDims.x * imgDims.y * ch_i;

        	// diffuse gradient of current pixel at current channel
        	d_imgGradX[id + chOffset] = g_huber(EPSILON, d_imgGradNorm[id]) * d_imgGradX[id + chOffset];    
        	d_imgGradY[id + chOffset] = g_huber(EPSILON, d_imgGradNorm[id]) * d_imgGradY[id + chOffset];
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
            size_t chOffset = imgDims.x * imgDims.y * ch_i;

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


__global__ void diffuse_image(float *d_imgIn, float *d_imgDiv, float *d_imgDiffused, dim3 imgDims, uint32_t nc, float TAU) {
	// get global idx in XY (channels exclusive)
    dim3 globalIdx_XY = globalIdx_Dim2();

    // only threads inside image boundary computes
    if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
    	// get linear index
        size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

        // for all channels
        for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
        	// channel offset
            size_t chOffset = imgDims.x * imgDims.y * ch_i;

            // evolve image
            d_imgDiffused[id + chOffset] =  d_imgIn[id + chOffset] + TAU * d_imgDiv[id + chOffset];
        }
    }
}


void huber_diffusion_caller(float *h_imgIn, float *h_imgOut, dim3 imgDims, uint32_t nc, float TAU, float EPSILON, uint32_t steps) {
	// size with channels
    size_t imgSizeBytes = imgDims.x * imgDims.y * nc * sizeof(float);

    // alloc GPU memory and copy data
    float *d_imgIn, *d_imgGradX, *d_imgGradY, *d_imgGradNorm, *d_imgDiv, *d_imgOut;
    cudaMalloc((void **) &d_imgIn, imgSizeBytes);
    CUDA_CHECK;
    cudaMemcpy(d_imgIn, h_imgIn, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
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

    // define block and grid
    dim3 block = dim3(16, 16, 1);
    dim3 grid = dim3((imgDims.x + block.x - 1) / block.x, (imgDims.y + block.y - 1) / block.y, 1);

    // for each time step
    for(uint32_t tStep = 0; tStep < ceil(steps / TAU); tStep++) {
    	// find gradient
    	gradient_fd<<<grid, block>>>(d_imgIn, d_imgGradX, d_imgGradY, imgDims, nc);
    	// normalise the gradient
    	gradient_normalized<<<grid, block>>>(d_imgGradX, d_imgGradY, d_imgGradNorm, imgDims, nc);
    	// huber_diffusivity := g * GRAD(U)
    	huber_diffusivity<<<grid, block>>>(d_imgGradX, d_imgGradY, d_imgGradNorm, imgDims, nc, EPSILON);
    	// divergence := DIV(huber_diffusivity)
    	divergence<<<grid, block>>>(d_imgGradX, d_imgGradY, d_imgDiv, imgDims, nc);
    	// diffuse image := U(t + 1) = U(t) + t * divergence
    	diffuse_image<<<grid, block>>>(d_imgIn, d_imgDiv, d_imgOut, imgDims, nc, TAU);

    	// switch images
    	float * temp = d_imgOut;
    	d_imgOut = d_imgIn;
    	d_imgIn = temp;
    }

    // copy back data
    cudaMemcpy(h_imgOut, d_imgOut, imgSizeBytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // free allocations
    cudaFree(d_imgIn);
    CUDA_CHECK;
    cudaFree(d_imgGradX);
    CUDA_CHECK;
    cudaFree(d_imgGradY);
    CUDA_CHECK;
    cudaFree(d_imgGradNorm);
    CUDA_CHECK;
    cudaFree(d_imgDiv);
    CUDA_CHECK;
    cudaFree(d_imgOut);
}