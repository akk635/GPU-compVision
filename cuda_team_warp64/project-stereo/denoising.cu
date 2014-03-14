

#include "denoising.h"

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


__global__ void gradient_abs(float *d_imgGradX, float *d_imgGradY, float *d_imgGradAbs, dim3 imgDims, uint32_t nc) {
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
        d_imgGradAbs[id] = sqrtf(absGradSq);
    }
}


__host__ __device__ float g_diffusivity(float EPSILON, float s, uint32_t type) {
    switch(type) {
        default: // DEFAULT
            return 1.f / max(EPSILON, s);

        // implement more here if needed and update enum in header file
    }
}


__global__ void jacobi_update(float *d_imgIn, float * d_IMG_NOISY, float * d_imgGradAbs, float *d_imgOut, dim3 imgDims, uint32_t nc, float EPSILON, float LAMBDA, uint32_t diffType, bool notRedBlack, int rbGroup) {
    // get global idx in XY (channels exclusive)
    dim3 globalIdx_XY = globalIdx_Dim2();

    // offsets and boundary contitions as per discretization in sides
    int3 uOffsets[] = {make_int3(1, 0, 0), make_int3(0, 1, 0), make_int3(-1, 0, 0), make_int3(0, -1, 0)};
    int3 gOffsets[] = {make_int3(0, 0, 0), make_int3(0, 0, 0), make_int3(-1, 0, 0), make_int3(0, -1, 0)};
    bool I[] = {globalIdx_XY.x + 1 < imgDims.x, globalIdx_XY.y + 1 < imgDims.y, globalIdx_XY.x > 0, globalIdx_XY.y > 0};


    // only threads inside image boundary computes
    if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
        // get linear index
        size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

        // for RED-BLACK UPDATE
        if (notRedBlack || ((globalIdx_XY.x + globalIdx_XY.y) % 2) == rbGroup) {
            // for all channels
            for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
                // channel offset
                size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

                // to calculate the Euler Lagrange update step value
                float numerator = 2.f * d_IMG_NOISY[id + chOffset], denominator = 2.f;

                // for all four directions around current pixel
                for(uint32_t dir = 0; dir < 4; dir++) {
                    // if boundary condition satisfied
                    if(I[dir]) {
                        // get linear ids of neighbours for u and g in current direction
                        size_t neighU = linearize_neighbour_globalIdx(globalIdx_XY, imgDims, uOffsets[dir]);
                        size_t neighG = linearize_neighbour_globalIdx(globalIdx_XY, imgDims, gOffsets[dir]);
                        
                        // current g for the direction
                        float g = g_diffusivity(EPSILON, d_imgGradAbs[neighG], diffType);

                        // current update step
                        numerator += LAMBDA * g * d_imgIn[neighU + chOffset];
                        denominator += LAMBDA * g;
                    }
                }
                // update the pixel
                d_imgOut[id + chOffset] = numerator / denominator;
            }
        }
    }
}


__global__ void SOR_update(float *d_imgOld, float *d_imgJacobied, float *d_imgOut, dim3 imgDims, uint32_t nc, float THETA) {
	 // get global idx in XY plane
    dim3 globalIdx_XY = globalIdx_Dim2();


    // only threads inside image boundary computes
    if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
        // get linear index
        size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

        // for all channels
        for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
            // channel offset
            size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

            // extrapolate
            float jacobiedUpdate = d_imgJacobied[id + chOffset];
            d_imgOut[id + chOffset] = jacobiedUpdate + THETA * (jacobiedUpdate - d_imgOld[id + chOffset]);
        }
    }
}


void denoise_euler_lagrange_caller(float *h_IMG_NOISY, float *h_imgDenoised, dim3 imgDims, uint32_t nc, float EPSILON, float LAMBDA, uint32_t steps, uint32_t diffType) {
    // size with channels
    size_t imgSizeBytes = (size_t) imgDims.x * imgDims.y * nc * sizeof(float);

    // alloc GPU memory and copy data
    float *d_imgIn, *d_IMG_NOISY, *d_imgGradX, *d_imgGradY, *d_imgGradAbs, *d_imgOut;
    cudaMalloc((void **) &d_imgIn, imgSizeBytes);
    CUDA_CHECK;    
    cudaMemcpy(d_imgIn, h_IMG_NOISY, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMalloc((void **) &d_IMG_NOISY, imgSizeBytes);
    CUDA_CHECK;
    cudaMemcpy(d_IMG_NOISY, h_IMG_NOISY, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgGradX, imgSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgGradY, imgSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgGradAbs, imgSizeBytes / nc);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgOut, imgSizeBytes);
    CUDA_CHECK;
    

    // define block and grid
    dim3 block = dim3(16, 16, 1);
    dim3 grid = dim3((imgDims.x + block.x - 1) / block.x, (imgDims.y + block.y - 1) / block.y, 1);

    // for each time step
    for(uint32_t tStep = 0; tStep < steps; tStep++) {
    	// find gradient
    	gradient_fd<<<grid, block>>>(d_imgIn, d_imgGradX, d_imgGradY, imgDims, nc);
    	// normalise the gradient
    	gradient_abs<<<grid, block>>>(d_imgGradX, d_imgGradY, d_imgGradAbs, imgDims, nc);
        // euler lagrange update
        jacobi_update<<<grid, block>>>(d_imgIn, d_IMG_NOISY, d_imgGradAbs, d_imgOut, imgDims, nc, EPSILON, LAMBDA, diffType);

        // swap pointers
        float *temp = d_imgIn;
        d_imgIn = d_imgOut;
        d_imgOut = temp;
    }

    // copy back data
    cudaMemcpy(h_imgDenoised, d_imgOut, imgSizeBytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // free allocations
    cudaFree(d_imgIn);
    CUDA_CHECK;
    cudaFree(d_IMG_NOISY);
    CUDA_CHECK;
    cudaFree(d_imgGradX);
    CUDA_CHECK;
    cudaFree(d_imgGradY);
    CUDA_CHECK;
    cudaFree(d_imgGradAbs);
    CUDA_CHECK;
    cudaFree(d_imgOut);
    CUDA_CHECK;
}


void denoise_gauss_seidel_caller(float *h_IMG_NOISY, float *h_imgDenoised, dim3 imgDims, uint32_t nc, float EPSILON, float LAMBDA, float THETA, uint32_t steps, uint32_t diffType) {
    // size with channels
    size_t imgSizeBytes = (size_t) imgDims.x * imgDims.y * nc * sizeof(float);

    // alloc GPU memory and copy data
    float *d_imgIn, *d_IMG_NOISY, *d_imgGradX, *d_imgGradY, *d_imgGradAbs, *d_imgOut;
    cudaMalloc((void **) &d_imgIn, imgSizeBytes);
    CUDA_CHECK;    
    cudaMemcpy(d_imgIn, h_IMG_NOISY, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMalloc((void **) &d_IMG_NOISY, imgSizeBytes);
    CUDA_CHECK;
    cudaMemcpy(d_IMG_NOISY, h_IMG_NOISY, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgGradX, imgSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgGradY, imgSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgGradAbs, imgSizeBytes / nc);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgOut, imgSizeBytes);
    CUDA_CHECK;
    

    // define block and grid
    dim3 block = dim3(16, 16, 1);
    dim3 grid = dim3((imgDims.x + block.x - 1) / block.x, (imgDims.y + block.y - 1) / block.y, 1);

    // for each time step
    for(uint32_t tStep = 0; tStep < steps; tStep++) {
    	// find gradient
    	gradient_fd<<<grid, block>>>(d_imgIn, d_imgGradX, d_imgGradY, imgDims, nc);
    	// normalise the gradient
    	gradient_abs<<<grid, block>>>(d_imgGradX, d_imgGradY, d_imgGradAbs, imgDims, nc);
        // Gauss Seidel update (same as JAcobi with RED BLACK) - black group
        jacobi_update<<<grid, block>>>(d_imgIn, d_IMG_NOISY, d_imgGradAbs, d_imgOut, imgDims, nc, EPSILON, LAMBDA, diffType, false, 0);
        // Gauss Seidel update (same as JAcobi with RED BLACK) - red group
        jacobi_update<<<grid, block>>>(d_imgOut, d_IMG_NOISY, d_imgGradAbs, d_imgOut, imgDims, nc, EPSILON, LAMBDA, diffType, false, 1);
        // SOR update step of Gauss Seidel
        SOR_update<<<grid, block>>>(d_imgIn, d_imgOut, d_imgOut, imgDims, nc, THETA);

        // swap pointers
        float *temp = d_imgIn;
        d_imgIn = d_imgOut;
        d_imgOut = temp;
    }

    // copy back data
    cudaMemcpy(h_imgDenoised, d_imgOut, imgSizeBytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // free allocations
    cudaFree(d_imgIn);
    CUDA_CHECK;
    cudaFree(d_IMG_NOISY);
    CUDA_CHECK;
    cudaFree(d_imgGradX);
    CUDA_CHECK;
    cudaFree(d_imgGradY);
    CUDA_CHECK;
    cudaFree(d_imgGradAbs);
    CUDA_CHECK;
    cudaFree(d_imgOut);
    CUDA_CHECK;
}