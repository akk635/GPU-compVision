/********************************************************************************
* 
* 
********************************************************************************/


#include "inpainting_EL_RBSOR.h"

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
        case 1:
            return 1;
        default: // HUBER
            return 1.f / max(EPSILON, s);

        // implement more here if needed and update enum in header file
    }
}


__global__ void jacobi_inpaint_update(float *d_imgIn, float * d_imgGradAbs, float *d_mask, float *d_imgOut, dim3 imgDims, uint32_t nc, float EPSILON, float LAMBDA, uint32_t diffType, bool notRedBlack, int rbGroup) {
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

        // inside mask and use RED-BLACK UPDATE
        if (d_mask[id] && (notRedBlack || ((globalIdx_XY.x + globalIdx_XY.y) % 2) == rbGroup)) {
            // for all channels
            for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
                // channel offset
                size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

                // to calculate the Euler Lagrange update step value (for inpainting there is no f)
                float numerator = 0, denominator = 0;

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


__global__ void SOR_inpaint_update(float *d_imgOld, float *d_imgJacobied, float *d_mask, float *d_imgOut, dim3 imgDims, uint32_t nc, float THETA) {
     // get global idx in XY plane
    dim3 globalIdx_XY = globalIdx_Dim2();


    // only threads inside image boundary computes
    if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
        // get linear index
        size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

        // only inside mask
        if (d_mask[id]) {
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
}


void inpainting_EL_RBSOR(float *h_imgIn, float *h_mask, float *h_imgOut, dim3 imgDims, uint32_t nc, float3 maskColor, float3 setColor, float EPSILON, float LAMBDA, float THETA, uint32_t steps, uint32_t diffType) {
	// size with channels
    size_t imgSizeBytes = (size_t) imgDims.x * imgDims.y * nc * sizeof(float);

    // alloc GPU memory and copy data
    float *d_imgIn, *d_imgGradX, *d_imgGradY, *d_imgGradAbs, *d_imgOut, *d_mask;
    cudaMalloc((void **) &d_imgIn, imgSizeBytes);
    CUDA_CHECK;
    cudaMemcpy(d_imgIn, h_imgIn, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgGradX, imgSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgGradY, imgSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgGradAbs, imgSizeBytes / nc);
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
    mask_and_set<<<grid, block>>>(d_imgIn, d_mask, maskColor, setColor, imgDims, nc);

    // for each time step
    for(uint32_t tStep = 0; tStep < steps; tStep++) {
    	// find gradient
    	gradient_fd<<<grid, block>>>(d_imgOut, d_imgGradX, d_imgGradY, imgDims, nc);
    	// normalise the gradient
    	gradient_abs<<<grid, block>>>(d_imgGradX, d_imgGradY, d_imgGradAbs, imgDims, nc);
        // Gauss Seidel inpaint (same as Jacobi with RED BLACK) - black group
        jacobi_inpaint_update<<<grid, block>>>(d_imgIn, d_imgGradAbs, d_mask, d_imgOut, imgDims, nc, EPSILON, LAMBDA, diffType, false, 0);
        // Gauss Seidel inpaint (same as Jacobi with RED BLACK) - red group
        jacobi_inpaint_update<<<grid, block>>>(d_imgOut, d_imgGradAbs, d_mask, d_imgOut, imgDims, nc, EPSILON, LAMBDA, diffType, false, 1);
        // SOR inpaint step of Gauss Seidel
        SOR_inpaint_update<<<grid, block>>>(d_imgIn, d_imgOut, d_mask, d_imgOut, imgDims, nc, THETA);

        // swap pointers
        float *temp = d_imgIn;
        d_imgIn = d_imgOut;
        d_imgOut = temp;
    }

    // copy back data
    cudaMemcpy(h_imgOut, d_imgIn, imgSizeBytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK;
    cudaMemcpy(h_mask, d_mask, imgSizeBytes / nc, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // free allocations
    cudaFree(d_imgIn);
    CUDA_CHECK;
    cudaFree(d_imgGradX);
    CUDA_CHECK;
    cudaFree(d_imgGradY);
    CUDA_CHECK;
    cudaFree(d_imgGradAbs);
    CUDA_CHECK;
    cudaFree(d_imgOut);
    CUDA_CHECK;
    cudaFree(d_mask);
    CUDA_CHECK;
}