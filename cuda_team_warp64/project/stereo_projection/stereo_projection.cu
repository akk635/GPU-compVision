/******************************************************************************
 * Author: Shiv
 * Date: 16/03/14
 * stereo_projection.cu - (kernels with caller)
	- finds depth given two images (one shifted relative to other along x-axis)
 ******************************************************************************/

#include "stereo_projection.h"

// cuda helpers by lab instructors
#include <aux.h>
// FIX
#include <global_idx.h>
#include <global_idx.cu>

#include <iostream>


__global__ void calc_data_term(float *d_imgLeft, float *d_imgRight, float *d_g, dim3 imgDims, uint32_t nc, dim3 convexGridDims, float MU) {
    // get global idx in convex grid space
    dim3 globalIdx = globalIdx_Dim3();

    // only threads inside convex grid space computes
    if (globalIdx.x < convexGridDims.x && globalIdx.y < convexGridDims.y && globalIdx.z < convexGridDims.z) {
		// get linear index in convex grid space
        size_t id = linearize_globalIdx(globalIdx, convexGridDims);
    	// get linear index in XY
        size_t id_XY = linearize_globalIdx(dim3(globalIdx.x, globalIdx.y, 0), imgDims);

        // to store calc of data term for current thread
        float g = 0.f;

        // for all channels
        for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
            // channel offset
            size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

            // get linear ids of shifted pixel for left image with clamping
            size_t shiftedPixel = linearize_neighbour_globalIdx(dim3(globalIdx.x, globalIdx.y, 0), imgDims, make_int3(globalIdx.z, 0, 0));

            // calculate difference in intensity for current channel and shift
            g += fabsf(d_imgRight[id_XY + chOffset] - (globalIdx.x + globalIdx.z >= imgDims.x ? 0.f : d_imgLeft[shiftedPixel + chOffset]));
        }

        // store final calculation
        d_g[id] = MU * g;
    }
}


__global__ void calc_data_term_pitch(float *d_imgLeft, float *d_imgRight, float *d_g, dim3 imgDims, uint32_t nc, dim3 convexGridDims, size_t pitch, float MU) {
    // get global idx in convex grid space
    dim3 globalIdx = globalIdx_Dim3();

    // pitched dims
    dim3 convexGridDimsPitched = dim3(pitch, convexGridDims.y, convexGridDims.z);

    // only threads inside convex grid space computes
    if (globalIdx.x < convexGridDims.x && globalIdx.y < convexGridDims.y && globalIdx.z < convexGridDims.z) {
        // get linear index in convex grid space
        size_t id = linearize_globalIdx(globalIdx, convexGridDimsPitched);
        // get linear index in XY
        size_t id_XY = linearize_globalIdx(dim3(globalIdx.x, globalIdx.y, 0), imgDims);

        // to store calc of data term for current thread
        float g = 0.f;

        // for all channels
        for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
            // channel offset
            size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

            // get linear ids of shifted pixel for left image with clamping
            size_t shiftedPixel = linearize_neighbour_globalIdx(dim3(globalIdx.x, globalIdx.y, 0), imgDims, make_int3(globalIdx.z, 0, 0));

            // calculate difference in intensity for current channel and shift
            g += fabsf(d_imgRight[id_XY + chOffset] - (globalIdx.x + globalIdx.z >= imgDims.x ? 0.f : d_imgLeft[shiftedPixel + chOffset]));
        }

        // store final calculation
        d_g[id] = MU * g;
    }
}


__global__ void calc_data_term_sm(float *d_imgLeft, float *d_imgRight, float *d_g, dim3 imgDims, uint32_t nc, dim3 convexGridDims, float MU) {
    // shared memory to store blocks of both images related to the block
    extern __shared__ float s_total[];
    float* s_imgRight = s_total;
    float* s_imgLeft = &(s_total[(size_t) blockDim.x * blockDim.y * nc]);

    // get global idx in convex grid space
    dim3 globalIdx = globalIdx_Dim3();

    // only threads inside convex grid space computes
    if (globalIdx.x < convexGridDims.x && globalIdx.y < convexGridDims.y && globalIdx.z < convexGridDims.z) {
        // get linear index in convex grid space
        size_t id = linearize_globalIdx(globalIdx, convexGridDims);
        // get linear index in XY
        size_t id_XY = linearize_globalIdx(dim3(globalIdx.x, globalIdx.y, 0), imgDims);
        // get linear id of thread in block's XY plane
        size_t id_block_XY = linearize_threadIdx(dim3(threadIdx.x, threadIdx.y, 0));

        // for all channels - copy to shared memory
        for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
            // channel offsets for glabal image and shared image black
            size_t chOffsetImg = (size_t) imgDims.x * imgDims.y * ch_i;
            size_t chOffsetBlock = (size_t) blockDim.x * blockDim.y * ch_i;

            // get linear ids of shifted pixel in left image
            size_t shiftedPixelImg = linearize_neighbour_globalIdx(dim3(globalIdx.x, globalIdx.y, 0), imgDims, make_int3(globalIdx.z, 0, 0));

            // threads on the XY surface of block copies left and right images chunks related to current thread block
            if (threadIdx.z == 0) {
                // the related right image corresponds to the XY location of the block in the grid
                s_imgRight[id_block_XY + chOffsetBlock] = d_imgRight[id_XY + chOffsetImg];

                // the related left image block is shifted along XY location of the block in the grid by z depth of block
                s_imgLeft[id_block_XY + chOffsetBlock] = d_imgLeft[shiftedPixelImg + chOffsetImg];
            }
        }

        // make sure shared memory data copy is done before moving forward
        __syncthreads();

        // to store calc of data term for current thread
        float g = 0.f;

        // for all channels - calcuate data term
        for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
            // channel offsets for glabal image and shared image black
            size_t chOffsetImg = (size_t) imgDims.x * imgDims.y * ch_i;
            size_t chOffsetBlock = (size_t) blockDim.x * blockDim.y * ch_i;

            // get linear ids of shifted pixel for left image and shared left image block
            size_t shiftedPixelImg = linearize_neighbour_globalIdx(dim3(globalIdx.x, globalIdx.y, 0), imgDims, make_int3(globalIdx.z, 0, 0));
            size_t shiftedPixelBlock = linearize_neighbour_globalIdx(dim3(threadIdx.x, threadIdx.y, 0), blockDim, make_int3(threadIdx.z, 0, 0));

            // read left image shifted pixel from shared memory or global memory depending on thread block boundary check
            float leftImgPixel = (threadIdx.x + threadIdx.z >= blockDim.x) ? d_imgLeft[shiftedPixelImg + chOffsetImg] : s_imgLeft[shiftedPixelBlock + chOffsetBlock];

            // calculate difference in intensity for current channel
            g += fabsf(s_imgRight[id_block_XY + chOffsetBlock] - ((globalIdx.x + globalIdx.z >= imgDims.x) ? 0.f : leftImgPixel));
        }

        // store final calculation
        d_g[id] = MU * g;
    }
}


__global__ void update_dual(float *d_vCap, float *d_g, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims, float SIGMA) {
    // get global idx in convex grid space
    dim3 globalIdx = globalIdx_Dim3();

    // only threads inside convex grid space computes
    if (globalIdx.x < convexGridDims.x && globalIdx.y < convexGridDims.y && globalIdx.z < convexGridDims.z) {
        // get linear index in convex grid space
        size_t id = linearize_globalIdx(globalIdx, convexGridDims);

        // get linear ids of neighbours of offset +1 in x, y and z dir
        size_t neighX = linearize_neighbour_globalIdx(globalIdx, convexGridDims, make_int3(1, 0, 0));
        size_t neighY = linearize_neighbour_globalIdx(globalIdx, convexGridDims, make_int3(0, 1, 0));
        size_t neighZ = linearize_neighbour_globalIdx(globalIdx, convexGridDims, make_int3(0, 0, 1));

        // chalculate differentials along x, y and z of primal variable cap and update of dual variable without projection
        float v = d_vCap[id];
        float phiX = d_phiX[id] + SIGMA * ((globalIdx.x + 1) < convexGridDims.x ? d_vCap[neighX] - v : 0);
        float phiY = d_phiY[id] + SIGMA * ((globalIdx.y + 1) < convexGridDims.y ? d_vCap[neighY] - v : 0);
        float phiZ = d_phiZ[id] + SIGMA * ((globalIdx.z + 1) < convexGridDims.z ? d_vCap[neighZ] - v : 0);

        // update the dual variable with projection back to set K
        float g = d_g[id];
        float normalizer = fmaxf(1.f, sqrtf(powf(phiX, 2) + powf(phiY, 2)));
        d_phiX[id] = phiX / normalizer;
        d_phiY[id] = phiY / normalizer;
        d_phiZ[id] = fmaxf(0.f, phiZ + g) - g;
    }
}


__global__ void update_dual_tex(float *d_vCap, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims, float SIGMA) {
    // get global idx in convex grid space
    dim3 globalIdx = globalIdx_Dim3();

    // only threads inside convex grid space computes
    if (globalIdx.x < convexGridDims.x && globalIdx.y < convexGridDims.y && globalIdx.z < convexGridDims.z) {
        // get linear index in convex grid space
        size_t id = linearize_globalIdx(globalIdx, convexGridDims);

        // get linear ids of neighbours of offset +1 in x, y and z dir
        size_t neighX = linearize_neighbour_globalIdx(globalIdx, convexGridDims, make_int3(1, 0, 0));
        size_t neighY = linearize_neighbour_globalIdx(globalIdx, convexGridDims, make_int3(0, 1, 0));
        size_t neighZ = linearize_neighbour_globalIdx(globalIdx, convexGridDims, make_int3(0, 0, 1));

        // chalculate differentials along x, y and z of primal variable cap and update of dual variable without projection
        float v = d_vCap[id];
        float phiX = d_phiX[id] + SIGMA * ((globalIdx.x + 1) < convexGridDims.x ? d_vCap[neighX] - v : 0);
        float phiY = d_phiY[id] + SIGMA * ((globalIdx.y + 1) < convexGridDims.y ? d_vCap[neighY] - v : 0);
        float phiZ = d_phiZ[id] + SIGMA * ((globalIdx.z + 1) < convexGridDims.z ? d_vCap[neighZ] - v : 0);

        // update the dual variable with projection back to set K
        float g = tex2D(texRef2D, globalIdx.x + 0.5f, globalIdx.y + (size_t) globalIdx.z * convexGridDims.y + 0.5f);
        float normalizer = fmaxf(1.f, sqrtf(powf(phiX, 2) + powf(phiY, 2)));
        d_phiX[id] = phiX / normalizer;
        d_phiY[id] = phiY / normalizer;
        d_phiZ[id] = fmaxf(0.f, phiZ + g) - g;
    }
}


__global__ void update_dual_pitch(float *d_vCap, float *d_g, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims, size_t pitch, float SIGMA) {
    // get global idx in convex grid space
    dim3 globalIdx = globalIdx_Dim3();

    // pitched dims
    dim3 convexGridDimsPitched = dim3(pitch, convexGridDims.y, convexGridDims.z);
    
    // only threads inside convex grid space computes
    if (globalIdx.x < convexGridDims.x && globalIdx.y < convexGridDims.y && globalIdx.z < convexGridDims.z) {
        // get linear index in convex grid space
        size_t id = linearize_globalIdx(globalIdx, convexGridDimsPitched);

        // get linear ids of neighbours of offset +1 in x, y and z dir
        size_t neighX = linearize_neighbour_globalIdx(globalIdx, convexGridDimsPitched, make_int3(1, 0, 0));
        size_t neighY = linearize_neighbour_globalIdx(globalIdx, convexGridDimsPitched, make_int3(0, 1, 0));
        size_t neighZ = linearize_neighbour_globalIdx(globalIdx, convexGridDimsPitched, make_int3(0, 0, 1));

        // chalculate differentials along x, y and z of primal variable cap and update of dual variable without projection
        float v = d_vCap[id];
        float phiX = d_phiX[id] + SIGMA * ((globalIdx.x + 1) < convexGridDims.x ? d_vCap[neighX] - v : 0);
        float phiY = d_phiY[id] + SIGMA * ((globalIdx.y + 1) < convexGridDims.y ? d_vCap[neighY] - v : 0);
        float phiZ = d_phiZ[id] + SIGMA * ((globalIdx.z + 1) < convexGridDims.z ? d_vCap[neighZ] - v : 0);

        // update the dual variable with projection back to set K
        float g = d_g[id];
        float normalizer = fmaxf(1.f, sqrtf(powf(phiX, 2) + powf(phiY, 2)));
        d_phiX[id] = phiX / normalizer;
        d_phiY[id] = phiY / normalizer;
        d_phiZ[id] = fmaxf(0.f, phiZ + g) - g;
    }
}


__global__ void update_dual_sm(float *d_vCap, float *d_g, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims, float SIGMA) {
    // shared memory to store blocks of both images related to the block
    extern __shared__ float s_vCap[];

    // get global idx in convex grid space
    dim3 globalIdx = globalIdx_Dim3();

    // only threads inside convex grid space computes
    if (globalIdx.x < convexGridDims.x && globalIdx.y < convexGridDims.y && globalIdx.z < convexGridDims.z) {
        // get linear index in convex grid space
        size_t id = linearize_globalIdx(globalIdx, convexGridDims);
        // get linear id of thread in block
        size_t id_block = linearize_threadIdx();

        // copy to shared memory
        s_vCap[id_block] = d_vCap[id];

        // make sure shared memory data copy is done before moving forward
        __syncthreads();

        // get linear ids of neighbours of offset +1 in x, y and z dir in shared memory space or grid space as per boundary check
        size_t neighX = (threadIdx.x < blockDim.x - 1) ? linearize_neighbour_globalIdx(threadIdx, blockDim, make_int3(1, 0, 0)) : linearize_neighbour_globalIdx(globalIdx, convexGridDims, make_int3(1, 0, 0));
        size_t neighY = (threadIdx.y < blockDim.y - 1) ? linearize_neighbour_globalIdx(threadIdx, blockDim, make_int3(0, 1, 0)) : linearize_neighbour_globalIdx(globalIdx, convexGridDims, make_int3(0, 1, 0));
        size_t neighZ = (threadIdx.z < blockDim.z - 1) ? linearize_neighbour_globalIdx(threadIdx, blockDim, make_int3(0, 0, 1)) : linearize_neighbour_globalIdx(globalIdx, convexGridDims, make_int3(0, 0, 1));

        // chalculate differentials along x, y and z of primal variable cap to update dual variable with thread block boundary check
        float v = s_vCap[id_block];
        float phiX = d_phiX[id] + SIGMA * ((globalIdx.x + 1) < convexGridDims.x ? ((threadIdx.x < blockDim.x - 1) ? s_vCap[neighX] : d_vCap[neighX]) - v : 0);
        float phiY = d_phiY[id] + SIGMA * ((globalIdx.y + 1) < convexGridDims.y ? ((threadIdx.y < blockDim.y - 1) ? s_vCap[neighY] : d_vCap[neighY]) - v : 0);
        float phiZ = d_phiZ[id] + SIGMA * ((globalIdx.z + 1) < convexGridDims.z ? ((threadIdx.z < blockDim.z - 1) ? s_vCap[neighZ] : d_vCap[neighZ]) - v : 0);

        // update the dual variable with projection back to set K
        float g = d_g[id];
        float normalizer = fmaxf(1.f, sqrtf(powf(phiX, 2) + powf(phiY, 2)));
        d_phiX[id] = phiX / normalizer;
        d_phiY[id] = phiY / normalizer;
        d_phiZ[id] = fmaxf(0.f, phiZ + g) - g;
    }
}


__global__ void update_primal_and_extrapolate(float *d_vn, float *d_phiX, float *d_phiY, float *d_phiZ, float *d_vCap, dim3 convexGridDims, float TAU) {
    // get global idx in convex grid space
    dim3 globalIdx = globalIdx_Dim3();

    // only threads inside convex grid space computes
    if (globalIdx.x < convexGridDims.x && globalIdx.y < convexGridDims.y && globalIdx.z < convexGridDims.z) {
        // get linear index in convex grid space
        size_t id = linearize_globalIdx(globalIdx, convexGridDims);

        // get linear ids of neighbours of offset -1 in x, y and z dir
        size_t neighX = linearize_neighbour_globalIdx(globalIdx, convexGridDims, make_int3(-1, 0, 0));
        size_t neighY = linearize_neighbour_globalIdx(globalIdx, convexGridDims, make_int3(0, -1, 0));
        size_t neighZ = linearize_neighbour_globalIdx(globalIdx, convexGridDims, make_int3(0, 0, -1));

        // calculate divergence
        float dxPhiX = ((globalIdx.x + 1) < convexGridDims.x ? d_phiX[id] : 0) - (globalIdx.x > 0 ? d_phiX[neighX] : 0);
        float dyPhiY = ((globalIdx.y + 1) < convexGridDims.y ? d_phiY[id] : 0) - (globalIdx.y > 0 ? d_phiY[neighY] : 0);
        float dzPhiZ = ((globalIdx.z + 1) < convexGridDims.z ? d_phiZ[id] : 0) - (globalIdx.z > 0 ? d_phiZ[neighZ] : 0);
        float divPhi = dxPhiX + dyPhiY + dzPhiZ;

        // temporary udpate calc of vn+1 with projection back to set C with boundary conditions preserved
        float vnOld = d_vn[id];
        float vnNew = globalIdx.z == 0 ? 1.f : (float)(globalIdx.z != convexGridDims.z - 1) * fminf(1.f, fmaxf(0.f, vnOld + TAU * divPhi));

        // update to actual memory
        d_vn[id] = vnNew;
        d_vCap[id] = 2.f * vnNew - vnOld;
    }
}


__global__ void update_primal_and_extrapolate_pitch(float *d_vn, float *d_phiX, float *d_phiY, float *d_phiZ, float *d_vCap, dim3 convexGridDims, size_t pitch, float TAU) {
    // get global idx in convex grid space
    dim3 globalIdx = globalIdx_Dim3();

    // pitched dims
    dim3 convexGridDimsPitched = dim3(pitch, convexGridDims.y, convexGridDims.z);

    // only threads inside convex grid space computes
    if (globalIdx.x < convexGridDims.x && globalIdx.y < convexGridDims.y && globalIdx.z < convexGridDims.z) {
        // get linear index in convex grid space
        size_t id = linearize_globalIdx(globalIdx, convexGridDimsPitched);

        // get linear ids of neighbours of offset -1 in x, y and z dir
        size_t neighX = linearize_neighbour_globalIdx(globalIdx, convexGridDimsPitched, make_int3(-1, 0, 0));
        size_t neighY = linearize_neighbour_globalIdx(globalIdx, convexGridDimsPitched, make_int3(0, -1, 0));
        size_t neighZ = linearize_neighbour_globalIdx(globalIdx, convexGridDimsPitched, make_int3(0, 0, -1));

        // calculate divergence
        float dxPhiX = ((globalIdx.x + 1) < convexGridDims.x ? d_phiX[id] : 0) - (globalIdx.x > 0 ? d_phiX[neighX] : 0);
        float dyPhiY = ((globalIdx.y + 1) < convexGridDims.y ? d_phiY[id] : 0) - (globalIdx.y > 0 ? d_phiY[neighY] : 0);
        float dzPhiZ = ((globalIdx.z + 1) < convexGridDims.z ? d_phiZ[id] : 0) - (globalIdx.z > 0 ? d_phiZ[neighZ] : 0);
        float divPhi = dxPhiX + dyPhiY + dzPhiZ;

        // temporary udpate calc of vn+1 with projection back to set C with boundary conditions preserved
        float vnOld = d_vn[id];
        float vnNew = globalIdx.z == 0 ? 1.f : (float)(globalIdx.z != convexGridDims.z - 1) * fminf(1.f, fmaxf(0.f, vnOld + TAU * divPhi));

        // update to actual memory
        d_vn[id] = vnNew;
        d_vCap[id] = 2.f * vnNew - vnOld;
    }
}


__global__ void init_primal_dual(float *d_v, float *d_vCap, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims) {
	// get global idx in convex grid space
    dim3 globalIdx = globalIdx_Dim3();

	// only threads inside convex grid space computes
    if (globalIdx.x < convexGridDims.x && globalIdx.y < convexGridDims.y && globalIdx.z < convexGridDims.z) {
    	// get linear index in convex grid space
        size_t id = linearize_globalIdx(globalIdx, convexGridDims);

        // since convex problem so doesn't matter on initialisation as long as satisfies definitions of C and K
        d_v[id] = globalIdx.z == 0 ? 1.f : 0.f;
        d_vCap[id] = 0.f;
        d_phiX[id] = 0.f;
        d_phiY[id] = 0.f;
        d_phiZ[id] = 0.f;
    }
}


__global__ void init_primal_dual_pitch(float *d_v, float *d_vCap, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims, size_t pitch) {
    // get global idx in convex grid space
    dim3 globalIdx = globalIdx_Dim3();

    // pitched dims
    dim3 convexGridDimsPitched = dim3(pitch, convexGridDims.y, convexGridDims.z);

    // only threads inside convex grid space computes
    if (globalIdx.x < convexGridDims.x && globalIdx.y < convexGridDims.y && globalIdx.z < convexGridDims.z) {
        // get linear index in convex grid space
        size_t id = linearize_globalIdx(globalIdx, convexGridDimsPitched);

        // since convex problem so doesn't matter on initialisation as long as satisfies definitions of C and K
        d_v[id] = globalIdx.z == 0 ? 1.f : 0.f;
        d_vCap[id] = 0.f;
        d_phiX[id] = 0.f;
        d_phiY[id] = 0.f;
        d_phiZ[id] = 0.f;
    }
}


__global__ void compute_depth_map(float *d_v, float *d_depthMap, dim3 convexGridDims, dim3 imgDims) {
    // get global idx in image plane (channels exclusive)
    dim3 globalIdx_XY = globalIdx_Dim2();

     // only threads inside image boundary computes
    if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
    	// get cur pos in v matrix
        size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

        // stride for moving along z axis
        size_t imgSize = imgDims.x * imgDims.y;

        // sum over z axis with thresholding
        float sum = 0.f;
        for(uint32_t z = 0; z < convexGridDims.z; z++) sum += (d_v[id + z * imgSize] > 0.5f ? 1.f : 0.f);

        // store calculated depth map
        d_depthMap[id] = sum;
    }
}


__global__ void compute_depth_map_pitch(float *d_v, float *d_depthMap, dim3 convexGridDims, dim3 imgDims, size_t pitch) {
    // get global idx in image plane (channels exclusive)
    dim3 globalIdx_XY = globalIdx_Dim2();

    // pitched dims
    dim3 imgDimsPitched = dim3(pitch, imgDims.y, 0);

     // only threads inside image boundary computes
    if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
        // get cur pos in v matrix
        size_t id = linearize_globalIdx(globalIdx_XY, imgDims);
        size_t id_pitch = linearize_globalIdx(globalIdx_XY, imgDimsPitched);

        // stride for moving along z axis
        size_t imgSize = pitch * imgDims.y;

        // sum over z axis with thresholding
        float sum = 0.f;
        for(uint32_t z = 0; z < convexGridDims.z; z++) sum += (d_v[id_pitch + z * imgSize] > 0.5f ? 1.f : 0.f);

        // store calculated depth map
        d_depthMap[id] = sum;
    }
}


void stereo_projection_PD(float *h_imgLeft, float *h_imgRight, float  *h_depthMap, dim3 imgDims, uint32_t nc, dim3 convexGridDims, uint32_t steps, float MU, float SIGMA, float TAU) {
	// some sizes in bytes
    size_t imgSizeBytes = (size_t) imgDims.x * imgDims.y * nc * sizeof(float);
    size_t convexGridSizeBytes = (size_t) convexGridDims.x * convexGridDims.y * convexGridDims.z * sizeof(float);
    size_t depthMapSizeBytes = (size_t) imgDims.x * imgDims.y * sizeof(float);

    // alloc GPU memory and copy data
    float *d_imgLeft, *d_imgRight, *d_g, *d_vn, *d_vCap, *d_phiX, *d_phiY, *d_phiZ, *d_depthMap;
    cudaMalloc((void **) &d_imgLeft, imgSizeBytes);
    CUDA_CHECK;
    cudaMemcpy(d_imgLeft, h_imgLeft, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgRight, imgSizeBytes);
    CUDA_CHECK;
    cudaMemcpy(d_imgRight, h_imgRight, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMalloc((void **) &d_g, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_vn, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_vCap, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_phiX, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_phiY, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_phiZ, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_depthMap, depthMapSizeBytes);
    CUDA_CHECK;

    // define block and grid for convex grid size
    dim3 block = dim3(16, 8, 8);
    dim3 grid = dim3((convexGridDims.x + block.x - 1) / block.x, (convexGridDims.y + block.y - 1) / block.y, (convexGridDims.z + block.z - 1) / block.z);

    //calculate data term
    calc_data_term<<<grid, block>>>(d_imgLeft, d_imgRight, d_g, imgDims, nc, convexGridDims, MU);
    CUDA_CHECK;
    // init primal dual
    init_primal_dual<<<grid, block>>>(d_vn, d_vCap, d_phiX, d_phiY, d_phiZ, convexGridDims);
    CUDA_CHECK;

    // for each time step
    for(uint32_t tStep = 0; tStep < steps; tStep++) {
    	// update dual
        update_dual<<<grid, block>>>(d_vCap, d_g, d_phiX, d_phiY, d_phiZ, convexGridDims, SIGMA);
        CUDA_CHECK;
    	// update primal and extrapolate
        update_primal_and_extrapolate<<<grid, block>>>(d_vn, d_phiX, d_phiY, d_phiZ, d_vCap, convexGridDims, TAU);
        CUDA_CHECK;
    }

    // define block and grid for computing depth map
    block = dim3(32, 32, 1);
    grid = dim3((imgDims.x + block.x - 1) / block.x, (imgDims.y + block.y - 1) / block.y, 1);

    // compute depth map
    compute_depth_map<<<grid, block>>>(d_vn, d_depthMap, convexGridDims, imgDims);
    CUDA_CHECK;

    // copy back data
    cudaMemcpy(h_depthMap, d_depthMap, depthMapSizeBytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // free allocations
    cudaFree(d_imgLeft);
    CUDA_CHECK;
    cudaFree(d_imgRight);
    CUDA_CHECK;
    cudaFree(d_g);
    CUDA_CHECK;
    cudaFree(d_vn);
    CUDA_CHECK;
    cudaFree(d_vCap);
    CUDA_CHECK;
    cudaFree(d_phiX);
    CUDA_CHECK;
    cudaFree(d_phiY);
    CUDA_CHECK;
    cudaFree(d_phiZ);
    CUDA_CHECK;
    cudaFree(d_depthMap);
    CUDA_CHECK;
}


void stereo_projection_PD_tex(float *h_imgLeft, float *h_imgRight, float  *h_depthMap, dim3 imgDims, uint32_t nc, dim3 convexGridDims, uint32_t steps, float MU, float SIGMA, float TAU) {
    // some sizes in bytes
    size_t imgSizeBytes = (size_t) imgDims.x * imgDims.y * nc * sizeof(float);
    size_t convexGridSizeBytes = (size_t) convexGridDims.x * convexGridDims.y * convexGridDims.z * sizeof(float);
    size_t depthMapSizeBytes = (size_t) imgDims.x * imgDims.y * sizeof(float);

    // alloc GPU memory and copy data
    float *d_imgLeft, *d_imgRight, *d_g, *d_vn, *d_vCap, *d_phiX, *d_phiY, *d_phiZ, *d_depthMap;
    cudaMalloc((void **) &d_imgLeft, imgSizeBytes);
    CUDA_CHECK;
    cudaMemcpy(d_imgLeft, h_imgLeft, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgRight, imgSizeBytes);
    CUDA_CHECK;
    cudaMemcpy(d_imgRight, h_imgRight, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMalloc((void **) &d_g, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_vn, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_vCap, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_phiX, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_phiY, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_phiZ, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_depthMap, depthMapSizeBytes);
    CUDA_CHECK;

    // define block and grid for convex grid size
    dim3 block = dim3(16, 8, 8);
    dim3 grid = dim3((convexGridDims.x + block.x - 1) / block.x, (convexGridDims.y + block.y - 1) / block.y, (convexGridDims.z + block.z - 1) / block.z);

    //calculate data term
    calc_data_term<<<grid, block>>>(d_imgLeft, d_imgRight, d_g, imgDims, nc, convexGridDims, MU);
    CUDA_CHECK;

    // bind data term to 2D texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, texRef2D, d_g, desc, convexGridDims.x, convexGridDims.y * convexGridDims.z, convexGridDims.x * sizeof(float));
    CUDA_CHECK;

    // init primal dual
    init_primal_dual<<<grid, block>>>(d_vn, d_vCap, d_phiX, d_phiY, d_phiZ, convexGridDims);
    CUDA_CHECK;

    // for each time step
    for(uint32_t tStep = 0; tStep < steps; tStep++) {
        // update dual
        update_dual_tex<<<grid, block>>>(d_vCap, d_phiX, d_phiY, d_phiZ, convexGridDims, SIGMA);
        CUDA_CHECK;
        // update primal and extrapolate
        update_primal_and_extrapolate<<<grid, block>>>(d_vn, d_phiX, d_phiY, d_phiZ, d_vCap, convexGridDims, TAU);
        CUDA_CHECK;
    }

    // define block and grid for computing depth map
    block = dim3(32, 32, 1);
    grid = dim3((imgDims.x + block.x - 1) / block.x, (imgDims.y + block.y - 1) / block.y, 1);

    // compute depth map
    compute_depth_map<<<grid, block>>>(d_vn, d_depthMap, convexGridDims, imgDims);
    CUDA_CHECK;

    // copy back data
    cudaMemcpy(h_depthMap, d_depthMap, depthMapSizeBytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // free allocations
    cudaFree(d_imgLeft);
    CUDA_CHECK;
    cudaFree(d_imgRight);
    CUDA_CHECK;
    cudaFree(d_g);
    CUDA_CHECK;
    cudaFree(d_vn);
    CUDA_CHECK;
    cudaFree(d_vCap);
    CUDA_CHECK;
    cudaFree(d_phiX);
    CUDA_CHECK;
    cudaFree(d_phiY);
    CUDA_CHECK;
    cudaFree(d_phiZ);
    CUDA_CHECK;
    cudaFree(d_depthMap);
    CUDA_CHECK;
}


void stereo_projection_PD_pitch(float *h_imgLeft, float *h_imgRight, float  *h_depthMap, dim3 imgDims, uint32_t nc, dim3 convexGridDims, uint32_t steps, float MU, float SIGMA, float TAU) {
    // some sizes in bytes
    size_t imgSizeBytes = (size_t) imgDims.x * imgDims.y * nc * sizeof(float);
    size_t depthMapSizeBytes = (size_t) imgDims.x * imgDims.y * sizeof(float);
    size_t widthBytes = (size_t) convexGridDims.x * sizeof(float);
    size_t height_x_depth = (size_t) convexGridDims.y * convexGridDims.z;

    // alloc GPU memory and copy data
    float *d_imgLeft, *d_imgRight, *d_g, *d_vn, *d_vCap, *d_phiX, *d_phiY, *d_phiZ, *d_depthMap;
    size_t pitchBytes = 0, pitch = 0;
    cudaMalloc((void **) &d_imgLeft, imgSizeBytes);
    CUDA_CHECK;
    cudaMemcpy(d_imgLeft, h_imgLeft, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgRight, imgSizeBytes);
    CUDA_CHECK;
    cudaMemcpy(d_imgRight, h_imgRight, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMallocPitch((void **) &d_g, &pitchBytes, widthBytes, height_x_depth);
    CUDA_CHECK;
    cudaMallocPitch((void **) &d_vn, &pitchBytes, widthBytes, height_x_depth);
    CUDA_CHECK;
    cudaMallocPitch((void **) &d_vCap,&pitchBytes, widthBytes, height_x_depth);
    CUDA_CHECK;
    cudaMallocPitch((void **) &d_phiX, &pitchBytes, widthBytes, height_x_depth);
    CUDA_CHECK;
    cudaMallocPitch((void **) &d_phiY, &pitchBytes, widthBytes, height_x_depth);
    CUDA_CHECK;
    cudaMallocPitch((void **) &d_phiZ, &pitchBytes, widthBytes, height_x_depth);
    CUDA_CHECK;
    cudaMalloc((void **) &d_depthMap, depthMapSizeBytes);
    CUDA_CHECK;

    // pitch length as multiple of sizeof(float) (always true as smallest pitch size is 32 in cuda)
    pitch = pitchBytes / sizeof(float);

    // define block and grid for convex grid size
    dim3 block = dim3(16, 8, 8);
    dim3 grid = dim3((convexGridDims.x + block.x - 1) / block.x, (convexGridDims.y + block.y - 1) / block.y, (convexGridDims.z + block.z - 1) / block.z);

    //calculate data term
    calc_data_term_pitch<<<grid, block>>>(d_imgLeft, d_imgRight, d_g, imgDims, nc, convexGridDims, pitch, MU);
    CUDA_CHECK;
    // init primal dual
    init_primal_dual_pitch<<<grid, block>>>(d_vn, d_vCap, d_phiX, d_phiY, d_phiZ, convexGridDims, pitch);
    CUDA_CHECK;

    // for each time step
    for(uint32_t tStep = 0; tStep < steps; tStep++) {
        // update dual
        update_dual_pitch<<<grid, block>>>(d_vCap, d_g, d_phiX, d_phiY, d_phiZ, convexGridDims, pitch, SIGMA);
        CUDA_CHECK;
        // update primal and extrapolate
        update_primal_and_extrapolate_pitch<<<grid, block>>>(d_vn, d_phiX, d_phiY, d_phiZ, d_vCap, convexGridDims, pitch, TAU);
        CUDA_CHECK;
    }

    // define block and grid for computing depth map
    block = dim3(32, 32, 1);
    grid = dim3((imgDims.x + block.x - 1) / block.x, (imgDims.y + block.y - 1) / block.y, 1);

    // compute depth map
    compute_depth_map_pitch<<<grid, block>>>(d_vn, d_depthMap, convexGridDims, imgDims, pitch);
    CUDA_CHECK;

    // copy back data
    cudaMemcpy(h_depthMap, d_depthMap, depthMapSizeBytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // free allocations
    cudaFree(d_imgLeft);
    CUDA_CHECK;
    cudaFree(d_imgRight);
    CUDA_CHECK;
    cudaFree(d_g);
    CUDA_CHECK;
    cudaFree(d_vn);
    CUDA_CHECK;
    cudaFree(d_vCap);
    CUDA_CHECK;
    cudaFree(d_phiX);
    CUDA_CHECK;
    cudaFree(d_phiY);
    CUDA_CHECK;
    cudaFree(d_phiZ);
    CUDA_CHECK;
    cudaFree(d_depthMap);
    CUDA_CHECK;
}


void stereo_projection_PD_sm(float *h_imgLeft, float *h_imgRight, float  *h_depthMap, dim3 imgDims, uint32_t nc, dim3 convexGridDims, uint32_t steps, float MU, float SIGMA, float TAU) {
    // some sizes in bytes
    size_t imgSizeBytes = (size_t) imgDims.x * imgDims.y * nc * sizeof(float);
    size_t convexGridSizeBytes = (size_t) convexGridDims.x * convexGridDims.y * convexGridDims.z * sizeof(float);
    size_t depthMapSizeBytes = (size_t) imgDims.x * imgDims.y * sizeof(float);

    // alloc GPU memory and copy data
    float *d_imgLeft, *d_imgRight, *d_g, *d_vn, *d_vCap, *d_phiX, *d_phiY, *d_phiZ, *d_depthMap;
    cudaMalloc((void **) &d_imgLeft, imgSizeBytes);
    CUDA_CHECK;
    cudaMemcpy(d_imgLeft, h_imgLeft, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMalloc((void **) &d_imgRight, imgSizeBytes);
    CUDA_CHECK;
    cudaMemcpy(d_imgRight, h_imgRight, imgSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMalloc((void **) &d_g, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_vn, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_vCap, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_phiX, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_phiY, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_phiZ, convexGridSizeBytes);
    CUDA_CHECK;
    cudaMalloc((void **) &d_depthMap, depthMapSizeBytes);
    CUDA_CHECK;

    // define block and grid for convex grid size
    dim3 block = dim3(16, 8, 8);
    dim3 grid = dim3((convexGridDims.x + block.x - 1) / block.x, (convexGridDims.y + block.y - 1) / block.y, (convexGridDims.z + block.z - 1) / block.z);

    // some shared shared memory sizes for different kernels (in bytes)
    size_t sm_dataTermBytes = (size_t) block.x * block.y * nc * sizeof(float) * 2;
    size_t sm_updateDualBytes = (size_t) block.x * block.y * block.z * sizeof(float);

    //calculate data term
    calc_data_term_sm<<<grid, block, sm_dataTermBytes>>>(d_imgLeft, d_imgRight, d_g, imgDims, nc, convexGridDims, MU);
    CUDA_CHECK;
    // init primal dual
    init_primal_dual<<<grid, block>>>(d_vn, d_vCap, d_phiX, d_phiY, d_phiZ, convexGridDims);
    CUDA_CHECK;

    // for each time step
    for(uint32_t tStep = 0; tStep < steps; tStep++) {
        // update dual
        update_dual_sm<<<grid, block, sm_updateDualBytes>>>(d_vCap, d_g, d_phiX, d_phiY, d_phiZ, convexGridDims, SIGMA);
        CUDA_CHECK;
        // update primal and extrapolate
        update_primal_and_extrapolate<<<grid, block>>>(d_vn, d_phiX, d_phiY, d_phiZ, d_vCap, convexGridDims, TAU);
        CUDA_CHECK;
    }

    // define block and grid for computing depth map
    block = dim3(32, 32, 1);
    grid = dim3((imgDims.x + block.x - 1) / block.x, (imgDims.y + block.y - 1) / block.y, 1);

    // compute depth map
    compute_depth_map<<<grid, block>>>(d_vn, d_depthMap, convexGridDims, imgDims);
    CUDA_CHECK;

    // copy back data
    cudaMemcpy(h_depthMap, d_depthMap, depthMapSizeBytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // free allocations
    cudaFree(d_imgLeft);
    CUDA_CHECK;
    cudaFree(d_imgRight);
    CUDA_CHECK;
    cudaFree(d_g);
    CUDA_CHECK;
    cudaFree(d_vn);
    CUDA_CHECK;
    cudaFree(d_vCap);
    CUDA_CHECK;
    cudaFree(d_phiX);
    CUDA_CHECK;
    cudaFree(d_phiY);
    CUDA_CHECK;
    cudaFree(d_phiZ);
    CUDA_CHECK;
    cudaFree(d_depthMap);
    CUDA_CHECK;
}
