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


__global__ void calc_data_term(float *d_imgLeft, float *d_imgRight, float *d_g, dim3 imgDims, uint32_t nc, dim3 convexGridDims, float MU) {
    // get global idx in convex grid space
    dim3 globalIdx = globalIdx_Dim3();
    // get global idx in image plane (channels exclusive)
    dim3 globalIdx_XY = globalIdx_Dim2();

    // only threads inside convex grid space computes
    if (globalIdx.x < convexGridDims.x && globalIdx.y < convexGridDims.y && globalIdx.z < convexGridDims.z) {
		// get linear index in convex grid space
        size_t id = linearize_globalIdx(globalIdx, convexGridDims);
    	// get linear index in XY
        size_t id_XY = linearize_globalIdx(globalIdx_XY, imgDims);

        // to store calc of data term for current thread
        float g = 0.f;

        // for all channels
        for(uint32_t ch_i = 0; ch_i < nc; ch_i++) {
            // channel offset
            size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

            // get linear ids of shifted pixel in right image with clamping
            size_t shiftedPixel = linearize_neighbour_globalIdx(globalIdx_XY, imgDims, make_int3(-globalIdx.z, 0, 0), CLAMP);

            // calculate difference in intensity for current channel and shift
            g += fabsf(d_imgLeft[id_XY + chOffset] - d_imgRight[shiftedPixel + chOffset]);
        }

        // store final calculation
        d_g[id] = MU * g;
    }
}


__global__ void update_dual(float *d_vCap, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims, float SIGMA) {
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
        float projDenominator = fmaxf(1.f, sqrtf(powf(phiX, 2) + powf(phiY, 2) + powf(phiZ, 2)));
        d_phiX[id] = phiX / projDenominator;
        d_phiY[id] = phiY / projDenominator;
        d_phiZ[id] = phiZ / projDenominator;
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
        float3 q = make_float3(phiX, phiY, phiZ + d_g[id]);
        float3 p = make_float3(q.x / fmaxf(1.f, sqrtf(powf(q.x, 2) + powf(q.y, 2))), q.y / fmaxf(1.f, sqrtf(powf(q.x, 2) + powf(q.y, 2))), fmaxf(0.f, q.z));
        d_phiX[id] = p.x;
        d_phiY[id] = p.y;
        d_phiZ[id] = p.z - d_g[id];
    }
}


__global__ void update_primal(float *d_vn, float *d_phiX, float *d_phiY, float *d_phiZ, float *d_g, float *d_vnp, dim3 convexGridDims, float TAU) {
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

        // udpate vn+1 with projection back to set C
        d_vnp[id] = fminf(1.f, fmaxf(0.f, d_vn[id] - TAU * (divPhi + d_g[id])));
    }
}


__global__ void update_primal(float *d_vn, float *d_phiX, float *d_phiY, float *d_phiZ, float *d_vnp, dim3 convexGridDims, float TAU) {
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

        // udpate vn+1 with projection back to set C
        d_vnp[id] = fminf(1.f, fmaxf(0.f, d_vn[id] - TAU * divPhi));
    }
}


__global__ void update_primal_cap(float *d_vn, float *d_vnp, float *d_vCap, dim3 convexGridDims) {
	// get global idx in convex grid space
    dim3 globalIdx = globalIdx_Dim3();

	// only threads inside convex grid space computes
    if (globalIdx.x < convexGridDims.x && globalIdx.y < convexGridDims.y && globalIdx.z < convexGridDims.z) {
    	// get linear index in convex grid space
        size_t id = linearize_globalIdx(globalIdx, convexGridDims);

        // update v-cap
        d_vCap[id] = 2.f * d_vnp[id] - d_vn[id];
    }
}


__global__ void z(float *d_g, float *d_v, float *d_vCap, float *d_phiX, float *d_phiY, float *d_phiZ, dim3 convexGridDims) {
	// get global idx in convex grid space
    dim3 globalIdx = globalIdx_Dim3();

	// only threads inside convex grid space computes
    if (globalIdx.x < convexGridDims.x && globalIdx.y < convexGridDims.y && globalIdx.z < convexGridDims.z) {
    	// get linear index in convex grid space
        size_t id = linearize_globalIdx(globalIdx, convexGridDims);

        // since convex problem so doesn't matter on initialisation as long as satisfies definitions of C and K
        d_v[id] = globalIdx.z == convexGridDims.z - 1 ? 0.f : d_g[id];
        d_vCap[id] = globalIdx.z == convexGridDims.z - 1 ? 0.f : d_g[id];
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

        // sum over z axis
        float sum = 0.f;
        for(uint32_t z = 0; z < convexGridDims.z; z++) sum += d_v[id + z * imgSize];

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
    float *d_imgLeft, *d_imgRight, *d_g, *d_vn, *d_vnp, *d_vCap, *d_phiX, *d_phiY, *d_phiZ, *d_depthMap;
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
    cudaMalloc((void **) &d_vnp, convexGridSizeBytes);
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
    dim3 block = dim3(8, 8, 8);
    dim3 grid = dim3((convexGridDims.x + block.x - 1) / block.x, (convexGridDims.y + block.y - 1) / block.y, (convexGridDims.z + block.z - 1) / block.z);

    //calculate data term
    calc_data_term<<<grid, block>>>(d_imgLeft, d_imgRight, d_g, imgDims, nc, convexGridDims, MU);
    // init primal dual
    //init_primal_dual<<<grid, block>>>(d_vn, d_vCap, d_phiX, d_phiY, d_phiZ, convexGridDims);
    init_primal_dual<<<grid, block>>>(d_g, d_vn, d_vCap, d_phiX, d_phiY, d_phiZ, convexGridDims);

    // for each time step
    for(uint32_t tStep = 0; tStep < steps; tStep++) {
    	// update dual
    	//update_dual<<<grid, block>>>(d_vCap, d_phiX, d_phiY, d_phiZ, convexGridDims, SIGMA);
        update_dual<<<grid, block>>>(d_vCap, d_g, d_phiX, d_phiY, d_phiZ, convexGridDims, SIGMA);
    	// update primal
    	//update_primal<<<grid, block>>>(d_vn, d_phiX, d_phiY, d_phiZ, d_g, d_vnp, convexGridDims, TAU);
        update_primal<<<grid, block>>>(d_vn, d_phiX, d_phiY, d_phiZ, d_vnp, convexGridDims, TAU);
    	// update primal bar
    	update_primal_cap<<<grid, block>>>(d_vn, d_vnp, d_vCap, convexGridDims);
    	
    	// swap pointers
        float *temp = d_vn;
        d_vn = d_vnp;
        d_vnp = temp;
    }

    // define block and grid for computing depth map
    block = dim3(16, 16, 1);
    grid = dim3((imgDims.x + block.x - 1) / block.x, (imgDims.y + block.y - 1) / block.y, 1);

    // compute depth map
    compute_depth_map<<<grid, block>>>(d_vn, d_depthMap, convexGridDims, imgDims);

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
    cudaFree(d_vnp);
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

