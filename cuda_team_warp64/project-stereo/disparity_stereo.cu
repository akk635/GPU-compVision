/*
 * disparity_stereo.cu
 *
 *  Created on: Mar 14, 2014
 *      Author: p054
 */
#include "disparity_stereo.h"
// FIX
#include <global_idx.h>
#include <global_idx.cu>

void disparity_computation_caller(float *h_imgInleft, float *h_imgInright,
		float *h_imgOut, dim3 imgDims, uint32_t nc, uint32_t ncOut, float sigma,
		float tau, uint32_t diffType) {
	// size with channels
	size_t imgSizeBytes = (size_t) imgDims.x * imgDims.y * nc * sizeof(float);
	size_t imgOutSizeBytes = (size_t) imgDims.x * imgDims.y * ncOut
			* sizeof(float);
	// alloc GPU memory and copy data
	float *d_imgInleft, *d_imgGradXleft, *d_imgGradYleft, *d_imgGradAbsleft;
	float *d_imgInright, *d_imgGradXright, *d_imgGradYright, *d_imgGradAbsright;
	float *d_imgOut, *d_imgOutFit, *d_f, *d_zetaX, *d_zetaY;
	cudaMalloc((void **) &d_imgInleft, imgSizeBytes);
	CUDA_CHECK;
	cudaMemcpy(d_imgInleft, h_imgInleft, imgSizeBytes, cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMalloc((void **) &d_imgInright, imgSizeBytes);
	CUDA_CHECK;
	cudaMemcpy(d_imgInright, h_imgInright, imgSizeBytes,
			cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMalloc((void **) &d_imgGradXleft, imgSizeBytes);
	CUDA_CHECK;
	cudaMalloc((void **) &d_imgGradYleft, imgSizeBytes);
	CUDA_CHECK;
	cudaMalloc((void **) &d_imgGradAbsleft, imgSizeBytes / nc);
	CUDA_CHECK;
	cudaMalloc((void **) &d_imgGradXright, imgSizeBytes);
	CUDA_CHECK;
	cudaMalloc((void **) &d_imgGradYright, imgSizeBytes);
	CUDA_CHECK;
	cudaMalloc((void **) &d_imgGradAbsright, imgSizeBytes / nc);
	CUDA_CHECK;

	cudaMalloc((void **) &d_imgOut, imgOutSizeBytes);
	CUDA_CHECK;
	cudaMalloc((void **) &d_imgOutFit, imgOutSizeBytes);
	CUDA_CHECK;
	cudaMalloc((void **) &d_zetaX, imgOutSizeBytes);
	CUDA_CHECK;
	cudaMalloc((void **) &d_zetaY, imgOutSizeBytes);
	CUDA_CHECK;

	// define block and grid
	dim3 block = dim3(16, 16, 1);
	dim3 grid = dim3((imgDims.x + block.x - 1) / block.x,
			(imgDims.y + block.y - 1) / block.y, 1);

	dataTerm<<<grid, block>>>(d_f, d_imgInleft, d_imgInright, nc, imgDims);

	initialize<<<grid, block>>>(d_imgOut, d_imgOutFit, ncOut, d_zetaX, d_zetaY);

	regularizer_update<<<grid, block>>>(d_zetaX, d_zetaY, d_imgOutFit, ncOut,
			sigma, imgDims);

	variational_update<<<grid, block>>>(d_imgOut, d_zetaX, d_zetaY, d_f,
			d_imgOutFit, imgDims, ncOut);

	// copy back data
	cudaMemcpy(h_imgOut, d_f, imgOutSizeBytes, cudaMemcpyDeviceToHost);
	CUDA_CHECK;

	// free allocations
	cudaFree(d_imgInleft);
	CUDA_CHECK;
	cudaFree(d_imgGradXleft);
	CUDA_CHECK;
	cudaFree(d_imgGradYleft);
	CUDA_CHECK;
	cudaFree(d_imgGradAbsleft);
	CUDA_CHECK;
	cudaFree(d_imgInright);
	CUDA_CHECK;
	cudaFree(d_imgGradXright);
	CUDA_CHECK;
	cudaFree(d_imgGradYright);
	CUDA_CHECK;
	cudaFree(d_imgGradAbsright);
	CUDA_CHECK;
	cudaFree(d_imgOut);
	CUDA_CHECK;
	cudaFree(d_f);
	CUDA_CHECK;
	cudaFree(d_zetaX);
	CUDA_CHECK;
	cudaFree(d_zetaY);
	CUDA_CHECK;
}

__global__ void dataTerm(float *d_f, float *d_imgInleft, float *d_imgInright,
		uint32_t nc, dim3 imgDims) {
	// get global idx in XY (channels exclusive)
	dim3 globalIdx_XY = globalIdx_Dim2();

	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);
		d_f[id] = 0.f;
		// for all channels
		for (uint32_t ch_i = 0; ch_i < nc; ch_i++) {
			// channel offset
			size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

			d_f[id] += fabsf(
					d_imgInleft[id + chOffset] - d_imgInright[id + chOffset]);
		}
	}
}

__global__ void initialize(float *d_imgOut, float *d_imgOutFit, size_t ncOut,
		float *d_zetaX, float *d_zetaY) {
	// get global idx in XY (channels exclusive)
	dim3 globalIdx_XY = globalIdx_Dim2();

	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

		// for all channels
		for (uint32_t ch_i = 0; ch_i < ncOut; ch_i++) {
			// channel offset
			size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

			d_imgOut[id + chOffset] = 0.f;
			d_imgOutFit[id + chOffset] = 0.f;
			d_zetaX[id + chOffset] = 0.f;
			d_zetaY[id + chOffset] = 0.f;
		}
	}
}

__global__ void regularizer_update(float *d_zetaX, float *d_zetaY,
		float *d_imgOutFit, float ncOut, float sigma, float imgDims) {
	// get global idx in XY (channels exclusive)
	dim3 globalIdx_XY = globalIdx_Dim2();

	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);
		float d_imgGradX, d_imgGradY, d_zetaAbs = 0.f;
		// for all channels
		for (uint32_t ch_i = 0; ch_i < ncOut; ch_i++) {
			// channel offset
			size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;
			gradient_imgFit(&d_imgGradX, &d_imgGradY, d_imgOutFit, globalIdx_XY,
					ch_i);
			d_zetaX[id + chOffset] = d_zetaX[id + chOffset]
					- sigma * d_imgGradX;
			d_zetaY[id + chOffset] = d_zetaY[id + chOffset]
					- sigma * d_imgGradY;

			//Projecting zeta onto K subspace
			d_zetaAbs += pow(d_zetaX[id + chOffset], 2)
					+ pow(d_zetaY[id + chOffset], 2);
		}
		float proj_scaleFactor = 1.f / max(1.f, d_zetaAbs);
		for (uint32_t ch_i = 0; ch_i < ncOut; ch_i++) {
			d_zetaX *= proj_scaleFactor;
			d_zetaY *= proj_scaleFactor;
		}
	}
}

__device__ void gradient_imgFit(float *d_imgGradX, float *d_imgGradY,
		float *d_imgOutFit, dim3 globalIdx_XY, size_t ch_i) {
	// get linear index
	size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

	// get linear ids of neighbours of offset +1 in x and y dir
	size_t neighX = linearize_neighbour_globalIdx(globalIdx_XY, imgDims,
			make_int3(1, 0, 0));
	size_t neighY = linearize_neighbour_globalIdx(globalIdx_XY, imgDims,
			make_int3(0, 1, 0));

	// chalculate differentials along x and y
	*d_imgGradX =
			(globalIdx_XY.x + 1) < imgDims.x ?
					(d_imgOutFit[neighX + chOffset] - d_imgOutFit[id + chOffset]) :
					0;
	*d_imgGradY =
			(globalIdx_XY.y + 1) < imgDims.y ?
					(d_imgOutFit[neighY + chOffset] - d_imgOutFit[id + chOffset]) :
					0;

}

__global__ void variational_update(float *d_imgOut, float *d_zetaX,
		float *d_zetaY, float *d_f, float* d_imgOutFit, dim3 imgDims,
		size_t ncOut) {
	// get global idx in XY (channels exclusive)
	dim3 globalIdx_XY = globalIdx_Dim2();

	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);
		float div_zeta;
	divergence_zeta( &div_zeta, globalIdx_XY, d_zetaX, d_zetaY);
}
}

