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
		float tau, uint32_t steps, uint32_t diffType) {
	// size with channels
	size_t imgSizeBytes = (size_t) imgDims.x * imgDims.y * nc * sizeof(float);
	size_t imgOutSizeBytes = (size_t) imgDims.x * imgDims.y * ncOut
			* sizeof(float);
	// alloc GPU memory and copy data
	float *d_imgInleft;
	float *d_imgInright;
	float *d_imgOutNew, *d_imgOutOld, *d_imgOutFit, *d_f, *d_zetaX, *d_zetaY;

	cudaMalloc((void **) &d_imgInleft, imgSizeBytes);
	CUDA_CHECK;
	cudaMemcpy(d_imgInleft, h_imgInleft, imgSizeBytes, cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMalloc((void **) &d_imgInright, imgSizeBytes);
	CUDA_CHECK;
	cudaMemcpy(d_imgInright, h_imgInright, imgSizeBytes,
			cudaMemcpyHostToDevice);
	CUDA_CHECK;

	cudaMalloc((void **) &d_imgOutNew, imgOutSizeBytes);
	CUDA_CHECK;
	cudaMalloc((void **) &d_imgOutOld, imgOutSizeBytes);
	CUDA_CHECK;
	cudaMalloc((void **) &d_imgOutFit, imgOutSizeBytes);
	CUDA_CHECK;
	cudaMalloc((void **) &d_f, imgOutSizeBytes);
	CUDA_CHECK;
	// define block and grid
	dim3 block = dim3(16, 16, 1);
	dim3 grid = dim3((imgDims.x + block.x - 1) / block.x,
			(imgDims.y + block.y - 1) / block.y, 1);

	dataTerm<<<grid, block>>>(d_f, d_imgInleft, d_imgInright, nc, imgDims,
			d_imgOutOld, d_imgOutFit);
	/*	cudaMalloc((void **) &d_imgGradXleft, imgSizeBytes);
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
	 CUDA_CHECK;*/

	cudaMalloc((void **) &d_zetaX, imgOutSizeBytes);
	CUDA_CHECK;
	cudaMalloc((void **) &d_zetaY, imgOutSizeBytes);
	CUDA_CHECK;

	initialize_zeta<<<grid, block>>>(d_imgOutOld, ncOut, d_zetaX, d_zetaY,
			imgDims);

	std::cout<< "hey steps" << steps << std::endl;
	// for each time step
	for (uint32_t tStep = 0; tStep < steps; tStep++) {
		regularizer_update<<<grid, block>>>(d_zetaX, d_zetaY, d_imgOutFit,
				ncOut, sigma, imgDims);

		variational_update<<<grid, block>>>(d_imgOutNew, d_imgOutOld, d_zetaX,
				d_zetaY, d_f, d_imgOutFit, imgDims, ncOut, tau);
		float *temp = d_imgOutOld;
		d_imgOutOld = d_imgOutNew;
		d_imgOutNew = temp;
	}

	// copy back data
	cudaMemcpy(h_imgOut, d_imgOutOld, imgOutSizeBytes, cudaMemcpyDeviceToHost);
	CUDA_CHECK;

	// free allocations
	cudaFree(d_imgInleft);
	CUDA_CHECK;
	/*	cudaFree(d_imgGradXleft);
	 CUDA_CHECK;
	 cudaFree(d_imgGradYleft);
	 CUDA_CHECK;
	 cudaFree(d_imgGradAbsleft);
	 CUDA_CHECK;*/
	cudaFree(d_imgInright);
	CUDA_CHECK;
	/*	cudaFree(d_imgGradXright);
	 CUDA_CHECK;
	 cudaFree(d_imgGradYright);
	 CUDA_CHECK;
	 cudaFree(d_imgGradAbsright);
	 CUDA_CHECK;*/
	cudaFree(d_imgOutNew);
	CUDA_CHECK;
	cudaFree(d_imgOutOld);
	CUDA_CHECK;
	cudaFree(d_f);
	CUDA_CHECK;
	cudaFree(d_zetaX);
	CUDA_CHECK;
	cudaFree(d_zetaY);
	CUDA_CHECK;
}

__global__ void dataTerm(float *d_f, float *d_imgInleft, float *d_imgInright,
		uint32_t nc, dim3 imgDims, float *d_imgOutOld, float *d_imgOutFit) {
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
		d_imgOutOld[id] = d_f[id];
		d_imgOutFit[id] = d_f[id];
	}
}

__global__ void initialize_zeta(float *d_imgOutOld, size_t ncOut,
		float *d_zetaX, float *d_zetaY, dim3 imgDims) {
	// get global idx in XY (channels exclusive)
	dim3 globalIdx_XY = globalIdx_Dim2();

	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

		float dZetaX, dZetaY, dZetaNorm = 0.f;
		// for all channels
		for (uint32_t ch_i = 0; ch_i < ncOut; ch_i++) {
			// channel offset
			size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

			gradient_imgFd(&dZetaX, &dZetaY, d_imgOutOld, globalIdx_XY, ch_i,
					imgDims);
			d_zetaX[id + chOffset] = dZetaX;
			d_zetaY[id + chOffset] = dZetaY;

			dZetaNorm += pow(d_zetaX[id + chOffset], 2)
					+ pow(d_zetaY[id + chOffset], 2);
		}

		float len_Vector = 1.f / sqrt(dZetaNorm);
		for (uint32_t ch_i = 0; ch_i < ncOut; ch_i++) {
			// channel offset
			size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

			d_zetaX[id + chOffset] *= len_Vector;
			d_zetaY[id + chOffset] *= len_Vector;
		}
	}
}

__global__ void regularizer_update(float *d_zetaX, float *d_zetaY,
		float *d_imgOutFit, float ncOut, float sigma, dim3 imgDims) {
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
			gradient_imgFd(&d_imgGradX, &d_imgGradY, d_imgOutFit, globalIdx_XY,
					ch_i, imgDims);
			d_zetaX[id + chOffset] = d_zetaX[id + chOffset]
					- sigma * d_imgGradX;
			d_zetaY[id + chOffset] = d_zetaY[id + chOffset]
					- sigma * d_imgGradY;

			//Projecting zeta onto K subspace
			d_zetaAbs += pow(d_zetaX[id + chOffset], 2)
					+ pow(d_zetaY[id + chOffset], 2);
		}
		float proj_scaleFactor = 1.f / max(1.f, sqrt(d_zetaAbs));
		for (uint32_t ch_i = 0; ch_i < ncOut; ch_i++) {
			// channel offset
			size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

			d_zetaX[id + chOffset] *= proj_scaleFactor;
			d_zetaY[id + chOffset] *= proj_scaleFactor;
		}
	}
}

__device__ void gradient_imgFd(float *d_imgGradX, float *d_imgGradY,
		float *d_imgOut, dim3 globalIdx_XY, size_t ch_i, dim3 imgDims) {
	// get linear index
	size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

	// get linear ids of neighbours of offset +1 in x and y dir
	size_t neighX = linearize_neighbour_globalIdx(globalIdx_XY, imgDims,
			make_int3(1, 0, 0));
	size_t neighY = linearize_neighbour_globalIdx(globalIdx_XY, imgDims,
			make_int3(0, 1, 0));
	size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

	// chalculate differentials along x and y
	*d_imgGradX =
			(globalIdx_XY.x + 1) < imgDims.x ?
					(d_imgOut[neighX + chOffset] - d_imgOut[id + chOffset]) : 0;
	*d_imgGradY =
			(globalIdx_XY.y + 1) < imgDims.y ?
					(d_imgOut[neighY + chOffset] - d_imgOut[id + chOffset]) : 0;

}

__global__ void variational_update(float *d_imgOutNew, float *d_imgOutOld,
		float *d_zetaX, float *d_zetaY, float *d_f, float* d_imgOutFit,
		dim3 imgDims, size_t ncOut, float tau) {
	// get global idx in XY (channels exclusive)
	dim3 globalIdx_XY = globalIdx_Dim2();

	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);
		float div_zeta;
		divergence_zeta(&div_zeta, d_zetaX, d_zetaY, globalIdx_XY, ncOut,
				imgDims);
		float output;
		// for all channels
		for (uint32_t ch_i = 0; ch_i < ncOut; ch_i++) {
			// channel offset
			size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;
			output = d_imgOutOld[id + chOffset]
					- tau * (div_zeta + d_f[id + chOffset]);

			//Projection onto the C space --clipping
			output = output < 1 ? output : 1;
			output = output > 0 ? output : 0;
			d_imgOutNew[id + chOffset] = output;

			//Calculate the image fitting
			d_imgOutFit[id + chOffset] = 2 * d_imgOutNew[id + chOffset]
					- d_imgOutOld[id + chOffset];

		}
	}
}

__device__ void divergence_zeta(float *div_zeta, float *d_zetaX, float *d_zetaY,
		dim3 globalIdx_XY, float ncOut, dim3 imgDims) {
	// only threads inside image boundary computes
	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

		*div_zeta = 0;
		for (uint32_t ch_i = 0; ch_i < ncOut; ch_i++) {
			// channel offset
			size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;

			// get linear ids of neighbours of offset -1 in x and y dir
			size_t neighX = linearize_neighbour_globalIdx(globalIdx_XY, imgDims,
					make_int3(-1, 0, 0));
			size_t neighY = linearize_neighbour_globalIdx(globalIdx_XY, imgDims,
					make_int3(0, -1, 0));

			// calculate divergence for the current pixel using backward difference
			float dxxU = (
					(globalIdx_XY.x + 1) < imgDims.x ?
							d_zetaX[id + chOffset] : 0)
					- (globalIdx_XY.x > 0 ? d_zetaX[neighX + chOffset] : 0);
			float dyyU = (
					(globalIdx_XY.y + 1) < imgDims.y ?
							d_zetaY[id + chOffset] : 0)
					- (globalIdx_XY.y > 0 ? d_zetaY[neighY + chOffset] : 0);
			*div_zeta += dxxU + dyyU;
		}
	}
}

