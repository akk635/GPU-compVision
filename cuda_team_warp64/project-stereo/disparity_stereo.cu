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
		float tau, uint32_t steps, uint32_t mu, uint32_t disparities, float *h_f) {

	// size with channels
	size_t imgSizeBytes = (size_t) imgDims.x * imgDims.y * nc * sizeof(float);
	size_t imgOutSizeBytes = (size_t) imgDims.x * imgDims.y * ncOut
			* sizeof(float);

	std::cout << "From disparities : " << disparities << std::endl;
	// alloc GPU memory and copy data
	float *d_imgInleft;
	float *d_imgInright;
	float *d_imgOutNew[disparities], *d_imgOutOld[disparities],
			*d_imgOutFit[disparities], *d_f, *d_phiX[disparities],
			*d_phiY[disparities], *d_phiZ[disparities];

	cudaMalloc((void **) &d_imgInleft, imgSizeBytes);
	CUDA_CHECK;
	cudaMemcpy(d_imgInleft, h_imgInleft, imgSizeBytes, cudaMemcpyHostToDevice);
	CUDA_CHECK;

	cudaMalloc((void **) &d_imgInright, imgSizeBytes);
	CUDA_CHECK;
	cudaMemcpy(d_imgInright, h_imgInright, imgSizeBytes,
			cudaMemcpyHostToDevice);
	CUDA_CHECK;

	// 1D long big array allocated
	cudaMalloc((void **) &d_f, imgOutSizeBytes * disparities);
	CUDA_CHECK;

	for (uint32_t i = 0; i < disparities; i++) {
		cudaMalloc((void **) &(d_imgOutNew[i]), imgOutSizeBytes);
		CUDA_CHECK;
		cudaMalloc((void **) &(d_imgOutOld[i]), imgOutSizeBytes);
		CUDA_CHECK;
		cudaMalloc((void **) &(d_imgOutFit[i]), imgOutSizeBytes);
		CUDA_CHECK;
		/*		cudaMalloc((void **) &(d_f[i]), imgOutSizeBytes);
		 CUDA_CHECK;*/
		cudaMalloc((void **) &(d_phiX[i]), imgOutSizeBytes);
		CUDA_CHECK;
		cudaMalloc((void **) &(d_phiY[i]), imgOutSizeBytes);
		CUDA_CHECK;
		cudaMalloc((void **) &(d_phiZ[i]), imgOutSizeBytes);
		CUDA_CHECK;
	}

	//Assigning the 1d pointers to the cuda mem
	float ** dptr_imgOutFit, **dptr_imgOutNew, **dptr_imgOutOld, **dptr_f,
			**dptr_phiX, **dptr_phiY, **dptr_phiZ;
	cudaMalloc((void ***) &dptr_imgOutFit, sizeof(float *) * disparities);
	CUDA_CHECK;
	cudaMemcpy(dptr_imgOutFit, d_imgOutFit, sizeof(float *) * disparities,
			cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMalloc((void ***) &dptr_imgOutNew, sizeof(float *) * disparities);
	CUDA_CHECK;
	cudaMemcpy(dptr_imgOutNew, d_imgOutNew, sizeof(float *) * disparities,
			cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMalloc((void ***) &dptr_imgOutOld, sizeof(float *) * disparities);
	CUDA_CHECK;
	cudaMemcpy(dptr_imgOutOld, d_imgOutOld, sizeof(float *) * disparities,
			cudaMemcpyHostToDevice);
	CUDA_CHECK;
	/*	cudaMalloc((void ***) &dptr_f, sizeof(float *) * disparities);
	 CUDA_CHECK;
	 cudaMemcpy(dptr_f, d_f, sizeof(float *) * disparities,
	 cudaMemcpyHostToDevice);
	 CUDA_CHECK;*/
	cudaMalloc((void ***) &dptr_phiX, sizeof(float *) * disparities);
	CUDA_CHECK;
	cudaMemcpy(dptr_phiX, d_phiX, sizeof(float *) * disparities,
			cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMalloc((void ***) &dptr_phiY, sizeof(float *) * disparities);
	CUDA_CHECK;
	cudaMemcpy(dptr_phiY, d_phiY, sizeof(float *) * disparities,
			cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMalloc((void ***) &dptr_phiZ, sizeof(float *) * disparities);
	CUDA_CHECK;
	cudaMemcpy(dptr_phiZ, d_phiZ, sizeof(float *) * disparities,
			cudaMemcpyHostToDevice);
	CUDA_CHECK;

	// clamp x to border
	texRefleftImage.addressMode[0] = cudaAddressModeClamp;
	texRefrightImage.addressMode[0] = cudaAddressModeClamp;
	// clamp y to border
	texRefleftImage.addressMode[1] = cudaAddressModeClamp;
	texRefrightImage.addressMode[1] = cudaAddressModeClamp;
	// linear interpolation
	texRefleftImage.filterMode = cudaFilterModeLinear;
	texRefrightImage.filterMode = cudaFilterModeLinear;
	// access as (x+0.5f,y+0.5f), not as ((x+0.5f)/w,(y+0.5f)/h)
	texRefleftImage.normalized = false;
	texRefrightImage.normalized = false;

	// no of bits for each texture channel
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, &texRefleftImage, d_imgInleft, &desc,
			(size_t) imgDims.x, (size_t) imgDims.y * nc,
			(size_t) imgDims.x * sizeof(d_imgInleft[0]));
	CUDA_CHECK;
	cudaBindTexture2D(NULL, &texRefrightImage, d_imgInright, &desc,
			(size_t) imgDims.x, (size_t) imgDims.y * nc,
			(size_t) imgDims.x * sizeof(d_imgInright[0]));
	CUDA_CHECK;

	// define block and grid
	dim3 block = dim3(16, 16, 1);
	dim3 grid = dim3((imgDims.x + block.x - 1) / block.x,
			(imgDims.y + block.y - 1) / block.y, 1);

	//Init all the terms with the dataterm
	/*	initialize<<<grid, block>>>(d_f, d_imgInleft, d_imgInright, nc, imgDims,
	 dptr_imgOutOld, dptr_imgOutFit, disparities, mu);*/
	initialize_tm<<<grid, block>>>(d_f, nc, imgDims, dptr_imgOutOld,
			dptr_imgOutFit, disparities, mu);
	initialize_phi<<<grid, block>>>(dptr_phiX, dptr_phiY, dptr_phiZ,
			dptr_imgOutOld, dptr_f, disparities, imgDims);

	// Allocating virtual 3D array
	cudaArray* cudaarray;
	cudaExtent volumesize;
	//set cuda array volume size
	volumesize = make_cudaExtent(imgDims.x, imgDims.y, disparities);
	//allocate device memory for cuda array
	cudaMalloc3DArray(&cudaarray, &desc, volumesize);
	CUDA_CHECK;

	// 3D memcpy parameters
	cudaMemcpy3DParms copyparms = { 0 };
	CUDA_CHECK;
	copyparms.extent = volumesize;
	copyparms.dstArray = cudaarray;
	copyparms.kind = cudaMemcpyDefault;

	copyparms.srcPtr = make_cudaPitchedPtr(d_f,
			(size_t) imgDims.x * sizeof(float), imgDims.x, (size_t) imgDims.y);
	cudaMemcpy3D(&copyparms);
	CUDA_CHECK;

	//Properties of the tex ref
	texRefDataTerm.addressMode[0] = cudaAddressModeClamp;
	texRefDataTerm.addressMode[1] = cudaAddressModeClamp;
	texRefDataTerm.addressMode[2] = cudaAddressModeClamp;

	texRefDataTerm.filterMode=cudaFilterModePoint;
	texRefDataTerm.normalized = false;
	cudaBindTextureToArray(texRefDataTerm,cudaarray,desc);

	// for each time step
	for (uint32_t tStep = 0; tStep < steps; tStep++) {

		regularizer_update<<<grid, block>>>(dptr_phiX, dptr_phiY, dptr_phiZ,
				dptr_imgOutFit, d_f, sigma, imgDims, disparities);

/*		regularizer_update_tm<<<grid, block>>>(dptr_phiX, dptr_phiY, dptr_phiZ,
		 dptr_imgOutFit, sigma, imgDims, disparities);*/

		variational_update<<<grid, block>>>(dptr_imgOutNew, dptr_imgOutOld,
				dptr_phiX, dptr_phiY, dptr_phiZ, dptr_imgOutFit, tau, imgDims,
				disparities);

		float **temp = dptr_imgOutOld;
		dptr_imgOutOld = dptr_imgOutNew;
		dptr_imgOutNew = temp;

	}

	float *d_imgOut;
	cudaMalloc((void **) &d_imgOut, imgOutSizeBytes);
	CUDA_CHECK;

	layers_summation<<<grid, block>>>(d_imgOut, dptr_imgOutOld, disparities,
			imgDims);

	cudaMemcpy(h_imgOut, d_imgOut, imgOutSizeBytes, cudaMemcpyDeviceToHost);
	CUDA_CHECK;

// free allocations
	cudaFree(d_imgInleft);
	CUDA_CHECK;
	cudaFree(d_imgInright);
	CUDA_CHECK;

	for (uint32_t disparity = 0; disparity < disparities; disparity++) {
		cudaFree(d_imgOutNew[disparity]);
		CUDA_CHECK;
		cudaFree(d_imgOutOld[disparity]);
		CUDA_CHECK;
		cudaFree(d_imgOutFit[disparity]);
		CUDA_CHECK;
		/*		cudaFree(d_f[disparity]);
		 CUDA_CHECK;*/
		cudaFree(d_phiX[disparity]);
		CUDA_CHECK;
		cudaFree(d_phiY[disparity]);
		CUDA_CHECK;
		cudaFree(d_phiZ[disparity]);
		CUDA_CHECK;
	}

	cudaFree(d_imgOut);
	CUDA_CHECK;
	cudaFree(dptr_imgOutFit);
	CUDA_CHECK;
	cudaFree(dptr_imgOutNew);
	CUDA_CHECK;
	cudaFree(dptr_imgOutOld);
	CUDA_CHECK;
	cudaFree(d_f);
	CUDA_CHECK;
	/*	cudaFree(dptr_f);
	 CUDA_CHECK;*/
	cudaFree(dptr_phiX);
	CUDA_CHECK;
	cudaFree(dptr_phiY);
	CUDA_CHECK;
	cudaFree(dptr_phiZ);
	CUDA_CHECK;

	cudaFreeArray(cudaarray);
	cudaUnbindTexture(texRefleftImage);
	cudaUnbindTexture(texRefrightImage);
	cudaUnbindTexture(texRefDataTerm);
}

__global__ void initialize(float *d_f, float *d_imgInleft, float *d_imgInright,
		uint32_t nc, dim3 imgDims, float **d_imgOutOld, float **d_imgOutFit,
		uint32_t disparities, uint32_t mu) {

	dim3 globalIdx_XY = globalIdx_Dim2();
	float threshold = 0.1f;
	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);
		float init_value;
		for (uint32_t disparity = 0; disparity < disparities; disparity++) {
			init_value = 0.f;
			// for all channels
			for (uint32_t ch_i = 0; ch_i < nc; ch_i++) {
				// channel offset
				size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;
				// get linear ids of shifted pixel in right image with clamping

				init_value +=
						fabsf(
								d_imgInleft[id + chOffset]
										- ((globalIdx_XY.x - disparity >= 0) ?
												d_imgInright[id + chOffset
														- disparity] :
												0.f));
			}
			d_imgOutOld[disparity][id] = 0.f;
			d_imgOutFit[disparity][id] = 0.f;
			d_f[(size_t) disparity * imgDims.x * imgDims.y + id] = init_value
					* mu;
		}
		//Disparity Boundary Cdn
		d_imgOutOld[0][id] = 1.f;
	}
}

__global__ void initialize_tm(float *d_f, uint32_t nc, dim3 imgDims,
		float **d_imgOutOld, float **d_imgOutFit, uint32_t disparities,
		uint32_t mu) {

	dim3 globalIdx_XY = globalIdx_Dim2();
	float threshold = 0.1f;
	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);
		float init_value;
		for (uint32_t disparity = 0; disparity < disparities; disparity++) {
			init_value = 0.f;
			// for all channels
			for (uint32_t ch_i = 0; ch_i < nc; ch_i++) {
				// channel offset
				size_t chOffset = (size_t) imgDims.x * imgDims.y * ch_i;
				init_value += fabsf(
						tex2D(texRefleftImage, globalIdx_XY.x + 0.5f,
								globalIdx_XY.y + ch_i * imgDims.y + 0.5f)
								- tex2D(texRefrightImage,
										globalIdx_XY.x + 0.5f - disparity,
										globalIdx_XY.y + ch_i * imgDims.y
												+ 0.5f));
			}
			d_imgOutOld[disparity][id] = 0.f;
			d_imgOutFit[disparity][id] = 0.f;
			d_f[(size_t) disparity * imgDims.x * imgDims.y + id] = init_value
					* mu;
		}
		//Disparity Boundary Cdn
		d_imgOutOld[0][id] = 1.f;
	}
}

__global__ void initialize_phi(float **dptr_phiX, float **dptr_phiY,
		float **dptr_phiZ, float **dptr_imgOutOld, float **dptr_f,
		uint32_t disparities, dim3 imgDims) {

	dim3 globalIdx_XY = globalIdx_Dim2();

	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

		for (uint32_t disparity = 0; disparity < disparities; disparity++) {
			/*			gradient_imgFd(&dphiX, &dphiY, &dphiZ, dptr_imgOutOld, disparity,
			 disparities, globalIdx_XY, imgDims);*/
			dptr_phiX[disparity][id] = 0.f;
			dptr_phiY[disparity][id] = 0.f;
			dptr_phiZ[disparity][id] = 0.f;
		}
	}
}

__device__ void gradient_imgFd(float *dphiX, float *dphiY, float *dphiZ,
		float **dptr_imgOutOld, uint32_t disparity, uint32_t disparities,
		dim3 globalIdx_XY, dim3 imgDims) {

	size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

// get linear ids of neighbours of offset +1 in x and y dir
	size_t neighX = linearize_neighbour_globalIdx(globalIdx_XY, imgDims,
			make_int3(1, 0, 0));
	size_t neighY = linearize_neighbour_globalIdx(globalIdx_XY, imgDims,
			make_int3(0, 1, 0));

// chalculate differentials along x and y
	*dphiX =
			(globalIdx_XY.x + 1) < imgDims.x ?
					(dptr_imgOutOld[disparity][neighX]
							- dptr_imgOutOld[disparity][id]) :
					0;
	*dphiY =
			(globalIdx_XY.y + 1) < imgDims.y ?
					(dptr_imgOutOld[disparity][neighY]
							- dptr_imgOutOld[disparity][id]) :
					0;
	*dphiZ =
			(disparity + 1) < disparities ?
					(dptr_imgOutOld[disparity + 1][id]
							- dptr_imgOutOld[disparity][id]) :
					0;

}

__global__ void regularizer_update(float **dptr_phiX, float **dptr_phiY,
		float **dptr_phiZ, float **dptr_imgOutFit, float *d_f, float sigma,
		dim3 imgDims, uint32_t disparities) {

	dim3 globalIdx_XY = globalIdx_Dim2();

	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);
		float dgradX, dgradY, dgradZ, dphiNorm, dphiZnorm, dphiX, dphiY, dphiZ;
		// for all channels
		for (uint32_t disparity = 0; disparity < disparities; disparity++) {
			gradient_imgFd(&dgradX, &dgradY, &dgradZ, dptr_imgOutFit, disparity,
					disparities, globalIdx_XY, imgDims);
			dphiX = dptr_phiX[disparity][id] + sigma * dgradX;
			dphiY = dptr_phiY[disparity][id] + sigma * dgradY;
			dphiZ = dptr_phiZ[disparity][id] + sigma * dgradZ;
			//Projection and maintaining the constraints
			dphiNorm = powf(dphiX, 2) + powf(dphiY, 2);
			dphiNorm = max(1.f, sqrtf(dphiNorm));
			dptr_phiX[disparity][id] = dphiX / dphiNorm;
			dptr_phiY[disparity][id] = dphiY / dphiNorm;

			//Forward translation
			dphiZ += d_f[(size_t) disparity * imgDims.x * imgDims.y + id];
			//Total Variation Term + projection constraint
			dphiZ = fmaxf(0.f, dphiZ);
			/*			dptr_phiZ[disparity][id] =
			 dphiZnorm < dptr_f[disparity][id] ?
			 dphiZnorm : dptr_f[disparity][id];*/
			//Backward translation
			dptr_phiZ[disparity][id] = dphiZ
					- d_f[(size_t) disparity * imgDims.x * imgDims.y + id];
		}
	}
}

__global__ void regularizer_update_tm(float **dptr_phiX, float **dptr_phiY,
		float **dptr_phiZ, float **dptr_imgOutFit, float sigma, dim3 imgDims,
		uint32_t disparities){

	dim3 globalIdx_XY = globalIdx_Dim2();

	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);
		float dgradX, dgradY, dgradZ, dphiNorm, dphiZnorm, dphiX, dphiY, dphiZ;
		// for all channels
		for (uint32_t disparity = 0; disparity < disparities; disparity++) {
			gradient_imgFd(&dgradX, &dgradY, &dgradZ, dptr_imgOutFit, disparity,
					disparities, globalIdx_XY, imgDims);
			dphiX = dptr_phiX[disparity][id] + sigma * dgradX;
			dphiY = dptr_phiY[disparity][id] + sigma * dgradY;
			dphiZ = dptr_phiZ[disparity][id] + sigma * dgradZ;
			//Projection and maintaining the constraints
			dphiNorm = powf(dphiX, 2) + powf(dphiY, 2);
			dphiNorm = max(1.f, sqrtf(dphiNorm));
			dptr_phiX[disparity][id] = dphiX / dphiNorm;
			dptr_phiY[disparity][id] = dphiY / dphiNorm;

			//Forward translation
			dphiZ += tex3D(texRefDataTerm, globalIdx_XY.x, globalIdx_XY.y, disparity);
			//Total Variation Term + projection constraint
			dphiZ = fmaxf(0.f, dphiZ);
			/*			dptr_phiZ[disparity][id] =
			 dphiZnorm < dptr_f[disparity][id] ?
			 dphiZnorm : dptr_f[disparity][id];*/
			//Backward translation
			dptr_phiZ[disparity][id] = dphiZ
					- tex3D(texRefDataTerm, globalIdx_XY.x, globalIdx_XY.y, disparity);
		}
	}
}

__global__ void variational_update(float **dptr_imgOutNew,
		float **dptr_imgOutOld, float **dptr_phiX, float **dptr_phiY,
		float **dptr_phiZ, float **dptr_imgOutFit, float tau, dim3 imgDims,
		uint32_t disparities) {

	dim3 globalIdx_XY = globalIdx_Dim2();

	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);
		float div_phi;
		float depthOutput;
		for (uint32_t disparity = 0; disparity < disparities; disparity++) {
			divergence_phi(&div_phi, dptr_phiX, dptr_phiY, dptr_phiZ, disparity,
					disparities, globalIdx_XY, imgDims);
			depthOutput = dptr_imgOutOld[disparity][id] + tau * div_phi;
			// Clipping the depthOutput  to range [0,1]
			depthOutput = fminf(1.f, fmaxf(0.f, depthOutput));
			dptr_imgOutNew[disparity][id] = depthOutput;
			if (disparity == 0) {
				dptr_imgOutNew[0][id] = 1.f;
			}
			//Updating the fitted image
			dptr_imgOutFit[disparity][id] = 2.f * dptr_imgOutNew[disparity][id]
					- dptr_imgOutOld[disparity][id];
		}
	}
}

__device__ void divergence_phi(float *div_phi, float **dptr_phiX,
		float **dptr_phiY, float **dptr_phiZ, uint32_t disparity,
		uint32_t disparities, dim3 globalIdx_XY, dim3 imgDims) {

	size_t id = linearize_globalIdx(globalIdx_XY, imgDims);

// get linear ids of neighbours of offset -1 in x and y dir
	size_t neighX = linearize_neighbour_globalIdx(globalIdx_XY, imgDims,
			make_int3(-1, 0, 0));
	size_t neighY = linearize_neighbour_globalIdx(globalIdx_XY, imgDims,
			make_int3(0, -1, 0));

// calculate divergence for the current pixel using backward difference
	float dxxU = (
			(globalIdx_XY.x + 1) < imgDims.x ? dptr_phiX[disparity][id] : 0)
			- (globalIdx_XY.x > 0 ? dptr_phiX[disparity][neighX] : 0);
	float dyyU = (
			(globalIdx_XY.y + 1) < imgDims.y ? dptr_phiY[disparity][id] : 0)
			- (globalIdx_XY.y > 0 ? dptr_phiY[disparity][neighY] : 0);
	float dzzU = ((disparity + 1 < disparities) ? dptr_phiZ[disparity][id] : 0)
			- ((disparity > 0) ? dptr_phiZ[disparity - 1][id] : 0);
	*div_phi = dxxU + dyyU + dzzU;
}

__global__ void layers_summation(float *d_imgOut, float **dptr_imgOutOld,
		uint32_t disparities, dim3 imgDims) {

	dim3 globalIdx_XY = globalIdx_Dim2();
	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		float thresholding = 0.5f;
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);
		d_imgOut[id] = 0.f;
		for (uint32_t disparity = 0; disparity < disparities; disparity++) {
			d_imgOut[id] += (
					dptr_imgOutOld[disparity][id] > thresholding ?
							dptr_imgOutOld[disparity][id] : 0.f);
		}
	}
}

