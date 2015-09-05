/*
 * initialize.cu
 *
 *  Created on: Mar 17, 2014
 *      Author: p054
 */
#include "initialize.h"
#include <global_idx.h>
#include <global_idx.cu>

__global__ void initialize(float **d_f, float *d_imgInleft, float *d_imgInright,
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
			d_f[disparity][id] = init_value * mu;
		}
		//Disparity Boundary Cdn
		d_imgOutOld[0][id] = 1.f;
	}
}

__global__ void initialize_tm(float **d_f, float *d_imgInright, uint32_t nc,
		dim3 imgDims, float **d_imgOutOld, float **d_imgOutFit,
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
				init_value += fabsf(
						tex2D(texRefleftImage, globalIdx_XY.x,
								globalIdx_XY.y + ch_i * imgDims.y));
			}
			d_imgOutOld[disparity][id] = 0.f;
			d_imgOutFit[disparity][id] = 0.f;
			d_f[disparity][id] = init_value * mu;
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
		float **dptr_phiZ, float **dptr_imgOutFit, float **dptr_f, float sigma,
		dim3 imgDims, uint32_t disparities) {

	dim3 globalIdx_XY = globalIdx_Dim2();

	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);
		float dgradX, dgradY, dgradZ, dphiNorm, dphiZnorm;
		// for all channels
		for (uint32_t disparity = 0; disparity < disparities; disparity++) {
			gradient_imgFd(&dgradX, &dgradY, &dgradZ, dptr_imgOutFit, disparity,
					disparities, globalIdx_XY, imgDims);
			dptr_phiX[disparity][id] += sigma * dgradX;
			dptr_phiY[disparity][id] += sigma * dgradY;
			dptr_phiZ[disparity][id] += sigma * dgradZ;
			//Projection and maintaining the constraints
			dphiNorm = powf(dptr_phiX[disparity][id], 2)
					+ powf(dptr_phiY[disparity][id], 2);
			dphiNorm = max(1.f, sqrtf(dphiNorm));
			dptr_phiX[disparity][id] /= dphiNorm;
			dptr_phiY[disparity][id] /= dphiNorm;

			//Forward translation
			dptr_phiZ[disparity][id] += dptr_f[disparity][id];
			//Total Variation Term + projection constraint
			dphiZnorm = max(0.f, dptr_phiZ[disparity][id]);
			dptr_phiZ[disparity][id] =
					dphiZnorm < dptr_f[disparity][id] ?
							dphiZnorm : dptr_f[disparity][id];
			//Backward translation
			dptr_phiZ[disparity][id] -= dptr_f[disparity][id];
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

