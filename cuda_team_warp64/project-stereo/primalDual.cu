/*
 * primalDual.cu
 *
 *  Created on: Mar 18, 2014
 *      Author: p054
 */
#include "primalDual.h"
#include "initialize.cu"

__global__ void regularizer_update(float **dptr_phiX, float **dptr_phiY,
		float **dptr_phiZ, float **dptr_imgOutFit, float **dptr_f, float sigma,
		dim3 imgDims, uint32_t disparities) {

	dim3 globalIdx_XY = globalIdx_Dim2();

	if (globalIdx_XY.x < imgDims.x && globalIdx_XY.y < imgDims.y) {
		// get linear index
		size_t id = linearize_globalIdx(globalIdx_XY, imgDims);
		float dgradX, dgradY, dgradZ, dphiNorm, dphiZproj;
		// for all channels
		for (uint32_t disparity = 0; disparity < disparities; disparity++) {

			gradient_imgFd(&dgradX, &dgradY, &dgradZ, dptr_imgOutFit, disparity,
					disparities, globalIdx_XY, imgDims);
			dptr_phiX[disparity][id] += sigma * dgradX;
			dptr_phiY[disparity][id] += sigma * dgradY;
			dptr_phiZ[disparity][id] += sigma * dgradZ;
			//Projection and maintaining the constraints
			dphiNorm = pow(dptr_phiX[disparity][id], 2)
					+ pow(dptr_phiY[disparity][id], 2);
			dphiNorm = max(1.f, sqrt(dphiNorm));
			dptr_phiX[disparity][id] /= dphiNorm;
			dptr_phiY[disparity][id] /= dphiNorm;
			//Forward translation
			dptr_phiZ[disparity][id] += dptr_f[disparity][id];
			//Total Variation Term + projection constraint
			dphiZproj = max(0.f, dptr_phiZ[disparity][id]);
			dptr_phiZ[disparity][id] =
					dphiZproj < dptr_f[disparity][id] ?
							dphiZproj : dptr_f[disparity][id];
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
			depthOutput = dptr_imgOutOld[disparity][id] - tau * div_phi;
			//Clipping the depthOutput  to range [0,1]
			depthOutput = depthOutput > 1 ? 1 : depthOutput;
			depthOutput = depthOutput < 0 ? 0 : depthOutput;
			dptr_imgOutNew[disparity][id] = depthOutput;
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
	*div_zeta = dxxU + dyyU + dzzU;
}

