
// global thread ids lib
#include <global_idx.cu>
// cordinates functions
#include <co_ordinates.cu>


__global__ void convolution(float *d_imgIn, float *d_kernel, float *d_imgOut, uint32_t w, uint32_t h, uint32_t nc, uint32_t wKernel, uint32_t hKernel, uint32_t nc) {
    // get thread global id in 3D
    dim3 globalIdx = globalIdx_Dim3();

    // declare shared memory to store neighbour pixels for the block
    extern __shared__ float imgBlock[];

	// only threads inside image dimensions compute   
    if(globalIdx.x < w && globalIdx.y < h && globalIdx.z < nc) {
    	// linearize globalIdx
    	size_t globalId = linearize_globalIdx();

    	// offset map from thread block to shared memory
    	dim3 offset = dim3(wKernel / 2, hKernel / 2, 0);

    	// theadIdx shifted by offset and linearized
    	dim3 idShifted3D = dim3(offset.x + threadIdx.x, offset.y + threadIdx.y, theadIdx.z);
    	size_t idShifted = linearize_coords(idShifted3D, dim3(blockDim.x + wKernel, blockDim.y + hKernel, 0));

    	// copy pixel to shared memory
    	imgBlock[idShifted] = d_imgIn[globalId];

    	// corner clamping
    	if(((threadIdx.x == blockDim.x - 1) || (threadIdx.x == 0)) && ((threadIdx.y == blockDim.y - 1) || (threadIdx.y == 0))) {
    		// booleans if on x edge
    		int onEdge_x = (threadIdx.x == blockDim.x - 1) || (threadIdx.x == 0);
    		int onEdge_y = (threadIdx.y == blockDim.y - 1) || (threadIdx.y == 0);

    		// direction to go depending on which sides being clamped
    		int directn_x = threadIdx.x == blockDim.x - 1 ? 1 : -1;
    		int directn_y = threadIdx.y == blockDim.y - 1 ? 1 : -1;

    		for(uint32_t w_i = 1; w_i <= onEdge_x * wKernel / 2; w_i++) {
    			for(uint32_t h_i = 1; h_i <= onEdge_y * hKernel / 2; h_i++) {
    				idShifted3D = dim3(offset.x + threadIdx.x + directn_x * w_i, offset.y + threadIdx.y + directn_y * h_i, theadIdx.z);
	 			   	idShifted = linearize_coords(idShifted3D, dim3(blockDim.x + wKernel, blockDim.y + hKernel, 0));

	 				// if not on edge of image
 			   		if(pixel % (w - 1) != 0 && pixel % w != 0) pixel = linearize_coords(dim3(globalIdx.x + directn * w_i, globalIdx.y, globalIdx.z), dim3(w, h, nc));
    			}
    		}
    	}

    	// boundary check for x clamping
    	else if((threadIdx.x == blockDim.x - 1) || (threadIdx.x == 0)) {
    		// current pixel on the image to be copied
    		size_t pixel = globalId;

    		// direction to go depending on which side being clamped
    		int directn = threadIdx.x == blockDim.x - 1 ? 1 : -1;
    		
    		for(uint32_t w_i = 1; w_i <= wKernel / 2; w_i++) {
    			idShifted3D = dim3(offset.x + threadIdx.x + directn * w_i, offset.y + threadIdx.y, theadIdx.z);
 			   	idShifted = linearize_coords(idShifted3D, dim3(blockDim.x + wKernel, blockDim.y + hKernel, 0));

 			   	// if not on edge of image
 			   	if(pixel % (w - 1) != 0 && pixel % w != 0) pixel = linearize_coords(dim3(globalIdx.x + directn * w_i, globalIdx.y, globalIdx.z), dim3(w, h, nc));

 			   	// copy pixel to shared memory
			    imgBlock[idShifted] = d_imgIn[pixel];
    		}
    	}

    	// boundary check for y clamping
    	else if((threadIdx.y == blockDim.y - 1) || (threadIdx.y == 0)) {
    		// current pixel on the image to be copied
    		size_t pixel = globalId;

    		// direction to go depending on which side being clamped
    		int directn = threadIdx.y == blockDim.y - 1 ? 1 : -1;
    		
    		for(uint32_t h_i = 1; h_i <= hKernel / 2; h_i++) {
    			idShifted3D = dim3(offset.x + threadIdx.x , offset.y + threadIdx.y, theadIdx.z);
 			   	idShifted = linearize_coords(idShifted3D, dim3(blockDim.x + wKernel, blockDim.y + hKernel + directn * h_i, 0));

 			   	// if not on edge of image
 			   	if(pixel % (h - 1) != 0 && pixel % h != 0) pixel = linearize_coords(dim3(globalIdx.x, globalIdx.y + directn * h_i, globalIdx.z), dim3(w, h, nc));

 			   	// copy pixel to shared memory
			    imgBlock[idShifted] = d_imgIn[pixel];
    		}
    	}

    	

    	}

    	imgBlock[localIdx_XY] = localIdx_XY <  d_ImgIn[localIdx_XY]
    }
        float value = 0.f;
        int indConvolution = x + y * w + k * w * h;  
        for(int filterY = 0; filterY < hGaussian; filterY++)
        {
            for(int filterX = 0; filterX < wGaussian; filterX++)
            {
                int imageX = (x - wGaussian / 2 + filterX + w) % w; 
                int imageY = (y - hGaussian / 2 + filterY + h) % h; 
                int ind = imageX + imageY * w + k * w * h;                        
                int indGaussian = filterX + filterY * wGaussian;
                value += d_a[ind] * d_b[indGaussian];
                            
            }
        }
        d_c[indConvolution] = value;  
    }  
}



void convolution_caller(float *h_imgIn, float *h_kernel, float *h_imgOut, uint32_t w, uint32_t h, uint32_t nc, uint32_t wKernel, uint32_t hKernel) {
	// define dimensions - 3D
	// NOTE: CC1.x doesn't support 3D grids
	dim3 block = dim3(8, 8, nc);
	dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (nc + block.z - 1) / block.z);

	// dyanmically allocate shared memory bytes - NOTE the size > kernel
	size_t smBytes = (block.x + wKernel) * (block.y + hKernel) * sizeof(float);

	// allocate and copy memory to GPU
	float *d_imgIn,*d_kernel, *d_imgOut;

	size_t imgSize = w * h * nc * sizeof(float);
	size_t kernelSize = wKernel * hKernel * sizeof(float);
	
	cudaMalloc((void **) &d_imgIn, imgSize);
	CUDA_CHECK;
	cudaMalloc((void **) &d_kernel, kernelSize);
	CUDA_CHECK;
	cudaMalloc((void **) &d_imgOut, imgSize);
	CUDA_CHECK;

	cudaMemcpy(d_imgIn, h_imgIn, imgSize, cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);
	CUDA_CHECK;

	// call convolution kernel
	convolution<<<grid, block, smBytes>>>()
}