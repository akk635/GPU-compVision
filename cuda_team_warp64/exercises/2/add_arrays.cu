// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2013/2014, March 3 - April 4
// ###
// ###
// ### Evgeny Strekalovskiy, Maria Klodt, Jan Stuehmer, Mohamed Souiai
// ###
// ###
// ### Shiv, painkiller047@gmail.com, p053


#include <cuda_runtime.h>
#include <iostream>
using namespace std;



// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        exit(1);
    }
}


// adds elements at a particular index (i) of all input arrays
__device__ float add_elements_at(size_t i, float ** d_in_arrays, int noOfInArrays) {
    float sum = 0;
        
    // sum all arrays
    for (int arrNo = 0; arrNo < noOfInArrays; arrNo++) {
        sum += d_in_arrays[arrNo][i];
    }
    
    // return result
    return sum;
}


// add arrays - given to be summed arrays as arrays or arrays, out array, size of each array (n), no of arrays to be summed
__global__ void add_arrays(float ** d_in_arrays, float * d_out_array, int n, int noOfInArrays) {
    // get thread id
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    
    // only threads inside array range compute
    if(id < n) {
        d_out_array[id] = add_elements_at(id, d_in_arrays, noOfInArrays);
    }
}


// GPU allocs, mem copy and calls add arrays kernel
void add_arrays_caller(float * h_AoA[], int n, int noOfInArrays) {
    // define block and grid sizes - 1D assumed
    // setting a block of 512 threads
    dim3 block = dim3(512, 1, 1);
    dim3 grid = dim3((n + block.x - 1) / block.x, 1, 1);
    
    // alloc GPU memory for all arrays be added and copy those arrays
    float * arraysOnGPU[noOfInArrays];
    int bytesPerArray = n * sizeof(float);
    for(int i = 0; i < noOfInArrays; i++) {
        cudaMalloc((void **) &arraysOnGPU[i], bytesPerArray);
        CUDA_CHECK;
        cudaMemcpy(arraysOnGPU[i], h_AoA[i], bytesPerArray, cudaMemcpyHostToDevice);
        CUDA_CHECK;
    }

    // GPU memory that contains the above allocation addresses to the in arrays on GPU
    float ** d_in_arrays;
    cudaMalloc((void ***) &d_in_arrays, sizeof(float *) * noOfInArrays);
    CUDA_CHECK;
    cudaMemcpy(d_in_arrays, arraysOnGPU, sizeof(float *) * noOfInArrays, cudaMemcpyHostToDevice);
    CUDA_CHECK;

    // allocate GPU memory for output array
    float * d_out_array;
    cudaMalloc((void **) &d_out_array, sizeof(float) * bytesPerArray);
    CUDA_CHECK;
    
    // call kernel
    add_arrays<<<grid, block>>>(d_in_arrays, d_out_array, n, noOfInArrays);

    // wait for kernel call to finish
    cudaDeviceSynchronize();
    CUDA_CHECK;

    // copy summed data in output array in GPU back to host
    cudaMemcpy(h_AoA[noOfInArrays], d_out_array, bytesPerArray, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // free GPU memory - for both in and out arrays
    for(int i = 0; i < noOfInArrays; i++) {
        cudaFree(arraysOnGPU[i]);
        CUDA_CHECK;
    }
    cudaFree(d_out_array);
    CUDA_CHECK;
}


int main(int argc, char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 20;
    int NO_IN_ARRAYS = 2;
    float *a = new float[n];
    float *b = new float[n];
    float *c = new float[n];

    
    for(int i=0; i<n; i++)
    {
        a[i] = i;
        b[i] = (i%5)+1;
        c[i] = 0;
    }

    // CPU computation
    for(int i=0; i<n; i++) c[i] = a[i] + b[i];

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;
    // init c
    for(int i=0; i<n; i++) c[i] = 0;
    


    // GPU computation
    // ###
    // ### TODO: Implement the array addition on the GPU, store the result in "c"
    // ###
    // ### Notes:
    // ### 1. Remember to free all GPU arrays after the computation
    // ### 2. Always use the macro CUDA_CHECK after each CUDA call, e.g. "cudaMalloc(...); CUDA_CHECK;"
    // ###    For convenience this macro is defined directly in this file, later we will only include "aux.h"
    // following convention

    // total arrays including result array
    float * h_AoA[] = {a, b, c};

    // kernel caller
    add_arrays_caller(h_AoA, n, NO_IN_ARRAYS);

    // print result
    cout << "GPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] a;
    delete[] b;
    delete[] c;    
}