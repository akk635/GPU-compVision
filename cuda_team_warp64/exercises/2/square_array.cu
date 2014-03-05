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


// squares element at a particular index (i) an array
__device__ float square_element(float *d_a, size_t i) {
    return d_a[i] * d_a[i];
}


// square kernel
__global__ void square_array(float *d_a, size_t n) {
    // get thread global id
    size_t id = threadIdx.x + (size_t) blockDim.x * blockIdx.x;
    
    // only threads inside array range compute
    if(id < n) d_a[id] = square_element(d_a, id);
}


// GPU allocs, mem copy and calls square kernel
void square_array_caller(float *h_a, size_t n) {
    // define block and grid sizes - 1D assumed
    // setting a block of 512 threads
    dim3 block = dim3(512, 1, 1);
    dim3 grid = dim3((n + block.x - 1) / block.x, 1, 1);
    
    // alloc GPU memory and copy data
    float *d_a;
    int bytesSize = n * sizeof(float);
    cudaMalloc((void **) &d_a, bytesSize);
    CUDA_CHECK;
    cudaMemcpy(d_a, h_a, bytesSize, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    
    // call kernel
    square_array<<<grid, block>>>(d_a, n);
    
    // wait for kernel call to finish
    cudaDeviceSynchronize();
    CUDA_CHECK;
    
    // copy back data
    cudaMemcpy(h_a, d_a, bytesSize, cudaMemcpyDeviceToHost);
    CUDA_CHECK;
    
    // free GPU array
    cudaFree(d_a);
    CUDA_CHECK;
}
    

int main(int argc,char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 10;
    float *a = new float[n];
    
    for(int i=0; i<n; i++) a[i] = i;

    // CPU computation
    for(int i=0; i<n; i++)
    {
        float val = a[i];
        val = val*val;
        a[i] = val;
    }

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    cout << endl;
    


    // GPU computation
    // reinit data
    for(int i=0; i<n; i++) a[i] = i;

    
    // ###
    // ### TODO: Implement the "square array" operation on the GPU and store the result in "a"
    // ###
    // ### Notes:
    // ### 1. Remember to free all GPU arrays after the computation
    // ### 2. Always use the macro CUDA_CHECK after each CUDA call, e.g. "cudaMalloc(...); CUDA_CHECK;"
    // ###    For convenience this macro is defined directly in this file, later we will only include "aux.h"
    
    // kernel caller
    square_array_caller(a, n);

    // print result
    cout << "GPU:" << endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    cout << endl;
    
    // free CPU arrays
    delete[] a;
}