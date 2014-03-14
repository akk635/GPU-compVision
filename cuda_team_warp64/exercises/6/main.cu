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



// ###
// ###
// ### TODO: For every student of your group, please provide here:
// ###
// ### name, email, login username (for example p123)
// ###
// ###


#include "aux.h"
#include <iostream>
#include <math.h>
#define PI 3.1412f
using namespace std;

// uncomment to use the camera
//#define CAMERA

void gaussian_kernel(float *imgOut, float sigma, int wGaussian, int hGaussian) {
    float sum = 0.f;
    size_t ind;   
    for(int y = (-1)*hGaussian/2; y < hGaussian/2; y++) {
        for(int x = (-1)*wGaussian/2; x < wGaussian/2; x++) {
            ind = (x+hGaussian/2) + (size_t) (y+wGaussian/2) * wGaussian;
            *(imgOut+ind) = (1.0/(2 * PI * sigma * sigma)) * exp(-1.0 *(((size_t) x * x + (size_t) y * y)/(2 * sigma * sigma)));
            sum += *(imgOut+ind);
        }
    } 
    
    for(int i = 0; i < wGaussian*hGaussian; i++) {
        imgOut[i] /= sum;
    } 
}


// Convolution Kernel
__global__ void convolution_image(float *d_a, float *d_b, float *d_c, int width, int height, int wGaussian, int hGaussian, int nc) {
    // get thread id
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
   
    for(int k = 0; k < nc; k++)
    {
        float value = 0.f;
        int indConvolution = x + y * width + k * width * height;  
        for(int filterY = 0; filterY < hGaussian; filterY++)
        {
            for(int filterX = 0; filterX < wGaussian; filterX++)
            {
                int imageX = (x - wGaussian / 2 + filterX + width) % width; 
                int imageY = (y - hGaussian / 2 + filterY + height) % height; 
                int ind = imageX + imageY * width + k * width * height;                        
                int indGaussian = filterX + filterY * wGaussian;
                value += d_a[ind] * d_b[indGaussian];
                            
            }
        }
        d_c[indConvolution] = value;  
    }  
}



int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();
    CUDA_CHECK;




    // Reading command line parameters:
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
#endif
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed    




    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
  	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;
    
#else
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    cout << "image: " << w << " x " << h << endl;




    // Set the output image format
    float sigma = 2.f;
    int rad = ceil(3*sigma);
    int wGaussian = 2*rad + 1;
    int hGaussian = 2*rad + 1;

    cout << "kernel: " << wGaussian << " x " << hGaussian << endl;
    
    //cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    cv::Mat mOut(hGaussian,wGaussian,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    cv::Mat mOutGaussian(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    cv::Mat mOutDifference(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    // ### Define your own output images here as needed




    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn  = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)wGaussian*hGaussian*mOut.channels()];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOutGaussian = new float[(size_t)w*h*mOutGaussian.channels()];
    
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOutDifference = new float[(size_t)w*h*mOutDifference.channels()];




    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
    // Get camera image
    camera >> mIn;
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
#endif

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);
   
    

    Timer timer; timer.start();

    // CPU version 
    gaussian_kernel(imgOut, sigma, wGaussian, hGaussian);
    
    // GPU version 
    int n = w*h;
    float *h_a = imgIn;
    float *h_b = imgOut;
    float *h_c = imgOutGaussian;
        
    // define block and grid sizes - 1D assumed
    // setting a block of 16 * 16 threads
    dim3 block = dim3(16, 16, 1);
    dim3 gridConvolution = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);
    //dim3 gridGaussian = dim3((wGaussian + block.x - 1) / block.x, (hGaussian + block.y - 1) / block.y, 1);
    
    // alloc GPU memeory and copy data
    float *d_a;
    cudaMalloc((void **) &d_a, n * nc * sizeof(float));
    cudaMemcpy(d_a, h_a, n * nc * sizeof(float), cudaMemcpyHostToDevice);    
    
    float *d_b;
    cudaMalloc((void **) &d_b, wGaussian * hGaussian * sizeof(float));
    cudaMemcpy(d_b, h_b, wGaussian * hGaussian * sizeof(float), cudaMemcpyHostToDevice);
    
    float *d_c;
    cudaMalloc((void **) &d_c, n * nc * sizeof(float));
    cudaMemcpy(d_c, h_c, n * nc * sizeof(float), cudaMemcpyHostToDevice);
    
    // call kernel
    convolution_image<<<gridConvolution, block>>>(d_a, d_b, d_c, w, h, wGaussian, hGaussian, nc);
    
    // wait for kernel call to finish
    cudaDeviceSynchronize();
    
    // check for error
    cudaGetLastError();
    
    // copy back data
    cudaMemcpy(h_c, d_c, n * nc * sizeof(float), cudaMemcpyDeviceToHost); 
    
    // free GPU array
    cudaFree(d_a);
    cudaFree(d_b); 
    cudaFree(d_c);
    
    for(int i = 0; i < w * h * nc; i++)
    imgOutDifference[i] = abs(imgIn[i] - imgOutGaussian[i]);       
        
    
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array    
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);         
    
    // show output image: first convert to interleaved opencv format from the layered raw array    
    convert_layered_to_mat(mOutGaussian, imgOutGaussian);
    showImage("Convolution", mOutGaussian, 100+w+200, 100);
    
    // show output image: first convert to interleaved opencv format from the layered raw array    
    convert_layered_to_mat(mOutDifference, imgOutDifference);
    showImage("Difference", mOutDifference, 100+3*w+120, 100);

    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif




    // save input and result
    cv::imwrite("image_result.png",mOutGaussian*255.f);
    cv::imwrite("image_difference.png",mOutDifference*255.f);
    cv::imwrite("image_gaussian_kernel.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;
    delete[] imgOutGaussian;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



