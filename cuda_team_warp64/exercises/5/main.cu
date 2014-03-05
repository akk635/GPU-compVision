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
// ### Shiv, painkiller047@gmail.com, p053
// ### Shoubhik, shoubhikdn@gmail.com, p052
// ###
// ###


#include <aux.h>
#include <iostream>
#include <math.h>
using namespace std;

// TODO FIX
#include <global_idx.cu>

// uncomment to use the camera
//#define CAMERA


// forward difference kernel
__global__ void difference_image(float *d_imgIn, float *d_img_GradX, float *d_img_GradY, int w, int h, int nc) {
    // get global idx in 3D
    dim3 globalIdx = globalIdx_Dim3();

    // only threads inside image boundary computes
    if (globalIdx.x < w && globalIdx.y < h && globalIdx.z < nc) {
        // get linear index
        size_t id = linearize_globalIdx(w, h);

        // get linear ids of neighbours of offset +1 in x and y dir
        size_t neighX = linearize_neighbour_globalIdx(w, h, 1, 0, 0);
        size_t neighY = linearize_neighbour_globalIdx(w, h, 0, 1, 0);

        // calculate differentials along x and y
        d_img_GradX[id] = (globalIdx.x + 1) < w ? (d_imgIn[neighX] - d_imgIn[id]) : 0;    
        d_img_GradY[id] = (globalIdx.y + 1) < h ? (d_imgIn[neighY] - d_imgIn[id]) : 0;            
    }
}


// gradient kernel
__global__ void gradient_image(float *d_img_GradX, float *d_img_GradY, float *d_img_gradNorm, int w, int h, int nc) {
    // get global idx in 3D
    dim3 globalIdx = globalIdx_Dim3();

    // store the square of absolute value of gradient
    float absGradSq = 0;
    
    // only threads inside image dimensions computes
    if (globalIdx.x < w && globalIdx.y < h) {
        // get linear index
        size_t id = linearize_globalIdx(w, h);

        // for every channel
        for(size_t chNo = 0; chNo < nc; chNo++) {
            // get corresponding channel neighbour
            size_t neigh = linearize_neighbour_globalIdx(w, h, 0, 0, chNo);

            // squared abs value of gradient in the current channel is added to final sum
            absGradSq += d_img_GradX[neigh] * d_img_GradX[neigh] + d_img_GradY[neigh] * d_img_GradY[neigh];
        }

        // set norm of gradient
        d_img_gradNorm[id] = sqrt(absGradSq);
    }     
}


// caller
void kernels_caller(float *h_imgIn, float *h_img_gradX, float *h_img_gradY, float *h_img_gradNorm, uint32_t w, uint32_t h, uint32_t nc) {
    // size with channels
    size_t n = w * h * nc;

    // alloc GPU memory and copy data
    float *d_imgIn;
    cudaMalloc((void **) &d_imgIn, n * sizeof(float));
    CUDA_CHECK;
    cudaMemcpy(d_imgIn, h_imgIn, n * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK;
    
    float *d_img_GradX;
    cudaMalloc((void **) &d_img_GradX, n * sizeof(float));
    CUDA_CHECK;
    cudaMemcpy(d_img_GradX, h_img_gradX, n * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK;
    
    float *d_img_GradY;
    cudaMalloc((void **) &d_img_GradY, n * sizeof(float));
    CUDA_CHECK;
    cudaMemcpy(d_img_GradY, h_img_gradY, n * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK;
    
    float *d_img_gradNorm;
    cudaMalloc((void **) &d_img_gradNorm, n / nc * sizeof(float));
    CUDA_CHECK;
    cudaMemcpy(d_img_gradNorm, h_img_gradNorm, n / nc * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK;
    
    
    // define block and grid sizes - 3D assumed
    // setting a block of 8 * 8 * 8 threads
    dim3 block = dim3(8, 8, 8);
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (nc + block.z - 1) / block.z);
    // call kernel
    difference_image<<<grid, block>>>(d_imgIn, d_img_GradX, d_img_GradY, w, h, nc);

    // define block and grid sizes - 2D assumed
    // setting a block of 16 * 16 threads
    block = dim3(16, 16, 1);
    grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);
    // call kernel
    gradient_image<<<grid, block>>>(d_img_GradX, d_img_GradY, d_img_gradNorm, w, h, nc);
    
    // wait for kernel call to finish
    cudaDeviceSynchronize();
    CUDA_CHECK;
        
    // copy back data
    cudaMemcpy(h_img_gradX, d_img_GradX, n * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;
    cudaMemcpy(h_img_gradY, d_img_GradY, n * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;
    cudaMemcpy(h_img_gradNorm, d_img_gradNorm, w * h * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;
    
    // free GPU array
    cudaFree(d_imgIn);
    CUDA_CHECK;
    cudaFree(d_img_GradX);
    CUDA_CHECK;
    cudaFree(d_img_GradY);
    CUDA_CHECK;
    cudaFree(d_img_gradNorm);
    CUDA_CHECK;
}


int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;




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
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###

    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    cv::Mat mOut_gradX(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    cv::Mat mOut_gradY(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    cv::Mat mOut_gradNorm(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed




    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn = new float[(size_t) w * h * nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut_gradX = new float[(size_t)w * h * mOut_gradX.channels()];
    float *imgOut_gradY = new float[(size_t)w * h * mOut_gradY.channels()];
    float *imgOut_gradNorm = new float[(size_t)w * h * mOut_gradNorm.channels()];



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
    
    // GPU version
    kernels_caller(imgIn, imgOut_gradX, imgOut_gradY, imgOut_gradNorm, w, h, nc); 
   
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut_gradX, imgOut_gradX);
    showImage("Difference-X", mOut_gradX, 100, 100+h+40);

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut_gradY, imgOut_gradY);
    showImage("Difference-Y", mOut_gradY, 100+w+40, 100+h+40);
    
    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut_gradNorm, imgOut_gradNorm);
    showImage("Gradient", mOut_gradNorm, 100+w+40, 100);

    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif




    // save input and result
    cv::imwrite("images/out/gradX.png", mOut_gradX * 255.f);
    cv::imwrite("images/out/gradY.png", mOut_gradY * 255.f);
    cv::imwrite("images/out/gradNorm.png", mOut_gradNorm * 255.f);


    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut_gradX;
    delete[] imgOut_gradY;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}