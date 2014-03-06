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
#include "cpu_tasks.h"
#include "texture_mem.h"
#include "main.h"
using namespace std;

// uncomment to use the camera
//#define CAMERA

int main(int argc, char **argv) {
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
	if (!ret)
		cerr << "ERROR: no image specified" << endl;
	if (argc <= 1) {
		cout << "Usage: " << argv[0]
				<< " -i <image> [-repeats <repeats>] [-gray]" << endl;
		return 1;
	}
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

	float variance = 1.0f;
	getParam("variance", variance, argc, argv);
	cout << "variance :" << variance << endl;

	int kradius = ceil(3 * variance);
	int kwidth = 2 * kradius + 1;

	// Init camera / Load input image
#ifdef CAMERA

	// Init camera
	cv::VideoCapture camera(0);
	if(!camera.isOpened()) {cerr << "ERROR: Could not open camera" << endl; return 1;}
	int camW = 640;
	int camH = 480;
	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
	// read in first frame to get the dimensions
	cv::Mat mIn;
	camera >> mIn;

#else

	// Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
	cv::Mat mIn = cv::imread(image.c_str(),
			(gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
	// check
	if (mIn.data == NULL) {
		cerr << "ERROR: Could not load image " << image << endl;
		return 1;
	}

#endif

	// convert to float representation (opencv loads image values as single bytes by default)
	mIn.convertTo(mIn, CV_32F);
	// convert range of each channel to [0,1] (opencv default is [0,255])
	mIn /= 255.f;
	// get image dimensions
	int w = mIn.cols;         // width
	int h = mIn.rows;         // height
	int nc = mIn.channels();  // number of channels
	cout << "image: " << w << " x " << h << endl;
	cout << "Image Channels " << nc << endl;

	// Set the output image format
	// ###
	// ###
	// ### TODO: Change the output image format as needed
	// ###
	// ###
	cv::Mat mOut(h, w, mIn.type()); // mOut will have the same number of channels as the input image, nc layers
	//cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
	//cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
	// ### Define your own output images here as needed
	cv::Mat kOut(kwidth, kwidth, CV_32FC1); //Only with one layer

	// Allocate arrays
	// input/output image width: w
	// input/output image height: h
	// input image number of channels: nc
	// output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

	// allocate raw input image array
	float *imgIn = new float[(size_t) w * h * nc];

	// allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
	float *imgOut = new float[(size_t) w * h * mOut.channels()];
	int nc_out = mOut.channels();

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
	convert_mat_to_layered(imgIn, mIn);

	float *k = new float[(size_t) kwidth * kwidth];
	cpu_Gaussian_kernel(k, kradius, variance);
	/*	cpu_convolution(imgIn, imgOut, k, kradius, w, h, nc);*/
	/*	gpu_task gpu(imgIn, k, w, h, kradius, nc, nc_out);*/

	Timer timer;
	timer.start();
	texture_mem img_texture(imgIn, k, w, h, kwidth, kwidth, nc, nc_out);

	dim3 block = dim3(8, 8, 1);
	dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y,
			1);


	img_texture.texture_convolute_tm_gk(k, grid, block);

	img_texture.output(imgOut);
	timer.end();
	/*	gpu.convolute(grid, block);
	 gpu.output(imgOut);*/

	float *scaledK = new float[(size_t) kwidth * kwidth];
	float factor = 1 / k[kwidth * kradius + kradius];
	for (int i = 0; i < kwidth * kwidth; i++) {
		scaledK[i] = factor * k[i];
	}

	// ###
	// ###
	// ### TODO: Main computation
	// ###
	// ###

	float t = timer.get();  // elapsed time in seconds
	cout << "time: " << t * 1000 << " ms" << endl;

	// show input image
	showImage("Input", mIn, 100, 100); // show at position (x_from_left=100,y_from_above=100)

	// show output image: first convert to interleaved opencv format from the layered raw array
	convert_layered_to_mat(mOut, imgOut);
	showImage("Output", mOut, 100 + w + 40, 100);

	// ### Display your own output images here as needed
	convert_layered_to_mat(kOut, scaledK);
	showImage("kernel", kOut, 100 + w + 40 + w + 20, 100);

#ifdef CAMERA
	// end of camera loop
}
#else
	// wait for key inputs
	cv::waitKey(0);
#endif

	// save input and result
	cv::imwrite("image_input.png", mIn * 255.f); // "imwrite" assumes channel range [0,255]
	cv::imwrite("image_result.png", mOut * 255.f);

	// free allocated arrays
	delete[] imgIn;
	delete[] imgOut;
	cudaUnbindTexture(texRef);

	// close all opencv windows
	cvDestroyAllWindows();
	return 0;
}

