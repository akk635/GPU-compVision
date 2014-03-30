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

#include <aux.h>
#include <iostream>
using namespace std;

// uncomment to use the camera
//#define CAMERA
#include "stereo_projection.h"
#include "disparity_stereo.h"

int
main(int argc, char **argv)
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
  // input image - left and right
  string imageLeft = "", imageRight = "";
  bool ret = getParam("i_left", imageLeft, argc, argv)
      && getParam("i_right", imageRight, argc, argv);
  if (!ret)
    cerr << "ERROR: one or more image(s) not specified" << endl;
  if (argc <= 1)
    {
      cout << "Usage: " << argv[0]
          << " -i_left <image> -i_right <image> [-repeats <repeats>] [-gray]"
          << endl;
      return 1;
    }
#endif

  // suppress all outputs by program except file writes and errors
  bool suppressOut = false;
  getParam("suppress_out", suppressOut, argc, argv);
  cout << "Suppress Out: " << suppressOut << endl;

  // number of computation repetitions to get a better run time measurement
  int repeats = 1;
  getParam("repeats", repeats, argc, argv);

  // load the input image as grayscale if "-gray" is specifed
  bool gray = false;
  getParam("gray", gray, argc, argv);

  // ### Define your own parameters here as needed
  // get MU
  float MU;
  bool retVal = getParam("mu", MU, argc, argv);
  if (!retVal)
    {
      cerr << "ERROR: no MU specified" << endl;
      cout << "Usage: " << argv[0] << " -mu <value> " << endl;
      return 1;
    }

  // get sigma
  float SIGMA;
  retVal = getParam("sigma", SIGMA, argc, argv);
  if (!retVal)
    {
      cerr << "ERROR: no SIGMA specified" << endl;
      cout << "Usage: " << argv[0] << " -sigma <value>" << endl;
      return 1;
    }

  // get TAU
  float TAU;
  retVal = getParam("tau", TAU, argc, argv);
  if (!retVal)
    {
      cerr << "ERROR: no TAU specified" << endl;
      cout << "Usage: " << argv[0] << " -tau <value>" << endl;
      return 1;
    }

  // get discretization
  uint32_t nt;
  retVal = getParam("nt", nt, argc, argv);
  if (!retVal)
    {
      cerr << "ERROR: no discretization specified" << endl;
      cout << "Usage: " << argv[0] << " -nt <value>" << endl;
      return 1;
    }

  // get steps
  uint32_t steps;
  retVal = getParam("steps", steps, argc, argv);
  if (!retVal)
    {
      cerr << "ERROR: no step specified" << endl;
      cout << "Usage: " << argv[0] << " -steps <value>" << endl;
      return 1;
    }

  // get implementation type
  int impl;
  retVal = getParam("impl", impl, argc, argv);
  if (!retVal)
    {
      cerr << "ERROR: no implementation specified" << endl;
      cout << "Usage: " << argv[0]
          << " -impl <value> {0:'Global Memory', 1:'Texture', 2:'Pitch', 3:'Shared Memory', 4:'2D Grid'}"
          << endl;
      return 1;
    }

  if (!suppressOut)
    {
      // output parameters
      cout << "repeats: " << repeats << endl;
      cout << "gray: " << gray << endl;
      cout << "MU: " << MU << endl;
      cout << "SIGMA: " << SIGMA << endl;
      cout << "TAU: " << TAU << endl;
      cout << "nt: " << nt << endl;
      cout << "Steps: " << steps << endl;
      cout << "Impl: " << impl << endl;

    }

  // Init camera / Load input image
#ifdef CAMERA

  // Init camera
  cv::VideoCapture camera(0);
  if(!camera.isOpened())
    { cerr << "ERROR: Could not open camera" << endl; return 1;}
  int camW = 640;
  int camH = 480;
  camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
  // read in first frame to get the dimensions
  cv::Mat mInLeft;
  camera >> mInLeft;

#else

  // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
  cv::Mat mInLeft = cv::imread(imageLeft.c_str(),
      (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
  // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
  cv::Mat mInRight = cv::imread(imageRight.c_str(),
      (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
  // check if both images are loaded
  if (mInLeft.data == NULL || mInRight.data == NULL)
    {
      cerr << "ERROR: Could not load one or more image(s) specified" << endl;
      return 1;
    }

#endif

  // convert to float representation (opencv loads image values as single bytes by default)
  mInLeft.convertTo(mInLeft, CV_32F);
  mInRight.convertTo(mInRight, CV_32F);
  // convert range of each channel to [0,1] (opencv default is [0,255])
  mInLeft /= 255.f;
  mInRight /= 255.f;

  // get image dimensions and both must be same
  if (mInLeft.cols != mInRight.cols || mInLeft.rows != mInRight.rows)
    {
      cerr << "ERROR: Image dimensions don't match!" << endl;
      return 1;
    }
  int w = mInLeft.cols;         // width
  int h = mInLeft.rows;         // height
  int nc = mInLeft.channels();  // number of channels
  if (!suppressOut)
    cout << "image dimensions: " << w << " x " << h << endl;

  //cv::Mat mOut(h,w,mInLeft.type());  // mOut will have the same number of channels as the input image, nc layers
  //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
  // ### Define your own output images here as needed
  cv::Mat mOutDepth(h, w, CV_32FC1);    // mOut will be a color image, 1 layers

  // Allocate arrays
  // input/output image height: h
  // input image number of channels: nc
  // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)
  // allocate raw input image array
  float *imgInLeft = new float[(size_t) w * h * nc];
  float *imgInRight = new float[(size_t) w * h * nc];
  // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
  float *imgOutDepth = new float[(size_t) w * h * mOutDepth.channels()];

  // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
  // Read a camera image frame every 30 milliseconds:
  // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
  // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
  while (cv::waitKey(30) < 0)
    {
      // Get camera image
      camera >> mInLeft;
      // convert to float representation (opencv loads image values as single bytes by default)
      mInLeft.convertTo(mInLeft,CV_32F);
      // convert range of each channel to [0,1] (opencv default is [0,255])
      mInLeft /= 255.f;
#endif

  // Init raw input image array
  // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
  // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
  // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
  convert_mat_to_layered(imgInLeft, mInLeft);
  convert_mat_to_layered(imgInRight, mInRight);

  Timer timer;
  timer.start();

  // GPU version based on implementation type selected
  switch (impl)
    {
  case 0:
    stereo_projection_PD(imgInLeft, imgInRight, imgOutDepth, dim3(w, h, 0), nc,
        dim3(w, h, nt), steps, MU, SIGMA, TAU);
    break;
  case 1:
    stereo_projection_PD_tex(imgInLeft, imgInRight, imgOutDepth, dim3(w, h, 0),
        nc, dim3(w, h, nt), steps, MU, SIGMA, TAU);
    break;
  case 2:
    stereo_projection_PD_pitch(imgInLeft, imgInRight, imgOutDepth,
        dim3(w, h, 0), nc, dim3(w, h, nt), steps, MU, SIGMA, TAU);
    break;
  case 3:
    stereo_projection_PD_sm(imgInLeft, imgInRight, imgOutDepth, dim3(w, h, 0),
        nc, dim3(w, h, nt), steps, MU, SIGMA, TAU);
    break;
  case 4:
    disparity_computation_caller(imgInLeft, imgInRight, imgOutDepth,
        dim3(w, h, 0), nc, SIGMA, TAU, steps, MU, nt); break;
  case 5:
    disparity_computation_caller_tm(imgInLeft, imgInRight, imgOutDepth,
        dim3(w, h, 0), nc, SIGMA, TAU, steps, MU, nt); break;
  default:
    break;
    }

  timer.end();
  float t = timer.get();  // elapsed time in seconds
  if (!suppressOut)
    cout << "time: " << t * 1000 << " ms" << endl;

  // for output calculation convert back to interleaved open cv and normalize
  convert_layered_to_mat(mOutDepth, imgOutDepth);
  double minVal, maxVal;
  minMaxLoc(mOutDepth, &minVal, &maxVal);
  mOutDepth /= maxVal;

  // show input image
  // show at position (x_from_left=100,y_from_above=100)
  if (!suppressOut)
    showImage("Input Left", mInLeft, 100, 100);
  // show at position (x_from_left=100,y_from_above=100)
  if (!suppressOut)
    showImage("Input Right", mInRight, 100 + w + 40, 100);

  // ### Display your own output images here as needed
  if (!suppressOut)
    showImage("Depth Mapping", mOutDepth, 100 + 2 * w + 40, 100);

#ifdef CAMERA
  // end of camera loop
}
#else
  // wait for key inputs
  cv::waitKey(0);
#endif

  // save input and result
  cv::imwrite("images/out/depth_map.png", mOutDepth * 255.f);

  // free allocated arrays
  delete[] imgInLeft;
  delete[] imgInRight;
  delete[] imgOutDepth;

  // close all opencv windows
  cvDestroyAllWindows();
  return 0;
}
