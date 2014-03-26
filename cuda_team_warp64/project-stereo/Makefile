NVCCFLAGS_MAIN = --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core
NVCCFLAGS_DENOISE = --ptxas-options=-v --compiler-options -Wall

# library base directory
LIBBASE = ../libs

# Required Libraries objects
LIBAUX = ../libs/aux/aux.o

# includes of libraries
INCLUDES = -I../libs/includes
# TODO fix library includes of source
INCLUDES_CU = -I../libs/global_idx -I../../libs/co_ordinates


# compilers
NVCC = nvcc -arch=sm_11

################# main
all: lib.o disparity_stereo.o main.cu Makefile
	$(NVCC) $(INCLUDES) -o main main.cu $(LIBAUX) disparity_stereo.o $(NVCCFLAGS_MAIN)

################# dependencies
disparity_stereo.o: disparity_stereo.cu 
	$(NVCC) $(INCLUDES) $(INCLUDES_CU) -c disparity_stereo.cu -o disparity_stereo.o $(NVCCFLAGS_DENOISE)
	
initialize.o : initialize.cu 
	$(NVCC) $(INCLUDES) $(INCLUDES_CU) -c initialize.cu  -o initialize.o $(NVCCFLAGS_DENOISE)
	
denoising.o : denoising.cu 
	$(NVCC) $(INCLUDES) $(INCLUDES_CU) -c denoising.cu  -o denoising.o $(NVCCFLAGS_DENOISE)
	

lib.o:
	cd $(LIBBASE); make all;

################# clean
clean:
	rm *.o main