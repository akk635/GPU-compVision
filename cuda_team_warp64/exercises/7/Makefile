CPPFLAGS = -c
NVCCFLAGS_MAIN = --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core
NVCCFLAGS_INVERT_IMAGE = --ptxas-options=-v --compiler-options -Wall

# library base directory
LIBBASE = ../../libs

# Required Libraries objects
LIBAUX = ../../libs/aux/aux.o

# includes of libraries
INCLUDES = -I../../libs/includes
# TODO fix library includes of source
INCLUDES_CU = -I../../libs/global_idx

# compilers
CPP = g++
NVCC = nvcc

################# main
all: lib.o invert_image_CPU.o invert_image.o main.cu Makefile
	$(NVCC) $(INCLUDES) -o main main.cu $(LIBAUX) invert_image_CPU.o invert_image.o $(NVCCFLAGS_MAIN)

################# dependencies
invert_image.o: invert_image.cu
	$(NVCC) $(INCLUDES_CU) -c invert_image.cu -o invert_image.o $(NVCCFLAGS_INVERT_IMAGE)

invert_image_CPU.o: invert_image_CPU.cpp
	$(CPP) $(CPPFLAGS) invert_image_CPU.cpp -o invert_image_CPU.o

lib.o:
	cd $(LIBBASE); make all;

################# clean
clean:
	rm *.o main