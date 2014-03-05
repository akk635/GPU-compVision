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
all: lib.o main.cu $(LIBAUX) Makefile
	$(NVCC) $(INCLUDES_CU) $(INCLUDES) -o main main.cu $(LIBAUX) $(NVCCFLAGS_MAIN)

lib.o:
	cd $(LIBBASE); make all;

################# clean
clean:
	rm *.o main