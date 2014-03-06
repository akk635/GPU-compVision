NVCCFLAGS_MAIN = --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core
NVCCFLAGS_DIFFUSION = --ptxas-options=-v --compiler-options -Wall

# library base directory
LIBBASE = ../../libs

# Required Libraries objects
LIBAUX = ../../libs/aux/aux.o

# includes of libraries
INCLUDES = -I../../libs/includes
# TODO fix library includes of source
INCLUDES_CU = -I../../libs/global_idx -I../../libs/co_ordinates


# compilers
NVCC = nvcc

################# main
all: lib.o non_linear_diffusion.o main.cu Makefile
	$(NVCC) $(INCLUDES) -o main main.cu $(LIBAUX) non_linear_diffusion.o $(NVCCFLAGS_MAIN)

################# dependencies
non_linear_diffusion.o: non_linear_diffusion.cu
	$(NVCC) $(INCLUDES) $(INCLUDES_CU) -c non_linear_diffusion.cu -o non_linear_diffusion.o $(NVCCFLAGS_DIFFUSION)

lib.o:
	cd $(LIBBASE); make all;

################# clean
clean:
	rm *.o main