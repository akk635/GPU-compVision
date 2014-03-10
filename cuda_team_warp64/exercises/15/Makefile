NVCCFLAGS_MAIN = --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core
NVCCFLAGS_INPAINT = --ptxas-options=-v --compiler-options -Wall

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
all: lib.o inpainting_EL_RBSOR.o main.cu Makefile
	$(NVCC) $(INCLUDES) -o main main.cu $(LIBAUX) inpainting_EL_RBSOR.o $(NVCCFLAGS_MAIN)

################# dependencies
inpainting_EL_RBSOR.o: inpainting_EL_RBSOR.cu
	$(NVCC) $(INCLUDES) $(INCLUDES_CU) -c inpainting_EL_RBSOR.cu -o inpainting_EL_RBSOR.o $(NVCCFLAGS_INPAINT)

lib.o:
	cd $(LIBBASE); make all;

################# clean
clean:
	rm *.o main