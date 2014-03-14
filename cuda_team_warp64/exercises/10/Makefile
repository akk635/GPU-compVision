NVCCFLAGS_MAIN = --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core
NVCCFLAGS_CONVOLUTION = --ptxas-options=-v --compiler-options -Wall

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
all: lib.o  lapl_dfsion.o main.cu Makefile
	$(NVCC) $(INCLUDES) -o main main.cu $(LIBAUX) lapl_dfsion.o $(NVCCFLAGS_MAIN)

################# dependencies
#convolution.o: convolution.cu
#	$(NVCC) $(INCLUDES) $(INCLUDES_CU) -c convolution.cu -o convolution.o $(NVCCFLAGS_CONVOLUTION)

lapl_dfsion.o : lapl_dfsion.cu
	$(NVCC) $(INCLUDES) $(INCLUDES_CU) -c lapl_dfsion.cu -o lapl_dfsion.o $(NVCCFLAGS_CONVOLUTION)
lib.o:
	cd $(LIBBASE); make all;

################# clean
clean:
	rm *.o main