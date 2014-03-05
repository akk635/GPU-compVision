CPPFLAGS = -c
NVCCFLAGS = --ptxas-options=-v --compiler-options -Wall

# compilers
NVCC = nvcc

###################### main
all: square_array add_arrays

###################### square array
square_array: square_array.cu
	$(NVCC) -o square_array square_array.cu $(NVCCFLAGS)

###################### square array
add_arrays: add_arrays.cu
	$(NVCC) -o add_arrays add_arrays.cu $(NVCCFLAGS)

####################### clean
clean:
	rm square_array add_arrays