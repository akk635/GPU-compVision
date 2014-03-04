main: main.cu aux.cu aux.h Makefile gradient.cu gradient.cuh
	nvcc -o main main.cu aux.cu gradient.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

