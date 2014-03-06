all: main

main: main.cu aux.cu aux.h texture_mem.h texture_mem.cu Makefile
	nvcc -o main main.cu aux.cu texture_mem.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core
