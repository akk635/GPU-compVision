main: main.cu aux.cu aux.h Makefile gradientclass.cu gradientclass.h
	nvcc -o main main.cu aux.cu gradientclass.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

