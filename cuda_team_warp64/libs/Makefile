NVCC = nvcc

all: global_idx.o aux.o

global_idx.o:
	cd global_idx; make all;

aux.o:
	cd aux; make all;

clean:
	cd global_idx; make clean;