NVCC = nvcc

all: co_ordinates.o global_idx.o aux.o

co_ordinates.o:
	cd co_ordinates; make all;

global_idx.o:
	cd global_idx; make all;

aux.o:
	cd aux; make all;

clean:
	cd co_ordinates; make clean;
	cd global_idx; make clean;
	cd aux; make clean;