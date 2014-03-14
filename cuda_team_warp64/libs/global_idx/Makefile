NVCCFLAGS = --ptxas-options=-v --compiler-options -Wall

NVCC = nvcc

TARGET = global_idx

all: $(TARGET).o

$(TARGET).o:
	$(NVCC) -c $(TARGET).cu $(NVCCFLAGS)

clean:
	rm *.o