NVCCFLAGS = --ptxas-options=-v --compiler-options -Wall

NVCC = nvcc

TARGET = aux

all: $(TARGET).o

$(TARGET).o:
	$(NVCC) -c $(TARGET).cu $(NVCCFLAGS)

clean:
	rm *.o