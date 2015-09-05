NVCCFLAGS = -arch sm_20 --ptxas-options=-v --compiler-options -Wall

NVCC = nvcc

TARGET = global_idx

all: $(TARGET).o lib$(TARGET).a

$(TARGET).o:
	$(NVCC) -c $(TARGET).cu $(NVCCFLAGS)

lib$(TARGET).a:$(TARGET).o
	ar r lib$(TARGET).a $(TARGET).o
	ranlib lib$(TARGET).a
clean:
	rm *.o;
	rm *.a;
