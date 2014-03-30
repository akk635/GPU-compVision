NVCCFLAGS_MAIN = -arch sm_20 --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core
NVCCFLAGS_DEPTH_MAPPING = -arch sm_20 --ptxas-options=-v --compiler-options -Wall

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
all: stereo_projection.o lib.o stereo_projection_gui
	$(NVCC) $(INCLUDES) -o main main.cu $(LIBAUX) stereo_projection.o $(NVCCFLAGS_MAIN)

################# dependencies
stereo_projection.o: stereo_projection.cu
	$(NVCC) $(INCLUDES) $(INCLUDES_CU) -c stereo_projection.cu -o stereo_projection.o $(NVCCFLAGS_DEPTH_MAPPING)

lib.o:
	cd $(LIBBASE); make all;

stereo_projection_gui:
	cd stereo_projection_gui;
	qmake widget.pro CONFIG+=static;
	make -f Makefile;
	cp widget ../widget;

################# clean
clean:
	rm *.o main;
	cd stereo_projection_gui; make clean;