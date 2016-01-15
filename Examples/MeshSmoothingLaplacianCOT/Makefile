.PHONY:	all clean purge

# If the defaults for TERRAHOME and TERRA are not right for your installation, create a Makefile.inc
# and set them up there

-include Makefile.inc

UNAME := $(shell uname)

TERRAHOME ?= $(HOME)/terra/release
TERRA ?= $(HOME)/terra/terra

CXX = clang++
NVCC = /usr/local/cuda/bin/nvcc


MLIB = $(HOME)/mLib
MLIB_EXTERNAL = $(HOME)/mLibExternal

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
  S4 = 64
endif

CUDAHOME = /usr/local/cuda
FLAGS = -O3 -g -std=c++11 -Isrc -Isrc/cutil/inc -I../../API/build -I$(TERRAHOME)/include -I$(CUDAHOME)/include -I$(MLIB)/include -I$(MLIB_EXTERNAL)/include -I../Framework 
CXXFLAGS = -MD $(FLAGS) 
NVCCFLAGS = -arch sm_30 $(FLAGS)


LFLAGS = -L../../API/build -L$(CUDAHOME)/lib$(S4) -Wl,-rpath,$(CUDAHOME)/lib$(S4) -lcudart -lOpt -Wl,-rpath,$(shell pwd)/../../API/build -L$(TERRAHOME)/lib -lterra -ldl 
LFLAGS += -pthread -lOpenMeshCore -lOpenMeshTools

ifeq ($(UNAME), Darwin)
  LFLAGS += -pagezero_size 10000 -image_base 100000000 -L$(MLIB_EXTERNAL)/libsOSX -Wl,-rpath,$(MLIB_EXTERNAL)/libsOSX -lOpenMeshCore -lOpenMeshTools
endif

LINK = clang++

all:	meshsmoothing


%.o:	%.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o:	%.cu
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

meshsmoothing:	src/mLibSource.o src/main.o 
	$(LINK) $^ -o $@ $(LFLAGS)

clean:
	rm -rf */*.o */*.d
	rm -rf meshsmoothing

-include $(wildcard */*.d)
