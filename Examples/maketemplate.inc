CXX = g++
LINK = clang++
NVCC = /usr/local/cuda/bin/nvcc

CUDAHOME += /usr/local/cuda
TERRAHOME ?= $(HOME)/terra/release
TERRA ?= $(HOME)/terra/terra

FLAGS += -O3 -g -std=c++11 -Isrc -Isrc/cutil/inc -I../../API/build -I$(TERRAHOME)/include -I$(CUDAHOME)/include -I$(MLIB)/include -I$(MLIB_EXTERNAL)/include  
NVCCFLAGS += -arch sm_30 $(FLAGS)
LFLAGS += -std=c++11 -L../../API/build -L$(CUDAHOME)/lib$(S4) -Wl,-rpath,$(CUDAHOME)/lib$(S4) -lcudart -lOpt -Wl,-rpath,$(shell pwd)/../../API/build -L$(TERRAHOME)/lib -lterra -ldl -pthread
MLIB ?= $(HOME)/mLib
MLIB_EXTERNAL ?= $(HOME)/mLibExternal
UNAME := $(shell uname)

.SUFFIXES:
.PHONY:	all purge clean

ifeq ($(UNAME), Darwin)
  LFLAGS += -pagezero_size 10000 -image_base 100000000
endif

all: $(EXECUTABLE)

build/%.o:	src/%.cpp
	@mkdir -p build
	$(CXX) $(FLAGS) -MP -MD $< -c -o $@
build/%.o:	src/%.cu
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

$(EXECUTABLE):	$(OBJS)
	$(CXX) $^ -o $@ $(LFLAGS)

clean:
	rm -rf build/*.o build/*.d
	rm -rf $(EXECUTABLE)

purge: clean
	rm -rf build/*

# dependency rules
include $(wildcard build/*.d)