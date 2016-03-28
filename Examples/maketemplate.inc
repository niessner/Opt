CXX = clang++
LINK = clang++
NVCC = /usr/local/cuda/bin/nvcc

CUDAHOME += /usr/local/cuda
TERRAHOME ?= $(HOME)/terra/release
TERRA ?= $(HOME)/terra/terra

FLAGS += -O3 -g -std=c++11 -Isrc -Isrc/cutil/inc -I../../API/release/include -I$(TERRAHOME)/include -I$(CUDAHOME)/include -I$(MLIB)/include -I$(MLIB_EXTERNAL)/include  
NVCCFLAGS += -arch sm_30 $(FLAGS)
LFLAGS += -std=c++11 -L../../API/release/lib -L$(CUDAHOME)/lib$(S4) -Wl,-rpath,$(CUDAHOME)/lib$(S4) -lcudart -L$(TERRAHOME)/lib -lterra -ldl -pthread
MLIB ?= $(HOME)/mLib
MLIB_EXTERNAL ?= $(HOME)/mLibExternal
UNAME := $(shell uname)

.SUFFIXES:
.PHONY:	all purge clean

ifeq ($(UNAME), Darwin)
  LFLAGS += -pagezero_size 10000 -image_base 100000000
endif
ifeq ($(UNAME), Linux)
  S4 = 64
endif

all: $(EXECUTABLE)

OPTLIB = ../../API/release/lib/libOpt.a

build/%.o:	src/%.cpp
	@mkdir -p build
	$(CXX) $(FLAGS) -MP -MD $< -c -o $@
build/%.o:	src/%.cu
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

$(EXECUTABLE):	$(OBJS) $(OPTLIB)
	$(CXX) $^ -o $@ $(LFLAGS)

clean:
	rm -rf build/*.o build/*.d
	rm -rf $(EXECUTABLE)

purge: clean
	rm -rf build/*

# dependency rules
include $(wildcard build/*.d)
