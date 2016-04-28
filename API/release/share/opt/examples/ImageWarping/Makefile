EXECUTABLE = imagewarping
OBJS = build/mLibSource.o build/main.o

CXX = clang++
PREFIX = ../../../..
CUDAHOME = /usr/local/cuda

FLAGS += -O3 -g -std=c++11 -Isrc  -I$(PREFIX)/include -I$(CUDAHOME)/include   


UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  LFLAGS += -pagezero_size 10000 -image_base 100000000
endif
ifeq ($(UNAME), Linux)
  S4 = 64
endif

LFLAGS += -std=c++11 -L$(PREFIX)/lib -L$(CUDAHOME)/lib$(S4) -Wl,-rpath,$(CUDAHOME)/lib$(S4) -lOpt -lcudart -lterra -ldl -pthread

.SUFFIXES:
.PHONY:	all purge clean

all: $(EXECUTABLE)

build/%.o:	src/%.cpp
	@mkdir -p build
	$(CXX) $(FLAGS) -MP -MD $< -c -o $@

$(EXECUTABLE):	$(OBJS)
	$(CXX) $^ -o $@ $(LFLAGS)

clean:
	rm -rf build/*.o build/*.d
	rm -rf $(EXECUTABLE)

purge: clean
	rm -rf build/*

# dependency rules
include $(wildcard build/*.d)