.PHONY:	all clean purge

UNAME := $(shell uname)

CXX = clang++

MLIB = ../mLib

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
  S4 = 64
endif

CUDAHOME = /usr/local/cuda
FLAGS = -O3 -g -MD -Ibuild  -I$(TERRAHOME)/include -I$(CUDAHOME)/include -I$(MLIB)/include -I../Framework -std=c++11 -ferror-limit=0
TERRAHOME = $(HOME)/terra/release
TERRA = $(TERRAHOME)/terra
LFLAGS = -Lbuild -L$(CUDAHOME)/lib$(S4) -Wl,-rpath,$(CUDAHOME)/lib$(S4) -lcudart -lOpt -Wl,-rpath,$(shell pwd)/build -L$(TERRAHOME) -lterra -Wl,-rpath,$(TERRAHOME)

ifeq ($(UNAME), Darwin)
  LFLAGS += -pagezero_size 10000 -image_base 100000000
endif

LINK = clang++

EXAMPLE = example/example
TEST = test/test
TESTMLIB = testMLib/testMLib

EXECUTABLES = $(EXAMPLE) $(TEST) $(TESTMLIB)
all:	$(TESTMLIB)

TERRA = $(HOME)/terra/terra

build/Opt.h build/libOpt.so:	src/createwrapper.t src/o.t
	(cd build; $(TERRA) ../src/createwrapper.t Opt ../src/o.t)

%.o:	%.cpp build/Opt.h
	$(CXX) $(FLAGS) $< -c -o $@

$(EXAMPLE):	example/example.o
	$(LINK) $^ -o $@ $(LFLAGS)

$(TEST):	test/testFramework.o test/main.o
	$(LINK) $^ -o $@ $(LFLAGS)
	
$(TESTMLIB):	testMLib/main.o testMLib/mLibSource.o testMLib/testFramework.o testMLib/exampleImageSmoothing.o testMLib/exampleMeshSmoothing.o
	$(LINK) $^ -o $@ $(LFLAGS)
	
clean:
	rm -rf */*.o */*.d build/Opt.h build/libOpt.so
	rm -rf $(EXECUTABLES)

-include $(wildcard */*.d)
