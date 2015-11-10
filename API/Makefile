.PHONY:	all clean purge

# If the defaults for TERRAHOME and TERRA are not right for your installation, create a Makefile.inc
# and set them up there

-include Makefile.inc

# Setup-specific variables
CUDAHOME = /usr/local/cuda
TERRAHOME ?= $(HOME)/terra/release
TERRA ?= $(HOME)/terra/terra
MLIB = ../mLib

CXX = clang++
LINK = $(CXX)


UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
  S4 = 64
endif


FLAGS = -O3 -g -MD -Ibuild  -I$(TERRAHOME)/include -I$(CUDAHOME)/include -I$(MLIB)/include -I../Framework -std=c++11 -ferror-limit=0
LFLAGS = -Lbuild -L$(CUDAHOME)/lib$(S4) -Wl,-rpath,$(CUDAHOME)/lib$(S4) -lcudart -lOpt -Wl,-rpath,$(shell pwd)/build -L$(TERRAHOME)/lib -lterra -Wl,-rpath,$(TERRAHOME)/lib

ifeq ($(UNAME), Darwin)
  LFLAGS += -pagezero_size 10000 -image_base 100000000
endif

EXAMPLE = example/example
TEST = test/test
TESTMLIB = testMLib/testMLib

EXECUTABLES = $(EXAMPLE) $(TEST) $(TESTMLIB)
all:	$(TESTMLIB)

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
