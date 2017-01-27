EXECUTABLE = meshdeformation
OBJS = build/CUDAWarpingSolver.o build/WarpingSolver.o build/TerraSolverWarping.o build/mLibSource.o build/main.o 

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  LFLAGS += -L$(MLIB_EXTERNAL)/libsOSX -Wl,-rpath,$(MLIB_EXTERNAL)/libsOSX
endif

LFLAGS += -lOpenMeshCore -lOpenMeshTools

include ../maketemplate.inc
