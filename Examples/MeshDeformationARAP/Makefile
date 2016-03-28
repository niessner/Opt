EXECUTABLE = meshdeformation
OBJS = build/LandMark.o build/LandMarkSet.o build/TerraWarpingSolver.o build/CUDAWarpingSolver.o build/mLibSource.o build/main.o build/WarpingSolver.o

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  LFLAGS += -L$(MLIB_EXTERNAL)/libsOSX -Wl,-rpath,$(MLIB_EXTERNAL)/libsOSX
endif

LFLAGS += -lOpenMeshCore -lOpenMeshTools

include ../maketemplate.inc
