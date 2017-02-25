EXECUTABLE = robust_nonrigid_alignment
OBJS = build/mLibSource.o build/main.o

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  LFLAGS += -L$(MLIB_EXTERNAL)/libsOSX -Wl,-rpath,$(MLIB_EXTERNAL)/libsOSX
endif

LFLAGS += -lOpenMeshCore -lOpenMeshTools

include ../shared/make_template.inc
