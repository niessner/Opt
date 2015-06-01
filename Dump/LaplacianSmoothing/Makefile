.PHONY:	all clean purge

UNAME := $(shell uname)

CUDA = /usr/local/cuda

CXX = $(CUDA)/bin/nvcc

FLAGS = -Isrc/cutil/inc -arch=compute_30
CPPFLAGS = 
LFLAGS = -stdlib=libstdc++ -fPIC -L$(CUDA)/lib -lcudart

ifeq ($(UNAME), Linux)
LFLAGS += -shared
else
LFLAGS += -single_module -dynamiclib
endif

OBJS = CUDAHoleFiller.o CUDAPatchSolverStereo.o CUDASolverStereo.o GlobalAppState.o RealtimeStereo.o StdOutputLogger.o TimingLog.o SolverStereoStateUtil.o \
       CUDAHoleFiller_d.o CameraUtil_d.o PatchSolverStereo_d.o SolverStereo_d.o

LINK = clang++

EXECUTABLE = libRealtimeStereo.so

all:	$(EXECUTABLE)

build/%.o:	src/%.cpp
	$(CXX) $(FLAGS) $< -c -o $@

build/%_d.o:	src/%.cu
	$(CXX) $(FLAGS) $< -c -o $@

$(EXECUTABLE):	$(addprefix build/, $(OBJS))
	$(LINK) $^ -o $@ $(LFLAGS)
	
clean:
	rm -rf build/*.o build/*.d
	rm -rf $(EXECUTABLE)

# dependency rules
DEPENDENCIES = $(patsubst %.o,build/%.d,$(OBJS))
build/%.d:	src/%.cpp
	@$(CXX) $(FLAGS) $(CPPFLAGS) -w -M $< -o $@
	@sed -i "" "1s/^/build\/$(notdir $(@)) /" $@ 

build/%.d:	src/%.cu
	@$(CXX) $(FLAGS) $(CPPFLAGS) -w -M $< -o $@
	@sed -i "" "1s/^/build\/$(notdir $(@)) /" $@ 

	
#if we are cleaning, then don't include dependencies (which would require the header files are built)	
ifeq ($(findstring $(MAKECMDGOALS),purge clean),)
-include $(DEPENDENCIES)
endif
