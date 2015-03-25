.PHONY:	all clean purge

UNAME := $(shell uname)

CXX = clang++

FLAGS = -Ibuild  -I$(TERRAHOME)/include
TERRAHOME = $(HOME)/terra/release
TERRA = $(TERRAHOME)/terra
LFLAGS = -Lbuild -lOpt -Wl,-rpath,$(shell pwd)/build -L$(TERRAHOME) -lterra -Wl,-rpath,$(TERRAHOME) -pagezero_size 10000 -image_base 100000000

EXAMPLE_OBJS = example.o
TEST_OBJS = testFramework.o main.o
LINK = clang++

EXAMPLE = example/example
TEST = test/test


all:	$(EXAMPLE) $(TEST) 

TERRA = $(HOME)/terra/terra

build/Opt.h build/libOpt.so:	src/createwrapper.t src/o.t
	(cd build; $(TERRA) ../src/createwrapper.t Opt ../src/o.t)

build/%.o:	example/%.cpp build/Opt.h
	$(CXX) $(FLAGS) $< -c -o $@

build/%.o:	test/%.cpp build/Opt.h
	$(CXX) $(FLAGS) $< -std=c++11 -c -o $@


$(EXAMPLE):	$(addprefix build/, $(EXAMPLE_OBJS))
	$(LINK) $^ -o $@ $(LFLAGS)

$(TEST):	$(addprefix build/, $(TEST_OBJS))
	$(LINK) $^ -o $@ $(LFLAGS)

clean:
	rm -rf build/*.o build/*.d
	rm -rf $(EXECUTABLE)

# dependency rules
DEPENDENCIES = $(patsubst %.o,build/%.d,$(OBJS))
build/%.d:	src/%.cpp build/Opt.h
	@$(CXX) $(FLAGS) -w -M $< -o $@
	@sed -i "" "1s/^/build\/$(notdir $(@)) /" $@ 

#if we are cleaning, then don't include dependencies (which would require the header files are built)	
ifeq ($(findstring $(MAKECMDGOALS),purge clean),)
-include $(DEPENDENCIES)
endif
