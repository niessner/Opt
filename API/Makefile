.PHONY:	all clean purge

TERRAHOME ?= $(HOME)/terra/release
TERRA ?= $(HOME)/terra/terra

all: release/lib/libOpt.a release/lib/libOptDev.a

release/lib/libOpt.a:	$(wildcard src/*.t)
	@mkdir -p release/lib
	$(TERRA) src/createwrapper.t Opt src o release/include/Opt.h build/wrapper.o true
	rm -rf $@
	ar cq $@ build/wrapper.o

release/lib/libOptDev.a:	$(wildcard src/*.t)
	@mkdir -p release/lib
	$(TERRA) src/createwrapper.t Opt src o release/include/Opt.h build/wrapperdev.o false
	rm -rf $@
	ar cq $@ build/wrapperdev.o
	
clean:
	rm -rf build/wrapper.o
	rm -rf release/lib/libOpt.a
