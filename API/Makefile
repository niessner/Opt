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

# for release
OPENMESH_LIB=/Users/zdevito/mLibExternal/libsOSX
OPENMESH_INCLUDE=/Users/zdevito/mLibExternal/include

RELEASE_NAME := Opt-`uname | sed -e s/Darwin/OSX/ | sed -e s/CYGWIN.*/Windows/`-`uname -m`-`git rev-parse --short HEAD`
release: release/lib/libOpt.a release/lib/libOptDev.a
	make -C release/share/opt/examples/ImageWarping purge
	make -C release/share/opt/examples/MeshDeformation purge
	cp $(TERRAHOME)/lib/libterra.a release/lib
	@mkdir -p release/share/opt/examples/MeshDeformation/external/{lib,include}
	cp $(OPENMESH_LIB)/libOpenMesh* release/share/opt/examples/MeshDeformation/external/lib
	cp -r $(OPENMESH_INCLUDE)/OpenMesh release/share/opt/examples/MeshDeformation/external/include
	mv release $(RELEASE_NAME)
	zip -q -r $(RELEASE_NAME).zip $(RELEASE_NAME)
	mv $(RELEASE_NAME) release