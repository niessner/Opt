Opt
---

Opt is a DSL for optimization problems in graphics.

### Prerequisites ###

Opt and all of its examples require a recent version of [Terra](https://github.com/zdevito/terra)/, and [CUDA 7.0](https://developer.nvidia.com/cuda-toolkit-70). On Windows we use Visual Studio 2013 for development, though other versions may also work. Many of the examples also depend on mLib and mLibExternal.

Our recommended directory structure for development is:

/DSL/Opt (this repo)
/DSL/terra
/DSL/LLVM
/DSL/LuaJIT-2.0.4


Examples/ShapeFromShading works only on Windows and requires the following:

1. [DirectX SDK (June 2010)](https://www.microsoft.com/en-us/download/details.aspx?id=6812)
2. [OpenNI 2 SDK](http://structure.io/openni)
3. [Kinect SDK 1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278)
4. [Kinect SDK 2.0](http://www.microsoft.com/en-us/download/details.aspx?id=44561)
5. mLib
6. mLibExternal

Make sure mLibExternal/libsWindows/dll64 and OpenNI2\Redist are in your path.


### .imagedump file format ###

In order to save out results and intermediate buffers within Opt, we needed a lossless floating-point image format, saver, loader, and visualizer. Ideally we would use an extant and widely supported format. For our use case, we needed something that was either trivial to write save/load code for, or had a well-documented and easily-imported-into-terra C library.

File formats examined but ultimately discarded as impractical without significant engineering effort:

* .exr : Couldn't find lightweight C library
* .hdr : Not true floating-point (shared exponent)
* .tiff : Is probably worth exploring further; but hadn't found lightweight C read/write library

Ultimately decided to (for the moment) go the ad-hoc route, and created the .imagedump file format.

Specification:

width : int32
height : int32
channelCount : int32
datatype : int32 (0 means 32-bit floating point, all other values are reserved in case we need to support other types)
pixelData (row-major, no padding between rows, takes up width*height*channelCount*sizeof(type indicated by datatype)

Implementation in native Terra is in API/src/im.t. CPP implementation is currently ad-hoc within programs and needs to be abstracted; though one can be found in Examples/ShapeFromShading/DumpOptImage.*


### Troubleshooting ###

If you get "cuInit: cuda reported error 100" when using this code on OS X, you might have gfxCardStatus installed and set with dynamic switching. Fix: switch to "Discrete Only" mode in gfxCardStatus