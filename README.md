Opt
---

Opt is a DSL for optimization problems in graphics.

### Prerequisites ###

Opt and all of its examples require a recent version of [Terra](https://github.com/zdevito/terra)/, and [CUDA 7.0](https://developer.nvidia.com/cuda-toolkit-70). On Windows we use Visual Studio 2013 for development, though other versions may also work. Many of the examples also depend on mLib and mLibExternal.

Our recommended directory structure for development is:




Examples/ShapeFromShading works only on Windows and requires the following:

1. [DirectX SDK (June 2010)](https://www.microsoft.com/en-us/download/details.aspx?id=6812)
2. [OpenNI 2 SDK](http://structure.io/openni)
3. [Kinect SDK 1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278)
4. [Kinect SDK 2.0](http://www.microsoft.com/en-us/download/details.aspx?id=44561)
5. mLib
6. mLibExternal

Make sure mLibExternal/libsWindows/dll64 and OpenNI2\Redist are in your path.


### Troubleshooting ###

If you get "cuInit: cuda reported error 100" when using this code on OS X, you might have gfxCardStatus installed and set with dynamic switching. Fix: switch to "Discrete Only" mode in gfxCardStatus