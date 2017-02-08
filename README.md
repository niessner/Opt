Opt
---

Opt is a DSL for large-scale nonlinear least squares optimization problems in graphics.

### Prerequisites ###

Opt and all of its examples require a recent version of [Terra](https://github.com/zdevito/terra)/, and [CUDA 7.5](https://developer.nvidia.com/cuda-75-downloads-archive). On Windows we use Visual Studio 2013 for development/compilation, though other versions may also work. Many of the examples also depend on mLib and mLibExternal.

Download and unzip the [terra binary release for windows, release-2016-03-25](https://github.com/zdevito/terra/releases)

Our recommended directory structure for development is:

/DSL/Opt (this repo)
/DSL/terra (renamed from the fully qualified directory name from the release download)

If you change this repo structure, you must update Opt/API/buildOpt.bat's second argument on line 2 to point to the terra repo. If you compiled terra from scratch, its internal directory structure might also be different from the release, and you will need to update Opt/API/optMake.bat's line 3 to point to the terra binary.

The examples/ work on all platforms, but require the following (which will be simplified before release):
1. mLib, in /DSL/../mLib
2. mLibExternal, in /DSL/../mLibExternal
3. Changing your path to include /DSL/../mLibExternal/libsWindows/dll64 (for OpenMesh and Ceres)
4. Terra's binary path in your $PATH environment variable

### Troubleshooting ###

If you get "cuInit: cuda reported error 100" when using this code on OS X, you might have gfxCardStatus installed and set with dynamic switching. Fix: switch to "Discrete Only" mode in gfxCardStatus

"The program can't start because cudart64_75.dll is missing from your computer. Try reinstalling the program to fix this problem." Cuda 7.5 is not on your path. Perhaps you didn't install it, or haven't closed Visual Studio since installing it. If you have done both, you'll need to add it to your path environment variable. By default, the path will be "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin"

"The program can't start because terra.dll is missing from your computer. Try reinstalling the program to fix this problem." Terra is not on your path. Perhaps you didn't install it, or haven't closed Visual Studio since installing it.