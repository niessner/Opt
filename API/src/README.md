Documentation Maintainer: Michael Mara (mmara@stanford.edu)


Library Code:
* ad-dstar.t : Presumably an implementation of a D* symbolic differentiation system, ask Zach for details. Used by ad.t
* ad.t : Auto-differentiation code
* argparse.t : Lua argument parsing library (https://github.com/mpeterv/argparse)
* dbg.t : Small library for utility code for debugging
* im.t  : Tools for reading and writing our (simple) ad-hoc floating-point image format
* o.t : The main implementation of Opt
* solver*.t : Code for specific optimizers
* stb_perlin.* : Code for generating perlin noise.
* util.t : General utility code, including a Vector object, math functions (cpu and gpu), a Timer object, and several wrappers for cuda intrinsics.

Programs:
* createwrapper.t : I don't know. Ask Zach
* dbgADComparison.t : Compares two Opt problems to each other with either user-specified or randomly generated input parameters. Run "terra dbgADComparison.t --help" for more info.
* testbench.t : Checks the file specified in the first argument does not have syntax errors when parsed as an Opt problem.
* unittests.t : Small unit tests of code in this directory. Needs more documentation.


Example Commandlines:
terra unittests.t
terra testbench.t ../../Examples/SmoothingLaplacianFloat4/SmoothingLaplacianFloat4AD.t
terra dbgADComparison.t -t imagesmoothingImageTypes.txt