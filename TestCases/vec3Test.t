local S = require("std")

local C = terralib.includecstring [[
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#ifdef _WIN32
	#include <io.h>
#endif
]]

local cuda_version = cudalib.localversion()
local libdevice = terralib.cudahome..string.format("/nvvm/libdevice/libdevice.compute_%d.10.bc",cuda_version)
terralib.linklibrary(libdevice)

local mathParamCount = {sqrt = 1,
cos  = 1,
acos = 1,
sin  = 1,
asin = 1,
tan  = 1,
atan = 1,
pow  = 2,
fmod = 2
}

local gpuMath = {}
local cpuMath = {}
for k,v in pairs(mathParamCount) do
	local params = {}
	for i = 1,v do
		params[i] = float
	end
	gpuMath[k] = terralib.externfunction(("__nv_%sf"):format(k), params -> float)
    cpuMath[k] = C[k.."f"]
end


local warpSize = 32


__syncthreads = cudalib.nvvm_barrier0

local makeGPULauncher = function(compiledKernel, width, height)
	
	local terra GPULauncher(resultPtr : &float)
		var BLOCK_SIZE : int = 16*16
		var launch = terralib.CUDAParams { (width - 1) / BLOCK_SIZE + 1, (height - 1) / BLOCK_SIZE + 1, 1, BLOCK_SIZE, BLOCK_SIZE, 1, 0, nil }
	    compiledKernel(&launch, resultPtr, width)
		C.cudaDeviceSynchronize()
	end
	
	return GPULauncher
end

local makeGPUFunctions = function(kernels, resultPtr, width, height)
	local gpu = {}
	
	local compiledKernels = terralib.cudacompile(kernels, true)
	
	for k, v in pairs(compiledKernels) do
		gpu[k] = makeGPULauncher(compiledKernels[k], width, height)
	end
	
	
	return gpu
end



local GPUBlockDims = {{"blockIdx","ctaid"},
              {"gridDim","nctaid"},
              {"threadIdx","tid"},
              {"blockDim","ntid"}}
for i,d in ipairs(GPUBlockDims) do
    local a,b = unpack(d)
    local tbl = {}
    for i,v in ipairs {"x","y","z" } do
        local fn = cudalib["nvvm_read_ptx_sreg_"..b.."_"..v] 
        tbl[v] = `fn()
    end
    _G[a] = tbl
end

local kernels = {}

terra kernels.vectorTest(resultPtr : &float, width : int)
	var x : int = blockIdx.x * blockDim.x + threadIdx.x -- global col idx
	var y : int = blockIdx.y * blockDim.y + threadIdx.y -- global row idx
	var v0 = vector([float](x), [float](y), 	[float](x*y))
	var v1 = vector([float](x), [float](x+2), 	[float](x+1))
	var v2 = vector([float](y), [float](x*y), 	[float](x))
	var result = v0 * v1 + v2
	resultPtr[y*width + x] = result[0] + result[1] + result[2]
end

local launchWidth = 800
local launchHeight = 600



local gpu = makeGPUFunctions(kernels, launchWidth, launchHeight)

local terra runTest()
	var resultPtr : &float
	C.cudaMalloc([&&opaque](&(resultPtr)), sizeof(float)*launchWidth*launchHeight)
	gpu.vectorTest(resultPtr)
	var cpuResultPtr : &float = [&float](C.malloc(sizeof(float)*launchWidth*launchHeight))
	C.cudaMemcpy(cpuResultPtr, resultPtr, sizeof(float)*launchWidth*launchHeight, C.cudaMemcpyDeviceToHost)
	var i,j = 300, 323
	C.printf("(i: %d, j: %d): %f\n", i,j, cpuResultPtr[j*launchWidth + i])
end


runTest()
