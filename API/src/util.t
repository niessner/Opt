
local timeIndividualKernels = true
local debugDumpInfo = false
local shiftOptimizations = true

local S = require("std")

local util = {}
util.debugDumpInfo = debugDumpInfo
util.shiftOptimizations = shiftOptimizations

util.C = terralib.includecstring [[
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#ifdef _WIN32
	#include <io.h>
#endif
]]
local C = util.C

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

util.gpuMath = {}
util.cpuMath = {}
for k,v in pairs(mathParamCount) do
	local params = {}
	for i = 1,v do
		params[i] = float
	end
	util.gpuMath[k] = terralib.externfunction(("__nv_%sf"):format(k), params -> float)
    util.cpuMath[k] = C[k.."f"]
end
util.cpuMath.abs = C.fabsf
util.gpuMath.abs = terralib.externfunction("__nv_fmaxf", {float} -> float)
util.gpuMath.max = terralib.externfunction("__nv_max", {int,int} -> int)
util.gpuMath.min = terralib.externfunction("__nv_min", {int,int} -> int)


function Vector(T,debug)
    local struct Vector(S.Object) {
        _data : &T;
        _size : int32;
        _capacity : int32;
    }
    function Vector.metamethods.__typename() return ("Vector(%s)"):format(tostring(T)) end
    local assert = debug and S.assert or macro(function() return quote end end)
    terra Vector:init() : &Vector
        self._data,self._size,self._capacity = nil,0,0
        return self
    end
    terra Vector:init(cap : int32) : &Vector
        self:init()
        self:reserve(cap)
        return self
    end
    terra Vector:reserve(cap : int32)
        if cap > 0 and cap > self._capacity then
            var oc = self._capacity
            if self._capacity == 0 then
                self._capacity = 16
            end
            while self._capacity < cap do
                self._capacity = self._capacity * 2
            end
            self._data = [&T](S.realloc(self._data,sizeof(T)*self._capacity))
        end
    end
    terra Vector:__destruct()
        assert(self._capacity >= self._size)
        for i = 0ULL,self._size do
            S.rundestructor(self._data[i])
        end
        if self._data ~= nil then
            C.free(self._data)
            self._data = nil
        end
    end
    terra Vector:size() return self._size end
    
    terra Vector:get(i : int32)
        assert(i < self._size) 
        return &self._data[i]
    end
    Vector.metamethods.__apply = macro(function(self,idx)
        return `@self:get(idx)
    end)
    
    terra Vector:insert(idx : int32, N : int32, v : T) : {}
        assert(idx <= self._size)
        self._size = self._size + N
        self:reserve(self._size)

        if self._size > N then
            var i = self._size
            while i > idx do
                self._data[i - 1] = self._data[i - 1 - N]
                i = i - 1
            end
        end

        for i = 0ULL,N do
            self._data[idx + i] = v
        end
    end
    terra Vector:insert(idx : int32, v : T) : {}
        return self:insert(idx,1,v)
    end
    terra Vector:insert(v : T) : {}
        return self:insert(self._size,1,v)
    end
    terra Vector:insert() : &T
        self._size = self._size + 1
        self:reserve(self._size)
        return self:get(self._size - 1)
    end
    terra Vector:remove(idx : int32) : T
        assert(idx < self._size)
        var v = self._data[idx]
        self._size = self._size - 1
        for i = idx,self._size do
            self._data[i] = self._data[i + 1]
        end
        return v
    end
    terra Vector:remove() : T
        assert(self._size > 0)
        return self:remove(self._size - 1)
    end

    terra Vector:indexof(v : T) : int32
    	for i = 0LL,self._size do
            if (v == self._data[i]) then
            	return i
            end
        end
        return -1
	end

	terra Vector:contains(v : T) : bool
    	return self:indexof(v) >= 0
	end

    return Vector
end

Vector = S.memoize(Vector)

local warpSize = 32
util.warpSize = warpSize

util.symTable = function(typ, N, name)
	local r = terralib.newlist()
	for i = 1, N do
		r[i] = symbol(typ, name..tostring(i-1))
	end
	return r
end

util.ceilingDivide = terra(a : int32, b : int32)
	return (a + b - 1) / b
end

struct util.TimingInfo {
	startEvent : C.cudaEvent_t
	endEvent : C.cudaEvent_t
	duration : float
	eventName : rawstring
}
local TimingInfo = util.TimingInfo

struct util.Timer {
	timingInfo : &Vector(TimingInfo)
	currentIteration : int
}
local Timer = util.Timer

terra Timer:init() 
	self.timingInfo = [Vector(TimingInfo)].alloc():init()
	self.currentIteration = 0
end

terra Timer:cleanup()
	self.timingInfo:delete()
end 

terra Timer:nextIteration() 
	self.currentIteration = self.currentIteration + 1
end

terra Timer:reset() 
	self.timingInfo:fastclear()
	self.currentIteration = 0
end

terra Timer:evaluate()
	if ([timeIndividualKernels]) then
		var aggregateTimingInfo = [Vector(tuple(float,int))].salloc():init()
		var aggregateTimingNames = [Vector(rawstring)].salloc():init()
		for i = 0,self.timingInfo:size() do
			var eventInfo = self.timingInfo(i);
			C.cudaEventSynchronize(eventInfo.endEvent)
	    	C.cudaEventElapsedTime(&eventInfo.duration, eventInfo.startEvent, eventInfo.endEvent);
	    	var index = aggregateTimingNames:indexof(eventInfo.eventName)
	    	if index < 0 then
	    		aggregateTimingNames:insert(eventInfo.eventName)
	    		aggregateTimingInfo:insert({eventInfo.duration, 1})
	    	else
	    		aggregateTimingInfo(index)._0 = aggregateTimingInfo(index)._0 + eventInfo.duration
	    		aggregateTimingInfo(index)._1 = aggregateTimingInfo(index)._1 + 1
	    	end
	    end

		C.printf(		"--------------------------------------------------------\n")
	    C.printf(		"        Kernel        |   Count  |   Total   | Average \n")
		C.printf(		"----------------------+----------+-----------+----------\n")
	    for i = 0, aggregateTimingNames:size() do
	    	C.printf(	"----------------------+----------+-----------+----------\n")
			C.printf(" %-20s |   %4d   | %8.3fms| %7.4fms\n", aggregateTimingNames(i), aggregateTimingInfo(i)._1, aggregateTimingInfo(i)._0, aggregateTimingInfo(i)._0/aggregateTimingInfo(i)._1)
	    end
	    C.printf(		"--------------------------------------------------------\n")
	end
    
end


local terra atomicAdd(sum : &float, value : float)
	terralib.asm(terralib.types.unit,"red.global.add.f32 [$0],$1;","l,f", true, sum, value)
end
util.atomicAdd = atomicAdd

local terra __shfl_down(v : float, delta : uint, width : int)
	var ret : float;
    var c : int;
	c = ((warpSize-width) << 8) or 0x1F;
	ret = terralib.asm(float,"shfl.down.b32 $0, $1, $2, $3;","=f,f,r,r", true, v, delta, c)
	return ret;
end

local terra laneid()
    var laneid : int;
	laneid = terralib.asm(int,"mov.u32 $0, %laneid;","=r", true)
	return laneid;
end
util.laneid = laneid

-- Using the "Kepler Shuffle", see http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
local terra warpReduce(val : float) 

  var offset = warpSize >> 1
  while offset > 0 do 
    val = val + __shfl_down(val, offset, warpSize);
    offset =  offset >> 1
  end
-- Is unrolling worth it
  return val;

end
util.warpReduce = warpReduce

__syncthreads = cudalib.nvvm_barrier0

-- Straightforward implementation of: http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
-- sdata must be a block of 128 bytes of shared memory we are free to trash
local terra blockReduce(val : float, sdata : &float, threadIdx : int, threadsPerBlock : uint)
	var lane = laneid()
  	var wid = threadIdx / 32 -- TODO: check if this is right for 2D domains

  	val = warpReduce(val); -- Each warp performs partial reduction

  	if (lane==0) then
  		sdata[wid]=val; -- Write reduced value to shared memory
  	end

  	__syncthreads();   -- Wait for all partial reductions

  	--read from shared memory only if that warp existed
  	val = 0
  	if (threadIdx < threadsPerBlock / 32) then
  		val = sdata[lane]
  	end

  	if (wid==0) then
		val = warpReduce(val); --Final reduce within first warp
  	end
end
util.blockReduce = blockReduce


util.max = terra(x : double, y : double)
	return terralib.select(x > y, x, y)
end

local function noHeader(pd)
	return quote end
end

local function noFooter(pd)
	return quote end
end

util.getParameters = function(ProblemSpec, images, edgeValues, paramValues)
	local inits = terralib.newlist()
	for _, entry in ipairs(ProblemSpec.parameters) do
		if entry.kind == "image" then
			inits:insert(`entry.type { data = [&uint8](images[entry.idx])})
		elseif entry.kind == "adjacency" then
			inits:insert(`entry.obj)
		elseif entry.kind == "edgevalues" then
			inits:insert(`entry.type { data = [&entry.type.metamethods.type](edgeValues[entry.idx]) })
		elseif entry.kind == "param"then
		    inits:insert `@[&entry.type](paramValues[entry.idx])
		end
	end
	return `[ProblemSpec:ParameterType(false)]{ inits }	--don't use the blocked version
end

util.deadThread = function(problemSpec)
	local overcomputeSize = problemSpec:MaxOvercompute()
	if util.shiftOptimizations then
		return quote
			in
			threadIdx.x < overcomputeSize or 
			threadIdx.x >= blockDim.x-overcomputeSize or 
			threadIdx.y < overcomputeSize or 
			threadIdx.y >= blockDim.y-overcomputeSize 
		end
	else
		return `false
	end
end
local deadThread = util.deadThread

util.positionForValidLane = function(pd,mapMemberName,pw,ph, problemSpec)
	local overcomputeSize = problemSpec:MaxOvercompute()
	if util.shiftOptimizations then
		return quote
			@pw,@ph = (blockDim.x - 2*overcomputeSize) * blockIdx.x + threadIdx.x - overcomputeSize, (blockDim.y - 2*overcomputeSize) * blockIdx.y + threadIdx.y - overcomputeSize
		in
			@pw < pd.parameters.[mapMemberName]:W() and @ph < pd.parameters.[mapMemberName]:H() 
		end
	else
		return quote
			@pw,@ph = blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y
		in
			@pw < pd.parameters.[mapMemberName]:W() and @ph < pd.parameters.[mapMemberName]:H() 
		end
	end
end
local positionForValidLane = util.positionForValidLane

local cd = macro(function(cufunc)
    return quote
        var r = cufunc
        if r ~= 0 then  
            C.printf("cuda reported error %d",r)
            return r
        end
    end
end)

--TODO FIX THE CUDA DEVICE SYNCS
--TODO 
local makeGPULauncher = function(compiledKernel, kernelName, header, footer, problemSpec, PlanData)
	assert(problemSpec)
	assert(problemSpec:BlockSize())
	local kernelparams = compiledKernel:gettype().parameters
	local params = terralib.newlist {}
	for i = 3,#kernelparams do --skip GPU launcher and PlanData
	    params:insert(symbol(kernelparams[i]))
	end
	local overcomputeSize = problemSpec:MaxOvercompute()
	local terra GPULauncher(pd : &PlanData, [params])
		var BLOCK_SIZE : int = [problemSpec:BlockSize()]
		var effectiveBlockSize = BLOCK_SIZE
		if [shiftOptimizations] then
			effectiveBlockSize = BLOCK_SIZE - 2 * overcomputeSize
		end
		-- set effectiveBlockSize in params

		var launch = terralib.CUDAParams { (pd.parameters.X:W() - 1) / effectiveBlockSize + 1, (pd.parameters.X:H() - 1) / effectiveBlockSize + 1, 1, BLOCK_SIZE, BLOCK_SIZE, 1, 0, nil }
		--var launch = terralib.CUDAParams { (pd.parameters.X:W() - 1) / 32 + 1, (pd.parameters.X:H() - 1) / 32 + 1, 1, 32, 32, 1, 0, nil }
		C.cudaDeviceSynchronize()
		[header(pd)]
		C.cudaDeviceSynchronize()
		var stream : C.cudaStream_t = nil
		var timingInfo : TimingInfo 
		if ([timeIndividualKernels]) then
			C.cudaEventCreate(&timingInfo.startEvent)
			C.cudaEventCreate(&timingInfo.endEvent)
	        C.cudaEventRecord(timingInfo.startEvent, stream)
			timingInfo.eventName = kernelName
	    end
	    compiledKernel(&launch, @pd, params)
	    cd(C.cudaGetLastError())
		
		
		if ([timeIndividualKernels]) then
			cd(C.cudaEventRecord(timingInfo.endEvent, stream))
			pd.timer.timingInfo:insert(timingInfo)
		end

		cd(C.cudaDeviceSynchronize())
		cd(C.cudaGetLastError())
		[footer(pd)]
	end
	
	return GPULauncher
end


util.makeGPUFunctions = function(problemSpec, vars, PlanData, kernels)
	local gpu = {}
	local kernelTemplate = {}
	local wrappedKernels = {}
	
	local data = {}
	data.problemSpec = problemSpec
	data.PlanData = PlanData
	data.imageType = problemSpec:UnknownType(false) -- get non-blocked version
	
	for k, v in pairs(kernels) do
		kernelTemplate[k] = v(data)
	end
	local kernelFunctions = {}
	for k,v in pairs(kernelTemplate) do
	    kernelFunctions[k] = v.kernel
	end
	
	local compiledKernels = terralib.cudacompile(kernelFunctions, false)
	
	for k, v in pairs(compiledKernels) do
		gpu[k] = makeGPULauncher(compiledKernels[k], kernelFunctions[k].name, kernelTemplate[k].header, kernelTemplate[k].footer, problemSpec, PlanData)
	end
	
	
	return gpu
end

return util
