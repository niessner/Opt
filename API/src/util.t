
local timeIndividualKernels = true


local S = require("std")

local util = {}

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

local cuda_compute_version = 30
local libdevice = terralib.cudahome..string.format("/nvvm/libdevice/libdevice.compute_%d.10.bc",cuda_compute_version)

local extern = terralib.externfunction
if terralib.linkllvm then
    local obj = terralib.linkllvm(libdevice)
    function extern(...) return obj:extern(...) end
else
    terralib.linklibrary(libdevice)
end
local mathParamCount = {sqrt = 1,
cos  = 1,
acos = 1,
sin  = 1,
asin = 1,
tan  = 1,
atan = 1,
ceil = 1,
floor = 1,
log = 1,
pow  = 2,
fmod = 2,
fmax = 2,
fmin = 2
}

util.gpuMath = {}
util.cpuMath = {}
for k,v in pairs(mathParamCount) do
	local params = {}
	for i = 1,v do
		params[i] = float
	end
	util.gpuMath[k] = extern(("__nv_%sf"):format(k), params -> float)
    util.cpuMath[k] = C[k.."f"]
end
util.cpuMath["abs"] = C["fabsf"]
util.gpuMath["abs"] = terra (x : float)
	if x < 0 then
		x = -x
	end
	return x
end

local Vectors = {}
function util.isvectortype(t) return Vectors[t] end
util.Vector = terralib.memoize(function(typ,N)
    N = assert(tonumber(N),"expected a number")
    local ops = { "__sub","__add","__mul","__div" }
    local struct VecType { 
        data : typ[N]
    }
    Vectors[VecType] = true
    VecType.metamethods.type, VecType.metamethods.N = typ,N
    VecType.metamethods.__typename = function(self) return ("%s_%d"):format(tostring(self.metamethods.type),self.metamethods.N) end
    for i, op in ipairs(ops) do
        local i = symbol("i")
        local function template(ae,be)
            return quote
                var c : VecType
                for [i] = 0,N do
                    c.data[i] = operator(op,ae,be)
                end
                return c
            end
        end
        local terra doop(a : VecType, b : VecType) [template(`a.data[i],`b.data[i])]  end
        terra doop(a : typ, b : VecType) [template(`a,`b.data[i])]  end
        terra doop(a : VecType, b : typ) [template(`a.data[i],`b)]  end
       VecType.metamethods[op] = doop
    end
    terra VecType.metamethods.__unm(self : VecType)
        var c : VecType
        for i = 0,N do
            c.data[i] = -self.data[i]
        end
        return c
    end
    terra VecType:abs()
       var c : VecType
       for i = 0,N do
	  -- TODO: use opt.abs
	  if self.data[i] < 0 then
	     c.data[i] = -self.data[i]
	  else
	     c.data[i] = self.data[i]
	  end
       end
       return c
    end
    terra VecType:sum()
       var c : typ = 0
       for i = 0,N do
	  c = c + self.data[i]
       end
       return c
    end
    terra VecType:dot(b : VecType)
        var c : typ = 0
        for i = 0,N do
            c = c + self.data[i]*b.data[i]
        end
        return c
    end
	terra VecType:size()
        return N
    end
    terra VecType.methods.FromConstant(x : typ)
        var c : VecType
        for i = 0,N do
            c.data[i] = x
        end
        return c
    end
    VecType.metamethods.__apply = macro(function(self,idx) return `self.data[idx] end)
    VecType.metamethods.__cast = function(from,to,exp)
        if from:isarithmetic() and to == VecType then
            return `VecType.FromConstant(exp)
        end
        error(("unknown vector conversion %s to %s"):format(tostring(from),tostring(to)))
    end
    return VecType
end)

function Array(T,debug)
    local struct Array(S.Object) {
        _data : &T;
        _size : int32;
        _capacity : int32;
    }
    function Array.metamethods.__typename() return ("Array(%s)"):format(tostring(T)) end
    local assert = debug and S.assert or macro(function() return quote end end)
    terra Array:init() : &Array
        self._data,self._size,self._capacity = nil,0,0
        return self
    end
    terra Array:init(cap : int32) : &Array
        self:init()
        self:reserve(cap)
        return self
    end
    terra Array:reserve(cap : int32)
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
    terra Array:__destruct()
        assert(self._capacity >= self._size)
        for i = 0ULL,self._size do
            S.rundestructor(self._data[i])
        end
        if self._data ~= nil then
            C.free(self._data)
            self._data = nil
        end
    end
    terra Array:size() return self._size end
    
    terra Array:get(i : int32)
        assert(i < self._size) 
        return &self._data[i]
    end
    Array.metamethods.__apply = macro(function(self,idx)
        return `@self:get(idx)
    end)
    
    terra Array:insert(idx : int32, N : int32, v : T) : {}
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
    terra Array:insert(idx : int32, v : T) : {}
        return self:insert(idx,1,v)
    end
    terra Array:insert(v : T) : {}
        return self:insert(self._size,1,v)
    end
    terra Array:insert() : &T
        self._size = self._size + 1
        self:reserve(self._size)
        return self:get(self._size - 1)
    end
    terra Array:remove(idx : int32) : T
        assert(idx < self._size)
        var v = self._data[idx]
        self._size = self._size - 1
        for i = idx,self._size do
            self._data[i] = self._data[i + 1]
        end
        return v
    end
    terra Array:remove() : T
        assert(self._size > 0)
        return self:remove(self._size - 1)
    end

    terra Array:indexof(v : T) : int32
    	for i = 0LL,self._size do
            if (v == self._data[i]) then
            	return i
            end
        end
        return -1
	end

	terra Array:contains(v : T) : bool
    	return self:indexof(v) >= 0
	end

    return Array
end

local Array = S.memoize(Array)

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

util.TimerEvent = C.cudaEvent_t

struct util.Timer {
	timingInfo : &Array(TimingInfo)
}
local Timer = util.Timer

terra Timer:init() 
	self.timingInfo = [Array(TimingInfo)].alloc():init()
end

terra Timer:cleanup()
	self.timingInfo:delete()
end 

terra Timer:reset() 
	self.timingInfo:fastclear()
end

terra Timer:startEvent(name : rawstring,  stream : C.cudaStream_t, endEvent : &C.cudaEvent_t)
    var timingInfo : TimingInfo
    timingInfo.eventName = name
    C.cudaEventCreate(&timingInfo.startEvent)
    C.cudaEventCreate(&timingInfo.endEvent)
    C.cudaEventRecord(timingInfo.startEvent, stream)
    self.timingInfo:insert(timingInfo)
    @endEvent = timingInfo.endEvent
end
terra Timer:endEvent(stream : C.cudaStream_t, endEvent : C.cudaEvent_t)
    C.cudaEventRecord(endEvent, stream)
end

terra isprefix(pre : rawstring, str : rawstring) : bool
    if @pre == 0 then return true end
    if @str ~= @pre then return false end
    return isprefix(pre+1,str+1)
end
terra Timer:evaluate()
	if ([timeIndividualKernels]) then
		var aggregateTimingInfo = [Array(tuple(float,int))].salloc():init()
		var aggregateTimingNames = [Array(rawstring)].salloc():init()
		for i = 0,self.timingInfo:size() do
			var eventInfo = self.timingInfo(i);
			C.cudaEventSynchronize(eventInfo.endEvent)
	    	C.cudaEventElapsedTime(&eventInfo.duration, eventInfo.startEvent, eventInfo.endEvent);
	    	var index =  aggregateTimingNames:indexof(eventInfo.eventName)
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
            C.printf("TIMING ")
            for i = 0, aggregateTimingNames:size() do
		var n = aggregateTimingNames(i)
                if isprefix("PCGInit1",n) or isprefix("PCGStep1",n) or isprefix("overall",n) then
                    C.printf("%f ",aggregateTimingInfo(i)._0)
                end
            end
            C.printf("\n")
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
-- Is unrolling worth it?
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

util.initParameters = function(self, ProblemSpec, params, isInit)
    local stmts = terralib.newlist()
	for _, entry in ipairs(ProblemSpec.parameters) do
		if entry.kind == "ImageParam" then
		    if entry.idx ~= "alloc" then
                local function_name = isInit and "initFromGPUptr" or "setGPUptr"
                stmts:insert quote
                    self.[entry.name]:[function_name]([&uint8](params[entry.idx]))
                end
            end
		else
            local rhs
            if entry.kind == "GraphParam" then
                local graphinits = terralib.newlist { `@[&int](params[entry.idx]) }
                for i,e in ipairs(entry.type.metamethods.elements) do
                    graphinits:insert( `[&int](params[e.xidx]) )
                    graphinits:insert( `[&int](params[e.yidx]) )
                end
                rhs = `entry.type { graphinits }
            elseif entry.kind == "ScalarParam" then
                rhs = `@[&entry.type](params[entry.idx])
            end
            stmts:insert quote self.[entry.name] = rhs end
        end
	end
	return stmts
end

util.initPrecomputedImages = function(self, ProblemSpec)
    local stmts = terralib.newlist()
	for _, entry in ipairs(ProblemSpec.parameters) do
		if entry.kind == "image" and entry.idx == "alloc" then
            stmts:insert quote
    		    self.[entry.name]:initGPU()
    		end
    	end
    end
    return stmts
end




util.getValidUnknown = macro(function(pd,pw,ph)
	return quote
		@pw,@ph = blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y
	in
		 @pw < pd.parameters.X:W() and @ph < pd.parameters.X:H() 
	end
end)
util.getValidGraphElement = macro(function(pd,graphname,idx)
	graphname = graphname:asvalue()
	return quote
		@idx = blockDim.x * blockIdx.x + threadIdx.x
	in
		 @idx < pd.parameters.[graphname].N 
	end
end)

local positionForValidLane = util.positionForValidLane

local cd = macro(function(cufunc)
    return quote
        var r = cufunc
        if r ~= 0 then  
            C.printf("Cuda reported error %d: %s\n",r, C.cudaGetErrorString(r))
            return r
        end
    end
end)

function util.makeGPUFunctions(problemSpec, PlanData, kernels)
	
	local wrappedKernels = {}
	
	local imageType = problemSpec:UnknownType()
	local ispace = imageType.ispace
	
	local kernelFunctions = {}
	local key = "_"..tostring(os.time())
	for k,v in pairs(kernels) do
	    kernelFunctions[k..key] = { kernel = v , annotations = { {"maxntidx", 16}, {"maxntidy", #ispace.dims > 1 and 16 or 1}, {"maxntidz", #ispace.dims > 2 and 16 or 1}, {"minctasm",1} } }
	end
	local compiledKernels = terralib.cudacompile(kernelFunctions, false)
	
	local function makeGPULauncher(kernelName,compiledKernel)
        local kernelparams = compiledKernel:gettype().parameters
        local params = terralib.newlist {}
        for i = 3,#kernelparams do --skip GPU launcher and PlanData
            params:insert(symbol(kernelparams[i]))
        end
        local function createLaunchParameters(pd)
            if not kernelName:match("_Graph$") then
                local exps = terralib.newlist()
                assert(#ispace.dims <= 3, "cannot launch over images with more than 3 dims")
                local sizes = { {256,1,1}, {16,16,1}, {8,8,4} }
                for i = 1,3 do
                   local dim = #ispace.dims >= i and ispace.dims[i].size or 1
                    local bs = sizes[#ispace.dims][i]
                    exps:insert(dim)
                    exps:insert(sizes[#ispace.dims][i])
                end
                return exps
            else
                return quote
                    var N = 0
                    escape
                        for i,gf in ipairs(problemSpec.functions.cost.graphfunctions) do
                            local name = gf.graphname
                            emit quote
                                if N < pd.parameters.[name].N then
                                    N = pd.parameters.[name].N
                                end
                            end
                        end
                    end
                in 
                    N,256,1,1,1,1
                end
            end
        end
        local terra GPULauncher(pd : &PlanData, [params])
            var xdim,xblock,ydim,yblock,zdim,zblock = [ createLaunchParameters(pd) ]
            if xdim == 0 then -- early out for 0-sized kernels
                return 0
            end
            var launch = terralib.CUDAParams { (xdim - 1) / xblock + 1, (ydim - 1) / yblock + 1, (zdim - 1) / zblock + 1, 
                                                xblock, yblock, zblock, 
                                                0, nil }
            --C.cudaDeviceSynchronize()
            var stream : C.cudaStream_t = nil
            var endEvent : C.cudaEvent_t 
            if ([timeIndividualKernels]) then
                pd.timer:startEvent(kernelName,nil,&endEvent)
            end
            compiledKernel(&launch, @pd, params)
            
            --cd(C.cudaGetLastError())
            
            if ([timeIndividualKernels]) then
                pd.timer:endEvent(nil,endEvent)
            end

            --cd(C.cudaDeviceSynchronize())
            cd(C.cudaGetLastError())
        end
	    return GPULauncher
    end
	
	local gpu = {}
	for k, v in pairs(kernels) do
		gpu[k] = makeGPULauncher(k, compiledKernels[k..key])
	end
	
	return gpu
end

return util
