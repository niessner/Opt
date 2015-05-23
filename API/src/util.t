
local timeIndividualKernels = true
local debugDumpInfo = true

local S = require("std")

local util = {}
util.debugDumpInfo = debugDumpInfo

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


function Vector(T,debug)
    local struct Vector(S.Object) {
        _data : &T;
        _size : int64;
        _capacity : int64;
    }
    function Vector.metamethods.__typename() return ("Vector(%s)"):format(tostring(T)) end
    local assert = debug and S.assert or macro(function() return quote end end)
    terra Vector:init() : &Vector
        self._data,self._size,self._capacity = nil,0,0
        return self
    end
    terra Vector:init(cap : int64) : &Vector
        self:init()
        self:reserve(cap)
        return self
    end
    terra Vector:reserve(cap : int64)
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
    
    terra Vector:get(i : int64)
        assert(i < self._size) 
        return &self._data[i]
    end
    Vector.metamethods.__apply = macro(function(self,idx)
        return `@self:get(idx)
    end)
    
    terra Vector:insert(idx : int64, N : int64, v : T) : {}
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
    terra Vector:insert(idx : int64, v : T) : {}
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
    terra Vector:remove(idx : int64) : T
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

    terra Vector:indexof(v : T) : int64
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

util.ceilingDivide = terra(a : int64, b : int64)
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



util.positionForValidLane = macro(function(pd,mapMemberName,pw,ph)
	mapMemberName = mapMemberName:asvalue()
	return quote
		@pw,@ph = blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y
	in
		 @pw < pd.parameters.[mapMemberName]:W() and @ph < pd.parameters.[mapMemberName]:H() 
	end
end)
local positionForValidLane = util.positionForValidLane

local wrapGPUKernel = function(nakedKernel, PlanData, params)
	local terra wrappedKernel(pd : PlanData, [params])
		nakedKernel(&pd, params)
	end
	
	wrappedKernel:setname(nakedKernel.name)
		
	return wrappedKernel
end


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
local makeGPULauncher = function(compiledKernel, kernelName, header, footer, problemSpec, PlanData, params)
	assert(problemSpec)
	assert(problemSpec:BlockSize())
	local terra GPULauncher(pd : &PlanData, [params])
		var BLOCK_SIZE : int = [problemSpec:BlockSize()]
		var launch = terralib.CUDAParams { (pd.parameters.X:W() - 1) / BLOCK_SIZE + 1, (pd.parameters.X:H() - 1) / BLOCK_SIZE + 1, 1, BLOCK_SIZE, BLOCK_SIZE, 1, 0, nil }
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

util.makeComputeCostGPU = function(data)
	local terra computeCost(pd : &data.PlanData,  currentValues : data.imageType)
		var cost = 0.0f
		var w : int, h : int
		if positionForValidLane(pd, "X", &w, &h) then
			var params = pd.parameters
			cost = [float](data.problemSpec.functions.cost.boundary(w, h, w, h, params))
		end

		cost = warpReduce(cost)
		if (laneid() == 0) then
			atomicAdd(pd.scratchF, cost)
		end
	end
	local function header(pd)
		return quote @pd.scratchF = 0.0f end
	end
	local function footer(pd)
		return quote return @pd.scratchF end
	end
	return { kernel = computeCost, header = header, footer = footer, params = {symbol(data.imageType)}, mapMemberName = "X" }
end

util.makeInnerProduct = function(imageType)
	local terra innerProduct(a : imageType, b : imageType)
		var sum = 0.0
		for h = 0, a:H() do
			for w = 0, a:W() do
				sum = sum + a(w, h) * b(w, h)
			end
		end
		return sum
	end
	return innerProduct
end

util.makeSetImage = function(imageType)
	local terra setImage(targetImage : imageType, sourceImage : imageType, scale : float)
		for h = 0, targetImage:H() do
			for w = 0, targetImage:W() do
				targetImage(w, h) = sourceImage(w, h) * scale
			end
		end
	end
	return setImage
end

util.makeCopyImage = function(imageType)
	local terra copyImage(targetImage : imageType, sourceImage : imageType)
		for h = 0, targetImage:H() do
			for w = 0, targetImage:W() do
				targetImage(w, h) = sourceImage(w, h)
			end
		end
	end
	return copyImage
end

util.makeClearImage = function(imageType)
	local terra clearImage(targetImage : imageType, value : float)
		for h = 0, targetImage:H() do
			for w = 0, targetImage:W() do
				targetImage(w, h) = value
			end
		end
	end
	return clearImage
end

util.makeScaleImage = function(imageType)
	local terra scaleImage(targetImage : imageType, scale : float)
		for h = 0, targetImage:H() do
			for w = 0, targetImage:W() do
				targetImage(w, h) = targetImage(w, h) * scale
			end
		end
	end
	return scaleImage
end

util.makeAddImage = function(imageType)
	local terra addImage(targetImage : imageType, addedImage : imageType, scale : float)
		for h = 0, targetImage:H() do
			for w = 0, targetImage:W() do
				targetImage(w, h) = targetImage(w, h) + addedImage(w, h) * scale
			end
		end
	end
	return addImage
end


util.makeComputeResiduals = function(data)
	local terra computeResiduals(pd : &data.PlanData, values : data.imageType, residuals : data.imageType)
		var params = pd.parameters
		params.X = values
		for h = 0, values:H() do
			for w = 0, values:W() do
				residuals(w, h) = data.problemSpec.functions.cost.boundary(w, h, w, h, params)
			end
		end
	end
	return computeResiduals
end

util.makeComputeDeltaCost = function(data)
	local terra deltaCost(pd : &data.PlanData, baseResiduals : data.imageType, currentValues : data.imageType)
		var result : double = 0.0
		var params = pd.parameters
		params.X = currentValues
		for h = 0, currentValues:H() do
			for w = 0, currentValues:W() do
				var residual = data.problemSpec.functions.cost.boundary(w, h, w, h, params)
				var delta = residual - baseResiduals(w, h)
				result = result + delta
			end
		end
		return result
	end
	return deltaCost
end

util.makeComputeSearchCost = function(data, cpu)
	local terra searchCost(pd : &data.PlanData, baseValues : data.imageType, baseResiduals : data.imageType, searchDirection : data.imageType, alpha : float, valueStore : data.imageType)
		for h = 0, baseValues:H() do
			for w = 0, baseValues:W() do
				valueStore(w, h) = baseValues(w, h) + alpha * searchDirection(w, h)
			end
		end
		return cpu.deltaCost(pd, baseResiduals, valueStore)
	end
	return searchCost
end

util.makeComputeBiSearchCost = function(data, cpu)
	local terra searchCost(pd : &data.PlanData, baseValues : data.imageType, baseResiduals : data.imageType, searchDirectionA : data.imageType, searchDirectionB : data.imageType, alpha : float, beta : float, valueStore : data.imageType)
		for h = 0, baseValues:H() do
			for w = 0, baseValues:W() do
				valueStore(w, h) = baseValues(w, h) + alpha * searchDirectionA(w, h) + beta * searchDirectionB(w, h)
			end
		end
		return cpu.deltaCost(pd, baseResiduals, valueStore)
	end
	return searchCost
end

util.makeComputeSearchCostParallel = function(data, cpu)
	local terra searchCostParallel(pd : &data.PlanData, baseValues : data.imageType, baseResiduals : data.imageType, searchDirection : data.imageType, count : int, alphas : &float, costs : &float, valueStore : data.imageType)
		for i = 0, count do
			for h = 0, baseValues:H() do
				for w = 0, baseValues:W() do
					valueStore(w, h) = baseValues(w, h) + alphas[i] * searchDirection(w, h)
				end
			end
			costs[i] = cpu.deltaCost(pd, baseResiduals, valueStore)
		end
	end
	return searchCostParallel
end

util.makeLineSearchBruteForce = function(data, cpu)
	local terra lineSearchBruteForce(pd : &data.PlanData, baseValues : data.imageType, baseResiduals : data.imageType, searchDirection : data.imageType, valueStore : data.imageType)

		-- Constants
		var lineSearchMaxIters = 1000
		var lineSearchBruteForceStart = 1e-5
		var lineSearchBruteForceMultiplier = 1.1
				
		var alpha = lineSearchBruteForceStart
		var bestAlpha = 0.0
		
		var terminalCost = 10.0

		var bestCost = 0.0
		
		for lineSearchIndex = 0, lineSearchMaxIters do
			alpha = alpha * lineSearchBruteForceMultiplier
			
			var searchCost = cpu.computeSearchCost(pd, baseValues, baseResiduals, searchDirection, alpha, valueStore)
			
			if searchCost < bestCost then
				bestAlpha = alpha
				bestCost = searchCost
			elseif searchCost > terminalCost then
				break
			end
		end
		
		return bestAlpha
	end
	return lineSearchBruteForce
end

util.makeLineSearchQuadraticMinimum = function(data, cpu)
	local terra lineSearchQuadraticMinimum(pd : &data.PlanData, baseValues : data.imageType, baseResiduals : data.imageType, searchDirection : data.imageType, valueStore : data.imageType, alphaGuess : float)

		var alphas : float[4] = array(alphaGuess * 0.5f, alphaGuess * 1.0f, alphaGuess * 1.5f, 0.0f)
		var costs : float[4]
		
		cpu.computeSearchCostParallel(pd, baseValues, baseResiduals, searchDirection, 3, alphas, costs, valueStore)
		
		var a1 = alphas[0] var a2 = alphas[1] var a3 = alphas[2]
		var c1 = costs[0] var c2 = costs[1] var c3 = costs[2]
		var a = ((c2-c1)*(a1-a3) + (c3-c1)*(a2-a1))/((a1-a3)*(a2*a2-a1*a1) + (a2-a1)*(a3*a3-a1*a1))
		var b = ((c2 - c1) - a * (a2*a2 - a1*a1)) / (a2 - a1)
		alphas[3] = -b / (2.0 * a)
		costs[3] = cpu.computeSearchCost(pd, baseValues, baseResiduals, searchDirection, alphas[3], valueStore)
		
		var bestCost = 0.0
		var bestAlpha = 0.0
		for i = 0, 4 do
			if costs[i] < bestCost then
				bestAlpha = alphas[i]
				bestCost = costs[i]
			elseif i == 3 then
				logSolver("quadratic minimization failed, bestAlpha=%f\n", bestAlpha)
				--cpu.dumpLineSearch(baseValues, baseResiduals, searchDirection, valueStore, dataImages)
			end
		end
		
		return bestAlpha, bestCost
	end
	return lineSearchQuadraticMinimum
end

util.makeBiLineSearch = function(data, cpu)
	local terra biLineSearch(pd : &data.PlanData, baseValues : data.imageType, baseResiduals : data.imageType, searchDirectionA : data.imageType, searchDirectionB : data.imageType, alphaGuess : float, betaGuess : float, valueStore : data.imageType)
		
		var bestAlpha, bestAlphaCost = cpu.lineSearchQuadraticFallback(pd, baseValues, baseResiduals, searchDirectionA, valueStore, alphaGuess)
		var bestBeta, bestBetaCost = cpu.lineSearchQuadraticFallback(pd, baseValues, baseResiduals, searchDirectionB, valueStore, betaGuess)
		
		var jointCost = cpu.computeBiSearchCost(pd, baseValues, baseResiduals, searchDirectionA, searchDirectionB, bestAlpha, bestBeta, valueStore)
		
		if jointCost < bestAlphaCost and jointCost < bestBetaCost then
			return bestAlpha, bestBeta
		end
		
		logSolver("bi-search minimization failed\n")
		
		--cpu.dumpBiLineSearch(pd, baseValues, baseResiduals, searchDirectionA, searchDirectionB, valueStore)
		
		if bestAlphaCost < bestBetaCost then
			return bestAlpha, 0.0f
		else
			return 0.0f, bestBeta
		end
	end
	return biLineSearch
end

util.makeLineSearchQuadraticFallback = function(data, cpu)
	local terra lineSearchQuadraticFallback(pd : &data.PlanData, baseValues : data.imageType, baseResiduals : data.imageType, searchDirection : data.imageType, valueStore : data.imageType, alphaGuess : float)
		var bestAlpha = 0.0
		var bestCost = 0.0
		var useBruteForce = (alphaGuess == 0.0)
		if not useBruteForce then
			
			bestAlpha, bestCost = cpu.lineSearchQuadraticMinimum(pd, baseValues, baseResiduals, searchDirection, valueStore, alphaGuess)
			
			if bestAlpha == 0.0 then
				logSolver("quadratic guess=%f failed, trying again...\n", alphaGuess)
				bestAlpha, bestCost = cpu.lineSearchQuadraticMinimum(pd, baseValues, baseResiduals, searchDirection, valueStore, alphaGuess * 4.0)
				
				if bestAlpha == 0.0 then
					logSolver("quadratic minimization exhausted\n")
					
					--if iter >= 10 then
					--else
					useBruteForce = true
					--end
					--cpu.dumpLineSearch(baseValues, baseResiduals, searchDirection, valueStore, dataImages)
				end
			end
		end

		if useBruteForce then
			logSolver("brute-force line search\n")
			bestAlpha = cpu.lineSearchBruteForce(pd, baseValues, baseResiduals, searchDirection, valueStore)
		end
		
		return bestAlpha, bestCost
	end
	return lineSearchQuadraticFallback
end

util.makeDumpLineSearch = function(data, cpu)
	local terra dumpLineSearch(pd : &data.PlanData, baseValues : data.imageType, baseResiduals : data.imageType, searchDirection : data.imageType, valueStore : data.imageType)

		-- Constants
		var lineSearchMaxIters = 1000
		var lineSearchBruteForceStart = 1e-5
		var lineSearchBruteForceMultiplier = 1.1
				
		var alpha = lineSearchBruteForceStart
		
		var file = C.fopen("C:/code/debug.txt", "wb")

		for lineSearchIndex = 0, lineSearchMaxIters do
			alpha = alpha * lineSearchBruteForceMultiplier
			
			var searchCost = cpu.computeSearchCost(pd, baseValues, baseResiduals, searchDirection, alpha, valueStore)
			
			C.fprintf(file, "%15.15f\t%15.15f\n", alpha, searchCost)
			
			if searchCost >= 10.0 then break end
		end
		
		C.fclose(file)
		logSolver("debug alpha outputted")
		C.getchar()
	end
	return dumpLineSearch
end

util.makeDumpBiLineSearch = function(data, cpu)
	local terra dumpBiLineSearch(pd : &data.PlanData, baseValues : data.imageType, baseResiduals : data.imageType, searchDirectionA : data.imageType, searchDirectionB : data.imageType, valueStore : data.imageType)

		-- Constants
		var lineSearchMaxIters = 40
		var lineSearchBruteForceStart = 0.0
		var lineSearchBruteForceIncrement = 0.1
				
		var alpha = lineSearchBruteForceStart
		
		var file = C.fopen("C:/code/debug.txt", "wb")

		var betaHeader = lineSearchBruteForceStart	
		for lineSearchIndexB = 0, lineSearchMaxIters do
			C.fprintf(file, "\t%15.15f", betaHeader)
			betaHeader = betaHeader + lineSearchBruteForceIncrement
		end
		C.fprintf(file, "\n")
		
		for lineSearchIndexA = 0, lineSearchMaxIters do
			var beta = lineSearchBruteForceStart
			
			C.fprintf(file, "%15.15f", alpha)
			
			for lineSearchIndexB = 0, lineSearchMaxIters do
				var searchCost = cpu.computeBiSearchCost(pd, baseValues, baseResiduals, searchDirectionA, searchDirectionB, alpha, beta, valueStore)
				beta = beta + lineSearchBruteForceIncrement
			
				C.fprintf(file, "\t%15.15f", searchCost)
			end
			
			alpha = alpha + lineSearchBruteForceIncrement
			C.fprintf(file, "\n")
		end
		
		C.fclose(file)
		logSolver("debug alpha outputted")
		C.getchar()
	end
	return dumpBiLineSearch
end


util.makeGPUFunctions = function(problemSpec, vars, PlanData, kernels)
	local gpu = {}
	local kernelTemplate = {}
	local wrappedKernels = {}
	
	local data = {}
	data.problemSpec = problemSpec
	data.PlanData = PlanData
	data.imageType = problemSpec:UnknownType(false) -- get non-blocked version
	
	---- accumulate all naked kernels
	if not problemSpec.shouldblock then
		kernelTemplate.computeCost = util.makeComputeCostGPU(data)
	end

	for k, v in pairs(kernels) do
		kernelTemplate[k] = v(data)
	end
	
	-- clothe naked kernels
	for k, v in pairs(kernelTemplate) do
		wrappedKernels[k] = wrapGPUKernel(v.kernel, PlanData, v.params)
	end
	
	local compiledKernels = terralib.cudacompile(wrappedKernels, false)
	
	for k, v in pairs(compiledKernels) do
		gpu[k] = makeGPULauncher(compiledKernels[k], wrappedKernels[k].name, kernelTemplate[k].header, kernelTemplate[k].footer, problemSpec, PlanData, kernelTemplate[k].params)
	end
	
	
	return gpu
end

return util
