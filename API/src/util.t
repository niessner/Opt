
local timeIndividualKernels = true

local S = require("std")

local util = {}

util.C = terralib.includecstring [[
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
]]

local C = util.C
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


local warpSize = 32;


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

--[[ HOLD IT! http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/ 
claims that on Kepler, warpReduce followed by atomicAdd is either faster or the same as doing an optimized block reduce followed by atomicAdd.
This is ostensibly good news, our code is kept short and clear while still being faster

HOWEVER; floating point arithmetic is not associative; if we deem that important, we need to get rid of all atomics; so we should implement this anyways

local vload = macro(function(x) return `terralib.attrload(x, {isvolatile = true}) end)
local vstore = macro(function(x,v) return `terralib.attrstore(x,v, {isvolatile = true}) end)

local terra blockReduce(sdata : &float, threadIdx : int, threadsPerBlock : uint)
	--TODO: either we meta-program this ourself, or get #pragma unroll to work in Terra
	var stride = threadsPerBlock / 2
	while stride > 32 do
		if threadIdx < stride then
		    var t = vload(sdata + threadIdx) + vload(sdata + threadIdx + stride)
		    vstore(sdata + threadIdx,t)
		end
		__syncthreads()   
	    stride = stride / 2
	end
	warpReduce(sdata, threadIdx, threadsPerBlock)
end

local terra scanPart1(threadIdx : uint, blockIdx : uint, threadsPerBlock : uint, d_output : &float)
	__syncthreads()
	blockReduce(bucket, threadIdx, threadsPerBlock) 
	if  threadIdx == 0 then d_output[blockIdx] = bucket[0] end
end

local terra scanPart2(threadIdx : uint, threadsPerBlock : uint, blocksPerGrid : uint, d_tmp : &float)
	if threadIdx < blocksPerGrid then bucket[threadIdx] = d_tmp[threadIdx]
	else bucket[threadIdx] = 0.0f end
	
	__syncthreads();
	blockReduce(bucket, threadIdx, threadsPerBlock);
	__syncthreads();
end
]]--


util.max = terra(x : double, y : double)
	return terralib.select(x > y, x, y)
end

local function noHeader(pd)
	return quote end
end

local function noFooter(pd)
	return quote end
end

util.getImages = function(PlanData, images)
	local results = terralib.newlist()
	for i, field in ipairs(PlanData:getfield("images").type:getfields()) do
		results:insert(`field.type 
		 { data = [&uint8](images[i - 1])})
	end
	return results
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

util.makeComputeCost = function(data)
	local terra computeCost(pd : &data.PlanData)
		var result = 0.0
		--C.printf("W=%d,H=%d,size=%d,stride=%d\n", pd.images.unknown:H(), pd.images.unknown:W())
		for h = 0, pd.images.unknown:H() do
			for w = 0, pd.images.unknown:W() do
				var v = data.problemSpec.cost.boundary(w, h, unpackstruct(pd.images))
				result = result + v
			end
		end
		return result
	end
	return computeCost
end

util.makeComputeGradient = function(data)
	-- haha ha
	local terra gradientHack(pd : &data.PlanData, w : int, h : int, values : data.imageType)
		return data.problemSpec.gradient.boundary(w, h, values, pd.images.image0)
	end
	
	local terra computeGradient(pd : &data.PlanData, gradientOut : data.imageType, values : data.imageType)
		for h = 0, gradientOut:H() do
			for w = 0, gradientOut:W() do
				gradientOut(w, h) = gradientHack(pd, w, h, values)
			end
		end
	end
	return computeGradient
end

util.makeComputeResiduals = function(data)
	-- haha ha
	local terra costHack(pd : &data.PlanData, w : int, h : int, values : data.imageType)
		return data.problemSpec.cost.boundary(w, h, values, pd.images.image0)
	end
	
	local terra computeResiduals(pd : &data.PlanData, values : data.imageType, residuals : data.imageType)
		for h = 0, values:H() do
			for w = 0, values:W() do
				residuals(w, h) = costHack(pd, w, h, values)
			end
		end
	end
	return computeResiduals
end

util.makeComputeDeltaCost = function(data)
	-- haha ha
	local terra costHack(pd : &data.PlanData, w : int, h : int, values : data.imageType)
		return data.problemSpec.cost.boundary(w, h, values, pd.images.image0)
	end
	
	local terra deltaCost(pd : &data.PlanData, baseResiduals : data.imageType, currentValues : data.imageType)
		var result : double = 0.0
		for h = 0, currentValues:H() do
			for w = 0, currentValues:W() do
				var residual = costHack(pd, w, h, currentValues)
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
		
		var bestAlpha, bestAlphaCost = cpu.lineSearchQuadraticMinimum(pd, baseValues, baseResiduals, searchDirectionA, valueStore, alphaGuess)
		var bestBeta, bestBetaCost = cpu.lineSearchQuadraticMinimum(pd, baseValues, baseResiduals, searchDirectionA, valueStore, alphaGuess)
		
		var jointCost = cpu.computeBiSearchCost(pd, baseValues, baseResiduals, searchDirectionA, searchDirectionB, bestAlpha, bestBeta, valueStore)
		
		if jointCost < bestAlphaCost and jointCost < bestBetaCost then
			return bestAlpha, bestBeta
		end
		
		logSolver("bi-search minimization failed")
		
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
						--useBruteForce = true
					--end
					--cpu.dumpLineSearch(baseValues, baseResiduals, searchDirection, valueStore, dataImages)
				end
			end
		end

		if useBruteForce then
			logSolver("brute-force line search\n")
			bestAlpha = cpu.lineSearchBruteForce(pd, baseValues, baseResiduals, searchDirection, valueStore)
		end
		
		return bestAlpha
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

local wrapGPUKernel = function(nakedKernel, PlanData, mapMemberName, params)
	local terra wrappedKernel(pd : PlanData, [params])
		var w = blockDim.x * blockIdx.x + threadIdx.x
		var h = blockDim.y * blockIdx.y + threadIdx.y
		
		if w < pd.images.[mapMemberName]:W() and h < pd.images.[mapMemberName]:H() then
			nakedKernel(&pd, w, h, params)
		end
	end
	
	wrappedKernel:setname(nakedKernel.name)
		
	return wrappedKernel
end


local makeGPULauncher = function(compiledKernel, kernelName, header, footer, tbl, PlanData, params)
	local terra GPULauncher(pd : &PlanData, [params])
		var launch = terralib.CUDAParams { (pd.images.unknown:W() - 1) / 32 + 1, (pd.images.unknown:H() - 1) / 32 + 1, 1, 32, 32, 1, 0, nil }
		[header(pd)]

		var stream : C.cudaStream_t = nil
		var timingInfo : TimingInfo 
		if ([timeIndividualKernels]) then
			C.cudaEventCreate(&timingInfo.startEvent)
			C.cudaEventCreate(&timingInfo.endEvent)
	        C.cudaEventRecord(timingInfo.startEvent, stream);
	    end

		compiledKernel(&launch, @pd, params)
		
		if ([timeIndividualKernels]) then
			C.cudaEventRecord(timingInfo.endEvent, stream);
			timingInfo.eventName = kernelName
			pd.timer.timingInfo:insert(timingInfo)
		end

		C.cudaDeviceSynchronize()
		[footer(pd)]
	end
	
	return GPULauncher
end

util.makeComputeCostGPU = function(data)
	local terra computeCost(pd : &data.PlanData, w : int, h : int)
		var cost = [float](data.problemSpec.cost.boundary(w, h, unpackstruct(pd.images)))
		cost = warpReduce(cost);
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
	return { kernel = computeCost, header = header, footer = footer, params = {}, mapMemberName = "unknown" }
end

util.makeComputeDeltaCostGPU = function(data)
	-- haha ha
	local terra costHack(pd : &data.PlanData, w : int, h : int, values : data.imageType)
		return data.problemSpec.cost.boundary(w, h, values, pd.images.image0)
	end
	local terra computeDeltaCost(pd : &data.PlanData, w : int, h : int, baseResiduals : data.imageType, currentValues : data.imageType)
		var residual = [float](costHack(pd, w, h, currentValues))
		var delta = residual - baseResiduals(w, h)
		delta = warpReduce(delta);
		if (laneid() == 0) then
			atomicAdd(pd.scratchF, delta)
		end
	end
	local function header(pd)
		return quote @pd.scratchF = 0.0f end
	end
	local function footer(pd)
		return quote return @pd.scratchF end
	end
	return { kernel = computeDeltaCost, header = header, footer = footer, params = {symbol(data.imageType), symbol(data.imageType)}, mapMemberName = "unknown" }
end


util.makeInnerProductReductionGPU = function(data)
	local terra innerProductReduction(pd : &data.PlanData, w : int, h : int, imageA : data.imageType, imageB : data.imageType)
		var v = imageA(w, h) * imageB(w, h)
		v = warpReduce(v);
		--TODO: switch to block reduce
		if (laneid() == 0) then
			atomicAdd(pd.scratchF, v)
		end
	end
	local function header(pd)
		return quote @pd.scratchF = 0.0f end
	end
	local function footer(pd)
		return quote return @pd.scratchF end
	end
	return { kernel = innerProductReduction, header = header, footer = footer, params = {symbol(data.imageType), symbol(data.imageType)}, mapMemberName = "unknown" }
end

util.makeComputeGradientGPU = function(data)
	local terra computeGradient(pd : &data.PlanData, w : int, h : int, gradientOut : data.imageType)
		gradientOut(w, h) = data.problemSpec.gradient.boundary(w, h, unpackstruct(pd.images))
	end
	return { kernel = computeGradient, header = noHeader, footer = noFooter, params = {symbol(data.imageType)}, mapMemberName = "unknown" }
end

util.makeCopyImageGPU = function(data)
	local terra copyImage(pd : &data.PlanData, w : int, h : int, imageOut : data.imageType, imageIn : data.imageType)
		imageOut(w, h) = imageIn(w, h)
	end
	return { kernel = copyImage, header = noHeader, footer = noFooter, params = {symbol(data.imageType), symbol(data.imageType)}, mapMemberName = "unknown" }
end

util.makeCopyImageScaleGPU = function(data)
	local terra copyImageScale(pd : &data.PlanData, w : int, h : int, imageOut : data.imageType, imageIn : data.imageType, scale : float)
		imageOut(w, h) = imageIn(w, h) * scale
	end
	return { kernel = copyImageScale, header = noHeader, footer = noFooter, params = {symbol(data.imageType), symbol(data.imageType), symbol(float)}, mapMemberName = "unknown" }
end

util.makeAddImageGPU = function(data)
	local terra addImage(pd : &data.PlanData, w : int, h : int, imageOut : data.imageType, imageIn : data.imageType, scale : float)
		imageOut(w, h) = imageOut(w, h) + imageIn(w, h) * scale
	end
	return { kernel = addImage, header = noHeader, footer = noFooter, params = {symbol(data.imageType), symbol(data.imageType), symbol(float)}, mapMemberName = "unknown" }
end

-- out = A + B * scale
util.makeCombineImageGPU = function(data)
	local terra combineImage(pd : &data.PlanData, w : int, h : int, imageOut : data.imageType, imageA : data.imageType, imageB : data.imageType, scale : float)
		imageOut(w, h) = imageA(w, h) + imageB(w, h) * scale
	end
	return { kernel = combineImage, header = noHeader, footer = noFooter, params = {symbol(data.imageType), symbol(data.imageType), symbol(data.imageType), symbol(float)}, mapMemberName = "unknown" }
end

-- TODO: residuals should map over cost, not unknowns!!
util.makeComputeResidualsGPU = function(data)
	-- haha ha
	local terra costHack(pd : &data.PlanData, w : int, h : int, values : data.imageType)
		return data.problemSpec.cost.boundary(w, h, values, pd.images.image0)
	end
	local terra computeResiduals(pd : &data.PlanData, w : int, h : int, residuals : data.imageType, values : data.imageType)
		residuals(w, h) = costHack(pd, w, h, values)
	end
	return { kernel = computeResiduals, header = noHeader, footer = noFooter, params = {symbol(data.imageType), symbol(data.imageType)}, mapMemberName = "unknown" }
end

util.makeComputeSearchCostGPU = function(data, gpu)
	local terra computeSearchCost(pd : &data.PlanData, baseValues : data.imageType, baseResiduals : data.imageType, searchDirection : data.imageType, alpha : float, valueStore : data.imageType)

		gpu.combineImage(pd, valueStore, baseValues, searchDirection, alpha)
		return gpu.computeDeltaCost(pd, baseResiduals, valueStore)
	end
	return computeSearchCost
end

util.makeComputeSearchCostParallelGPU = function(data, gpu)
	local terra computeSearchCostParallel(pd : &data.PlanData, baseValues : data.imageType, baseResiduals : data.imageType, searchDirection : data.imageType, count : int, alphas : &float, costs : &float, valueStore : data.imageType)
		for i = 0, count do
			costs[i] = gpu.computeSearchCost(pd, baseValues, baseResiduals, searchDirection, alphas[i], valueStore)
		end
	end
	return computeSearchCostParallel
end

util.makeLineSearchBruteForceGPU = function(data, gpu)
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
			
			var searchCost = gpu.computeSearchCost(pd, baseValues, baseResiduals, searchDirection, alpha, valueStore)
			
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

util.makeLineSearchQuadraticMinimumGPU = function(data, gpu)
	local terra lineSearchQuadraticMinimum(pd : &data.PlanData, baseValues : data.imageType, baseResiduals : data.imageType, searchDirection : data.imageType, valueStore : data.imageType, alphaGuess : float)

		var alphas : float[4] = array(alphaGuess * 0.5f, alphaGuess * 1.0f, alphaGuess * 1.5f, 0.0f)
		var costs : float[4]
		
		gpu.computeSearchCostParallel(pd, baseValues, baseResiduals, searchDirection, 3, alphas, costs, valueStore)
		
		var a1 = alphas[0] var a2 = alphas[1] var a3 = alphas[2]
		var c1 = costs[0] var c2 = costs[1] var c3 = costs[2]
		var a = ((c2-c1)*(a1-a3) + (c3-c1)*(a2-a1))/((a1-a3)*(a2*a2-a1*a1) + (a2-a1)*(a3*a3-a1*a1))
		var b = ((c2 - c1) - a * (a2*a2 - a1*a1)) / (a2 - a1)
		alphas[3] = -b / (2.0 * a)
		costs[3] = gpu.computeSearchCost(pd, baseValues, baseResiduals, searchDirection, alphas[3], valueStore)
		
		var bestCost = 0.0
		var bestAlpha = 0.0
		for i = 0, 4 do
			if costs[i] < bestCost then
				bestAlpha = alphas[i]
				bestCost = costs[i]
			elseif i == 3 then
				logSolver("quadratic minimization failed, bestAlpha=%f\n", bestAlpha)
				--gpu.dumpLineSearch(baseValues, baseResiduals, searchDirection, valueStore, dataImages)
			end
		end
		
		return bestAlpha
	end
	return lineSearchQuadraticMinimum
end

util.makeLineSearchQuadraticFallbackGPU = function(data, gpu)
	local terra lineSearchQuadraticFallback(pd : &data.PlanData, baseValues : data.imageType, baseResiduals : data.imageType, searchDirection : data.imageType, valueStore : data.imageType, alphaGuess : float)
		var bestAlpha = 0.0
		var useBruteForce = (alphaGuess == 0.0)
		if not useBruteForce then
			
			bestAlpha = gpu.lineSearchQuadraticMinimum(pd, baseValues, baseResiduals, searchDirection, valueStore, alphaGuess)
			
			if bestAlpha == 0.0 then
				logSolver("quadratic guess=%f failed, trying again...\n", alphaGuess)
				bestAlpha = gpu.lineSearchQuadraticMinimum(pd, baseValues, baseResiduals, searchDirection, valueStore, alphaGuess * 4.0)
				
				if bestAlpha == 0.0 then
					logSolver("quadratic minimization exhausted\n")
					
					--if iter >= 10 then
					--else
						--useBruteForce = true
					--end
					--gpu.dumpLineSearch(baseValues, baseResiduals, searchDirection, valueStore, dataImages)
				end
			end
		end

		if useBruteForce then
			logSolver("brute-force line search\n")
			bestAlpha = gpu.lineSearchBruteForce(pd, baseValues, baseResiduals, searchDirection, valueStore)
		end
		
		return bestAlpha
	end
	return lineSearchQuadraticFallback
end

util.makeCPUFunctions = function(problemSpec, vars, PlanData)
	local cpu = {}
	
	local data = {}
	data.problemSpec = problemSpec
	data.PlanData = PlanData
	data.imageType = vars.unknownType
	
	cpu.copyImage = util.makeCopyImage(data.imageType)
	cpu.setImage = util.makeSetImage(data.imageType)
	cpu.addImage = util.makeAddImage(data.imageType)
	cpu.scaleImage = util.makeScaleImage(data.imageType)
	cpu.clearImage = util.makeClearImage(data.imageType)
	cpu.innerProduct = util.makeInnerProduct(data.imageType)
	
	cpu.computeCost = util.makeComputeCost(data)
	cpu.computeGradient = util.makeComputeGradient(data)
	cpu.deltaCost = util.makeComputeDeltaCost(data)
	cpu.computeResiduals = util.makeComputeResiduals(data)
	
	cpu.computeSearchCost = util.makeComputeSearchCost(data, cpu)
	cpu.computeBiSearchCost = util.makeComputeBiSearchCost(data, cpu)
	cpu.computeSearchCostParallel = util.makeComputeSearchCostParallel(data, cpu)
	cpu.dumpLineSearch = util.makeDumpLineSearch(data, cpu)
	cpu.dumpBiLineSearch = util.makeDumpBiLineSearch(data, cpu)
	cpu.lineSearchBruteForce = util.makeLineSearchBruteForce(data, cpu)
	cpu.lineSearchQuadraticMinimum = util.makeLineSearchQuadraticMinimum(data, cpu)
	cpu.lineSearchQuadraticFallback = util.makeLineSearchQuadraticFallback(data, cpu)
	cpu.biLineSearch = util.makeBiLineSearch(data, cpu)
	
	return cpu
end

util.makeGPUFunctions = function(problemSpec, vars, PlanData, specializedKernels)
	local gpu = {}
	local kernelTemplate = {}
	local wrappedKernels = {}
	
	local data = {}
	data.problemSpec = problemSpec
	data.PlanData = PlanData
	data.imageType = vars.unknownType
	
	-- accumulate all naked kernels
	kernelTemplate.computeCost = util.makeComputeCostGPU(data)
	kernelTemplate.computeGradient = util.makeComputeGradientGPU(data)
	kernelTemplate.copyImage = util.makeCopyImageGPU(data)
	kernelTemplate.copyImageScale = util.makeCopyImageScaleGPU(data)
	kernelTemplate.addImage = util.makeAddImageGPU(data)
	kernelTemplate.combineImage = util.makeCombineImageGPU(data)
	kernelTemplate.computeDeltaCost = util.makeComputeDeltaCostGPU(data)
	kernelTemplate.computeResiduals = util.makeComputeResidualsGPU(data)
	kernelTemplate.innerProduct = util.makeInnerProductReductionGPU(data)
		
	for k, v in pairs(specializedKernels) do
		kernelTemplate[k] = v(data)
	end
	
	-- clothe naked kernels
	for k, v in pairs(kernelTemplate) do
		wrappedKernels[k] = wrapGPUKernel(v.kernel, PlanData, v.mapMemberName, v.params)
	end
	
	local compiledKernels = terralib.cudacompile(wrappedKernels)
	
	for k, v in pairs(compiledKernels) do
		gpu[k] = makeGPULauncher(compiledKernels[k], wrappedKernels[k].name, kernelTemplate[k].header, kernelTemplate[k].footer, problemSpec, PlanData, kernelTemplate[k].params)
	end
	
	-- composite GPU functions
	gpu.computeSearchCost = util.makeComputeSearchCostGPU(data, gpu)
	gpu.computeSearchCostParallel = util.makeComputeSearchCostParallelGPU(data, gpu)
	gpu.lineSearchBruteForce = util.makeLineSearchBruteForceGPU(data, gpu)
	gpu.lineSearchQuadraticMinimum = util.makeLineSearchQuadraticMinimumGPU(data, gpu)
	gpu.lineSearchQuadraticFallback = util.makeLineSearchQuadraticFallbackGPU(data, gpu)
	
	return gpu
end

return util
