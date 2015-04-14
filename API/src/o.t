
--terralib.settypeerrordebugcallback( function(fn) fn:printpretty() end )

opt = {} --anchor it in global namespace, otherwise it can be collected
local S = require("std")

local C = terralib.includecstring [[
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
]]

local function newclass(name)
    local mt = { __name = name }
    mt.__index = mt
    function mt:is(obj)
        return getmetatable(obj) == self
    end
    function mt:__tostring()
        return "<"..name.." instance>"
    end
    function mt:new(obj)
        obj = obj or {}
        setmetatable(obj,self)
        return obj
    end
    return mt
end

local vprintf = terralib.externfunction("cudart:vprintf", {&int8,&int8} -> int)
local function createbuffer(args)
    local Buf = terralib.types.newstruct()
    return quote
        var buf : Buf
        escape
            for i,e in ipairs(args) do
                local typ = e:gettype()
                local field = "_"..tonumber(i)
                typ = typ == float and double or typ
                table.insert(Buf.entries,{field,typ})
                emit quote
                   buf.[field] = e
                end
            end
        end
    in
        [&int8](&buf)
    end
end
printf = macro(function(fmt,...)
    local buf = createbuffer({...})
    return `vprintf(fmt,buf) 
end)

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

__syncthreads = cudalib.nvvm_barrier0

local Problem = newclass("Problem")
local Dim = newclass("dimension")


local ffi = require('ffi')

local problems = {}

local terra max(x : double, y : double)
	return terralib.select(x > y, x, y)
end

-- this function should do anything it needs to compile an optimizer defined
-- using the functions in tbl, using the optimizer 'kind' (e.g. kind = gradientdecesnt)
-- it should generat the field makePlan which is the terra function that 
-- allocates the plan

local function getImages(vars,imageBindings,actualDims)
	local results = terralib.newlist()
	for i,argumentType in ipairs(vars.argumentTypes) do
		local Windex,Hindex = vars.dimIndex[argumentType.metamethods.W],vars.dimIndex[argumentType.metamethods.H]
		assert(Windex and Hindex)
		results:insert(`argumentType 
		 { W = actualDims[Windex], 
		   H = actualDims[Hindex], 
		   impl = 
		   @imageBindings[i - 1]}) 
	end
	return results
end

local function makeTotalCost(tbl, PlanData, images)
	local terra totalCost(pd : &PlanData, [images])
		var result = 0.0
		for h = 0,pd.gradH do
			for w = 0,pd.gradW do
				var v = tbl.cost.fn(w,h,images)
				result = result + v
			end
		end
		return result
	end
	return totalCost
end

local function gradientDescentCPU(tbl,vars)
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		gradStore : vars.unknownType
		dims : int64[#vars.dims + 1]
	}

	local images = vars.argumentTypes:map(symbol)
	local totalCost = makeTotalCost(tbl, PlanData, images)

	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)

		--C.printf("impl start\n")

		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dims = pd.dims

		--C.printf("impl A\n")

		var [images] = [getImages(vars, imageBindings, dims)]

		--C.printf("impl B\n")

		-- TODO: parameterize these
		var initialLearningRate = 0.01
		var maxIters = 5000
		var tolerance = 1e-10

		-- Fixed constants (these do not need to be parameterized)
		var learningLoss = 0.8
		var learningGain = 1.1
		var minLearningRate = 1e-25


		var learningRate = initialLearningRate

		--C.printf("impl D\n")

		for iter = 0, maxIters do

			--C.printf("impl iter start\n")

			var startCost = totalCost(pd, images)
			C.printf("iteration %d, cost=%f, learningRate=%f\n", iter, startCost, learningRate)
			--C.getchar()

			--
			-- compute the gradient
			--
			for h = 0,pd.gradH do
				for w = 0,pd.gradW do
					pd.gradStore(w,h) = tbl.gradient(w,h,images)
					--C.printf("%d,%d = %f\n",w,h,pd.gradStore(w,h))
					--C.getchar()
				end
			end

			--
			-- move along the gradient by learningRate
			--
			var maxDelta = 0.0
			for h = 0,pd.gradH do
				for w = 0,pd.gradW do
					var addr = &[ images[1] ](w,h)
					var delta = learningRate * pd.gradStore(w,h)
					--C.printf("delta (%d,%d)=%f\n", w, h, delta)
					@addr = @addr - delta
					maxDelta = max(C.fabsf(delta), maxDelta)
				end
			end

			--
			-- update the learningRate
			--
			var endCost = totalCost(pd, images)
			if endCost < startCost then
				learningRate = learningRate * learningGain

				if maxDelta < tolerance then
					C.printf("terminating, maxDelta=%f\n", maxDelta)
					break
				end
			else
				learningRate = learningRate * learningLoss

				if learningRate < minLearningRate then
					C.printf("terminating, learningRate=%f\n", learningRate)
					break
				end
			end

			--C.printf("impl iter end\n")
		end
		
		--[[var x00 = [ images[1] ](0, 0)
		var x10 = [ images[1] ](5, 0)
		var x01 = [ images[1] ](0, 5)
		
		C.printf("x(0,0) = %f\n", x00)
		C.printf("x(1,0) = %f\n", x10)
		C.printf("x(0,1) = %f\n", x01)]]
	end

	local gradWIndex = vars.dimIndex[ vars.gradientDim[1] ]
	local gradHIndex = vars.dimIndex[ vars.gradientDim[2] ]

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dims[0] = 1
		for i = 0,[#vars.dims] do
			pd.dims[i+1] = actualDims[i]
		end

		pd.gradW = pd.dims[gradWIndex]
		pd.gradH = pd.dims[gradHIndex]

		pd.gradStore:initCPU(pd.gradW, pd.gradH)

		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

local function gradientDescentGPU(tbl,vars)

	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		gradStore : vars.unknownType
		scratchF : &float
		scratchD : &double
		dims : int64[#vars.dims + 1]
	}

	local images = vars.argumentTypes:map(symbol)

	local cuda = {}

	terra cuda.computeGradient(pd : PlanData, [images])
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		--printf("%d, %d\n", w, h)

		if w < pd.gradW and h < pd.gradH then
			pd.gradStore(w,h) = tbl.gradient(w,h,images)
		end
	end

	terra cuda.updatePosition(pd : PlanData, learningRate : double, maxDelta : &double, [images])
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		--printf("%d, %d\n", w, h)
		
		if w < pd.gradW and h < pd.gradH then
			var addr = &[ images[1] ](w,h)
			var delta = learningRate * pd.gradStore(w,h)
			@addr = @addr - delta

			delta = delta * delta
			var deltaD : double = delta
			var deltaI64 = @[&int64](&deltaD)
			--printf("delta=%f, deltaI=%d\n", delta, deltaI)
			terralib.asm(terralib.types.unit,"red.global.max.u64 [$0],$1;", "l,l", true, maxDelta, deltaI64)
		end
	end

	terra cuda.costSum(pd : PlanData, sum : &float, [images])
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		if w < pd.gradW and h < pd.gradH then
			var v = [float](tbl.cost.fn(w,h,images))
			terralib.asm(terralib.types.unit,"red.global.add.f32 [$0],$1;","l,f",true,sum,v)
		end
	end
	
	cuda = terralib.cudacompile(cuda, false)
	 
	local terra totalCost(pd : &PlanData, [images])
		var launch = terralib.CUDAParams { (pd.gradW - 1) / 32 + 1, (pd.gradH - 1) / 32 + 1,1, 32,32,1, 0, nil }

		@pd.scratchF = 0.0f
		cuda.costSum(&launch, @pd, pd.scratchF, [images])
		C.cudaDeviceSynchronize()

		return @pd.scratchF
	end

	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)
		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dims = pd.dims

		var [images] = [getImages(vars,imageBindings,dims)]

		-- TODO: parameterize these
		var initialLearningRate = 0.01
		var maxIters = 5000
		var tolerance = 1e-10

		-- Fixed constants (these do not need to be parameterized)
		var learningLoss = 0.8
		var learningGain = 1.1
		var minLearningRate = 1e-25

		var learningRate = initialLearningRate

		for iter = 0,maxIters do

			var startCost = totalCost(pd, images)
			C.printf("iteration %d, cost=%f, learningRate=%f\n", iter, startCost, learningRate)
			
			--
			-- compute the gradient
			--
			var launch = terralib.CUDAParams { (pd.gradW - 1) / 32 + 1, (pd.gradH - 1) / 32 + 1,1, 32,32,1, 0, nil }

			--[[
			{ "gridDimX", uint },
							{ "gridDimY", uint },
							{ "gridDimZ", uint },
							{ "blockDimX", uint },
							{ "blockDimY", uint },
							{ "blockDimZ", uint },
							{ "sharedMemBytes", uint },
							{"hStream" , terra.types.pointer(opaque) } }
							--]]
			--C.printf("gridDimX: %d, gridDimY %d, blickDimX %d, blockdimY %d\n", launch.gridDimX, launch.gridDimY, launch.blockDimX, launch.blockDimY)
			cuda.computeGradient(&launch, @pd, [images])
			C.cudaDeviceSynchronize()
			
			--
			-- move along the gradient by learningRate
			--
			@pd.scratchD = 0.0
			cuda.updatePosition(&launch, @pd, learningRate, pd.scratchD, [images])
			C.cudaDeviceSynchronize()
			C.printf("maxDelta %f\n", @pd.scratchD)
			var maxDelta = @pd.scratchD

			--
			-- update the learningRate
			--
			var endCost = totalCost(pd, images)
			if endCost < startCost then
				learningRate = learningRate * learningGain

				if maxDelta < tolerance then
					break
				end
			else
				learningRate = learningRate * learningLoss

				if learningRate < minLearningRate then
					break
				end
			end
		end
	end

	local gradWIndex = vars.dimIndex[ vars.gradientDim[1] ]
	local gradHIndex = vars.dimIndex[ vars.gradientDim[2] ]

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dims[0] = 1
		for i = 0,[#vars.dims] do
			pd.dims[i+1] = actualDims[i]
		end

		pd.gradW = pd.dims[gradWIndex]
		pd.gradH = pd.dims[gradHIndex]

		pd.gradStore:initGPU(pd.gradW, pd.gradH)
		C.cudaMallocManaged([&&opaque](&(pd.scratchF)), sizeof(float), C.cudaMemAttachGlobal)
		C.cudaMallocManaged([&&opaque](&(pd.scratchD)), sizeof(double), C.cudaMemAttachGlobal)

		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

local function conjugateGradientCPU(tbl, vars)
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		dims : int64[#vars.dims + 1]
				
		currentValues : vars.unknownType
		bestValues : vars.unknownType
		
		gradient : vars.unknownType
		prevGradient : vars.unknownType
		
		searchDirection : vars.unknownType
	}
	
	local images = vars.argumentTypes:map(symbol)
	local totalCost = makeTotalCost(tbl, PlanData, images)

	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)

		--C.printf("impl start\n")

		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dims = pd.dims

		--C.printf("impl A\n")

		var [images] = [getImages(vars, imageBindings, dims)]

		--C.printf("impl B\n")

		-- TODO: parameterize these
		var lineSearchMaxIters = 10000
		var lineSearchBruteForceStart = 1e-6
		var lineSearchBruteForceMultiplier = 1.1
		var lineSearchBinaryMinMultiplier = 0.25
		var lineSearchBinaryMaxMultiplier = 1.5
		
		var maxIters = 1000

		--C.printf("impl D\n")
		
		var file = C.fopen("C:/code/debug.txt", "wb")
		
		var prevBestAlpha = 0.0

		for iter = 0,maxIters do

			--C.printf("impl iter start\n")

			var iterStartCost = totalCost(pd, images)
			C.printf("iteration %d, cost=%f\n", iter, iterStartCost)
			--C.getchar()

			--
			-- compute the gradient
			--
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.gradient(w, h) = tbl.gradient(w, h, images)
				end
			end

			--
			-- compute the search direction
			--
			var beta = 0.0
			if iter == 0 then
				for h = 0, pd.gradH do
					for w = 0, pd.gradW do
						pd.searchDirection(w, h) = -pd.gradient(w, h)
					end
				end
			else
				var num = 0.0
				var den = 0.0
				
				--
				-- Polak-Ribiere conjugacy
				-- 
				for h = 0, pd.gradH do
					for w = 0, pd.gradW do
						var g = pd.gradient(w, h)
						var p = pd.prevGradient(w, h)
						num = num + (-g * (-g + p))
						den = den + p * p
					end
				end
				beta = max(num / den, 0.0)
				
				var epsilon = 1e-5
				if den > epsilon and den < epsilon then
					beta = 0.0
				end
				
				for h = 0, pd.gradH do
					for w = 0, pd.gradW do
						pd.searchDirection(w, h) = -pd.gradient(w, h) + beta * pd.searchDirection(w, h)
					end
				end
			end
			
			C.memcpy(pd.prevGradient.impl.data, pd.gradient.impl.data, sizeof(float) * pd.gradW * pd.gradH)
			
			C.memcpy(pd.currentValues.impl.data, [ images[1] ].impl.data, sizeof(float) * pd.gradW * pd.gradH)
			
			--
			-- line search
			--
			var bestAlpha = 0.0
			
			var useBruteForce = (iter <= 1) or prevBestAlpha == 0.0
			if not useBruteForce then
				
				-- TODO: figure out how terra constant arrays work
				var a1 = prevBestAlpha * 0.25
				var c1 = 0.0
				var a2 = prevBestAlpha * 0.5
				var c2 = 0.0
				var a3 = prevBestAlpha * 0.75
				var c3 = 0.0
				var bestCost = iterStartCost
				
				for alphaIndex = 0, 4 do
					var alpha = 0.0
					if 	   alphaIndex == 0 then alpha = a1
					elseif alphaIndex == 1 then alpha = a2
					elseif alphaIndex == 2 then alpha = a3
					else
						var a = ((c2-c1)*(a1-a3) + (c3-c1)*(a2-a1))/((a1-a3)*(a2*a2-a1*a1) + (a2-a1)*(a3*a3-a1*a1))
						var b = ((c2 - c1) - a * (a2*a2 - a1*a1)) / (a2 - a1)
						var c = c1 - a * a1 * a1 - b * a1
						-- 2ax + b = 0, x = -b / 2a
						alpha = -b / (2.0 * a)
					end
					 
					for h = 0,pd.gradH do
						for w = 0,pd.gradW do
							[ images[1] ](w, h) = pd.currentValues(w, h) + alpha * pd.searchDirection(w, h)
						end
					end
					
					var searchCost = totalCost(pd, images)
					
					if searchCost <= bestCost then
						bestAlpha = alpha
						bestCost = searchCost
					elseif alphaIndex == 3 then
						C.printf("quadratic minimization failed\n")
					end
					
					if 	   alphaIndex == 0 then c1 = searchCost
					elseif alphaIndex == 1 then c2 = searchCost
					elseif alphaIndex == 2 then c3 = searchCost end
				end
				if bestAlpha == 0.0 then useBruteForce = true end
			end
			--C.fprintf(file, "iter %d\n", iter)
			if useBruteForce then
				var alpha = lineSearchBruteForceStart
				var bestCost = iterStartCost
				
				for lineSearchIndex = 0, lineSearchMaxIters do
					alpha = alpha * lineSearchBruteForceMultiplier
					
					for h = 0,pd.gradH do
						for w = 0,pd.gradW do
							[ images[1] ](w, h) = pd.currentValues(w, h) + alpha * pd.searchDirection(w, h)
						end
					end
					
					var searchCost = totalCost(pd, images)
					
					--C.fprintf(file, "%f\t%f\n", alpha * 1000.0, searchCost / 1000000.0)
					
					if searchCost < bestCost then
						bestAlpha = alpha
						bestCost = searchCost
					else
						break
					end
				end
			end
			
			for h = 0,pd.gradH do
				for w = 0,pd.gradW do
					[ images[1] ](w, h) = pd.currentValues(w, h) + bestAlpha * pd.searchDirection(w, h)
				end
			end
			
			prevBestAlpha = bestAlpha
			
			C.printf("alpha=%f, beta=%f\n\n", bestAlpha, beta)
			if bestAlpha <= 1e-8 and beta == 0.0 then
				break
			end
			--C.printf("impl iter end\n")
		end
		C.fclose(file)
	end

	local gradWIndex = vars.dimIndex[ vars.gradientDim[1] ]
	local gradHIndex = vars.dimIndex[ vars.gradientDim[2] ]

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dims[0] = 1
		for i = 0,[#vars.dims] do
			pd.dims[i+1] = actualDims[i]
		end

		pd.gradW = pd.dims[gradWIndex]
		pd.gradH = pd.dims[gradHIndex]

		pd.currentValues:initCPU(pd.gradW, pd.gradH)
		pd.bestValues:initCPU(pd.gradW, pd.gradH)
		
		pd.gradient:initCPU(pd.gradW, pd.gradH)
		pd.prevGradient:initCPU(pd.gradW, pd.gradH)
		
		pd.searchDirection:initCPU(pd.gradW, pd.gradH)

		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

local function linearizedConjugateGradientCPU(tbl, vars)
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		dims : int64[#vars.dims + 1]
		
		b : vars.unknownType
		r : vars.unknownType
		p : vars.unknownType
		Ap : vars.unknownType
	}
	
	local imagesAll = vars.argumentTypes:map(symbol)
	local unknownImage = imagesAll[1]
	local dataImages = terralib.newlist()
	for i = 2,#imagesAll do
		dataImages:insert(imagesAll[i])
	end
	
	local totalCost = makeTotalCost(tbl, PlanData, imagesAll)

	-- TODO: ask zach how to do templates so this can live at a higher scope
	local terra imageInnerProduct(width : int, height : int, a : vars.unknownType, b : vars.unknownType)
		var sum = 0.0
		for h = 0, height do
			for w = 0, width do
				sum = sum + a(w, h) * b(w, h)
			end
		end
		return sum
	end

	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)
		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dims = pd.dims

		var [imagesAll] = [getImages(vars, imageBindings, dims)]
		
		-- TODO: parameterize these
		var maxIters = 1000
		var tolerance = 1e-5

		-- here, Ap is being used as storage for zeroes
		for h = 0, pd.gradH do
			for w = 0, pd.gradW do
				pd.Ap(w, h) = 0.0
			end
		end
		
		for h = 0, pd.gradH do
			for w = 0, pd.gradW do
				pd.r(w, h) = -tbl.gradient(w, h, unknownImage, dataImages)
				pd.b(w, h) = tbl.gradient(w, h, pd.Ap, dataImages)
				pd.p(w, h) = pd.r(w, h)
			end
		end
		
		var rTr = imageInnerProduct(pd.gradW, pd.gradH, pd.r, pd.r)

		for iter = 0,maxIters do

			var iterStartCost = totalCost(pd, imagesAll)
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					--TODO: more elegant gradientHack
					pd.Ap(w, h) = tbl.gradientHack(w, h, pd.p, imagesAll) - pd.b(w, h)
				end
			end
			
			var den = imageInnerProduct(pd.gradW, pd.gradH, pd.p, pd.Ap)
			var alpha = rTr / den
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					[ imagesAll[1] ](w, h) = [ imagesAll[1] ](w, h) + alpha * pd.p(w, h)
					pd.r(w, h) = pd.r(w, h) - alpha * pd.Ap(w, h)
				end
			end
			
			var rTrNew = imageInnerProduct(pd.gradW, pd.gradH, pd.r, pd.r)
			
			C.printf("iteration %d, cost=%f, rTr=%f\n", iter, iterStartCost, rTrNew)
			
			if(rTrNew < tolerance) then break end
			
			var beta = rTrNew / rTr
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.p(w, h) = pd.r(w, h) + beta * pd.p(w, h)
				end
			end
			
			rTr = rTrNew
		end
		
		var finalCost = totalCost(pd, imagesAll)
		C.printf("final cost=%f\n", finalCost)
	end

	local gradWIndex = vars.dimIndex[ vars.gradientDim[1] ]
	local gradHIndex = vars.dimIndex[ vars.gradientDim[2] ]

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dims[0] = 1
		for i = 0,[#vars.dims] do
			pd.dims[i+1] = actualDims[i]
		end

		pd.gradW = pd.dims[gradWIndex]
		pd.gradH = pd.dims[gradHIndex]

		pd.b:initCPU(pd.gradW, pd.gradH)
		pd.r:initCPU(pd.gradW, pd.gradH)
		pd.p:initCPU(pd.gradW, pd.gradH)
		pd.Ap:initCPU(pd.gradW, pd.gradH)
		
		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

local function linearizedConjugateGradientGPU(tbl, vars)
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		dims : int64[#vars.dims + 1]
		
		b : vars.unknownType
		r : vars.unknownType
		p : vars.unknownType
		Ap : vars.unknownType
		zeroes : vars.unknownType
		
		scratchF : &float
	}
	
	local images = vars.argumentTypes:map(symbol)

	local cuda = {}
	
	terra cuda.initialize(pd : PlanData, [images])
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		if w < pd.gradW and h < pd.gradH then
			pd.r(w, h) = -tbl.gradientHack(w, h, [ images[1] ], images)
			pd.b(w, h) = tbl.gradientHack(w, h, pd.zeroes, images)
			pd.p(w, h) = pd.r(w, h)
		end
	end
	
	terra cuda.computeAp(pd : PlanData, [images])
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		if w < pd.gradW and h < pd.gradH then
			pd.Ap(w, h) = tbl.gradientHack(w, h, pd.p, images) - pd.b(w, h)
		end
	end

	terra cuda.updateResidualAndPosition(pd : PlanData, alpha : float, [images])
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		if w < pd.gradW and h < pd.gradH then
			[ images[1] ](w, h) = [ images[1] ](w, h) + alpha * pd.p(w, h)
			pd.r(w, h) = pd.r(w, h) - alpha * pd.Ap(w, h)
		end
	end
	
	terra cuda.updateP(pd : PlanData, beta : float)
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		if w < pd.gradW and h < pd.gradH then
			pd.p(w, h) = pd.r(w, h) + beta * pd.p(w, h)
		end
	end
	
	terra cuda.costSum(pd : PlanData, sum : &float, [images])
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		if w < pd.gradW and h < pd.gradH then
			var v = [float](tbl.cost.fn(w, h, images))
			terralib.asm(terralib.types.unit,"red.global.add.f32 [$0],$1;","l,f", true, sum, v)
		end
	end
	
	terra cuda.imageInnerProduct(pd : PlanData, a : vars.unknownType, b : vars.unknownType, sum : &float)
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		if w < pd.gradW and h < pd.gradH then
			var v = [float](a(w, h) * b(w, h))
			terralib.asm(terralib.types.unit,"red.global.add.f32 [$0],$1;","l,f", true, sum, v)
		end
	end
	
	cuda = terralib.cudacompile(cuda, false)
	
	local terra totalCost(pd : &PlanData, [images])
		var launch = terralib.CUDAParams { (pd.gradW - 1) / 32 + 1, (pd.gradH - 1) / 32 + 1,1, 32,32,1, 0, nil }

		C.cudaDeviceSynchronize()
		@pd.scratchF = 0.0f
		C.cudaDeviceSynchronize()
		cuda.costSum(&launch, @pd, pd.scratchF, [images])
		C.cudaDeviceSynchronize()
		
		return @pd.scratchF
	end
	
	-- TODO: ask zach how to do templates so this can live at a higher scope
	local terra imageInnerProduct(pd : &PlanData, a : vars.unknownType, b : vars.unknownType)
		var launch = terralib.CUDAParams { (pd.gradW - 1) / 32 + 1, (pd.gradH - 1) / 32 + 1,1, 32,32,1, 0, nil }
		
		C.cudaDeviceSynchronize()
		@pd.scratchF = 0.0f
		C.cudaDeviceSynchronize()
		cuda.imageInnerProduct(&launch, @pd, a, b, pd.scratchF)
		C.cudaDeviceSynchronize()
		
		return @pd.scratchF
	end

	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)
		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dims = pd.dims

		var [images] = [getImages(vars, imageBindings, dims)]

		var launch = terralib.CUDAParams { (pd.gradW - 1) / 32 + 1, (pd.gradH - 1) / 32 + 1,1, 32,32,1, 0, nil }
		
		-- TODO: parameterize these
		var maxIters = 1000
		var tolerance = 1e-5

		C.cudaDeviceSynchronize()
		cuda.initialize(&launch, @pd, [images])
		C.cudaDeviceSynchronize()
		
		var rTr = imageInnerProduct(pd, pd.r, pd.r)

		for iter = 0, maxIters do
	
			var iterStartCost = totalCost(pd, images)
			
			C.cudaDeviceSynchronize()
			cuda.computeAp(&launch, @pd, [images])
			
			var den = imageInnerProduct(pd, pd.p, pd.Ap)
			var alpha = rTr / den
			
			--C.printf("den=%f, alpha=%f\n", den, alpha)
			
			cuda.updateResidualAndPosition(&launch, @pd, alpha, [images])
			
			var rTrNew = imageInnerProduct(pd, pd.r, pd.r)
			
			C.printf("iteration %d, cost=%f, rTr=%f\n", iter, iterStartCost, rTrNew)
			
			if(rTrNew < tolerance) then break end
			
			var beta = rTrNew / rTr
			cuda.updateP(&launch, @pd, beta)
			
			rTr = rTrNew
		end
		
		var finalCost = totalCost(pd, images)
		C.printf("final cost=%f\n", finalCost)
	end

	local gradWIndex = vars.dimIndex[ vars.gradientDim[1] ]
	local gradHIndex = vars.dimIndex[ vars.gradientDim[2] ]

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dims[0] = 1
		for i = 0,[#vars.dims] do
			pd.dims[i+1] = actualDims[i]
		end

		pd.gradW = pd.dims[gradWIndex]
		pd.gradH = pd.dims[gradHIndex]

		pd.b:initGPU(pd.gradW, pd.gradH)
		pd.r:initGPU(pd.gradW, pd.gradH)
		pd.p:initGPU(pd.gradW, pd.gradH)
		pd.Ap:initGPU(pd.gradW, pd.gradH)
		pd.zeroes:initGPU(pd.gradW, pd.gradH)
		
		C.cudaMallocManaged([&&opaque](&(pd.scratchF)), sizeof(float), C.cudaMemAttachGlobal)
		
		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

local function linearizedPreconditionedConjugateGradientCPU(tbl, vars)
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		dims : int64[#vars.dims + 1]
		
		b : vars.unknownType
		r : vars.unknownType
		z : vars.unknownType
		p : vars.unknownType
		MInv : vars.unknownType
		Ap : vars.unknownType
		zeroes : vars.unknownType
	}
	
	local images = vars.argumentTypes:map(symbol)

	-- TODO: ask zach how to factor out totalCost
	local terra totalCost(data_ : &opaque, [images])
		var pd = [&PlanData](data_)
		var result = 0.0
		for h = 0,pd.gradH do
			for w = 0,pd.gradW do
				var v = tbl.cost.fn(w, h, images)
				result = result + v
			end
		end
		return result
	end
	
	-- TODO: ask zach how to do templates so this can live at a higher scope
	local terra imageInnerProduct(pd : &PlanData, a : vars.unknownType, b : vars.unknownType)
		var sum = 0.0
		for h = 0, pd.gradH do
			for w = 0, pd.gradW do
				sum = sum + a(w, h) * b(w, h)
			end
		end
		return sum
	end

	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)
		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dims = pd.dims

		var [images] = [getImages(vars, imageBindings, dims)]

		-- TODO: parameterize these
		var maxIters = 1000
		var tolerance = 1e-5

		for h = 0, pd.gradH do
			for w = 0, pd.gradW do
				pd.MInv(w, h) = 1.0 / tbl.gradientPreconditioner(w, h)
			end
		end
		
		for h = 0, pd.gradH do
			for w = 0, pd.gradW do
				pd.r(w, h) = -tbl.gradientHack(w, h, [ images[1] ], images)
				pd.b(w, h) = tbl.gradientHack(w, h, pd.zeroes, images)
				pd.z(w, h) = pd.MInv(w, h) * pd.r(w, h)
				pd.p(w, h) = pd.z(w, h)
			end
		end
		
		for iter = 0,maxIters do

			var iterStartCost = totalCost(data_, images)
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					--TODO: more elegant gradientHack
					pd.Ap(w, h) = tbl.gradientHack(w, h, pd.p, images) - pd.b(w, h)
				end
			end
			
			var rTzStart = imageInnerProduct(pd, pd.r, pd.z)
			var den = imageInnerProduct(pd, pd.p, pd.Ap)
			var alpha = rTzStart / den
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					[ images[1] ](w, h) = [ images[1] ](w, h) + alpha * pd.p(w, h)
					pd.r(w, h) = pd.r(w, h) - alpha * pd.Ap(w, h)
				end
			end
			
			var rTr = imageInnerProduct(pd, pd.r, pd.r)
			
			C.printf("iteration %d, cost=%f, rTr=%f\n", iter, iterStartCost, rTr)
			
			if(rTr < tolerance) then break end
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.z(w, h) = pd.MInv(w, h) * pd.r(w, h)
				end
			end
			
			var beta = imageInnerProduct(pd, pd.z, pd.r) / rTzStart
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.p(w, h) = pd.z(w, h) + beta * pd.p(w, h)
				end
			end
		end
		
		var finalCost = totalCost(data_, images)
		C.printf("final cost=%f\n", finalCost)
	end

	local gradWIndex = vars.dimIndex[ vars.gradientDim[1] ]
	local gradHIndex = vars.dimIndex[ vars.gradientDim[2] ]

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dims[0] = 1
		for i = 0,[#vars.dims] do
			pd.dims[i+1] = actualDims[i]
		end

		pd.gradW = pd.dims[gradWIndex]
		pd.gradH = pd.dims[gradHIndex]

		pd.b:initCPU(pd.gradW, pd.gradH)
		pd.r:initCPU(pd.gradW, pd.gradH)
		pd.z:initCPU(pd.gradW, pd.gradH)
		pd.p:initCPU(pd.gradW, pd.gradH)
		pd.MInv:initCPU(pd.gradW, pd.gradH)
		pd.Ap:initCPU(pd.gradW, pd.gradH)
		pd.zeroes:initCPU(pd.gradW, pd.gradH)
		
		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

local function compileProblem(tbl, kind)
	local vars = {
		dims = tbl.dims,
		dimIndex = { [1] = 0 },
		costDim = tbl.cost.dim,
		costType = tbl.cost.fn:gettype()
	}
	
	vars.unknownType = vars.costType.parameters[3] -- 3rd argument is the image that is the unknown we are mapping over
	vars.argumentTypes = terralib.newlist()
	vars.gradientDim = { vars.unknownType.metamethods.W, vars.unknownType.metamethods.H }
		
    for i, d in ipairs(vars.dims) do
        assert(Dim:is(d))
        vars.dimIndex[d] = i -- index into DimList, 0th entry is always 1
    end

	for i = 3,#vars.costType.parameters do
		vars.argumentTypes:insert(vars.costType.parameters[i])
	end

    if kind == "simpleexample" then

		local struct PlanData(S.Object) {
			plan : opt.Plan
			dims : int64[#dims+1]
		}

		local function emitcalltouserfunction(mapdims,actualDims,images,userfn)
			local typ = userfn:gettype()
			local di,dj = vars.dimIndex[mapdims[1]],vars.dimIndex[mapdims[2]]
			assert(di and dj)
			local imagewrappers = terralib.newlist()
			for i = 3,#typ.parameters do
				local imgtyp = typ.parameters[i]
				local W,H = imgtyp.metamethods.W,imgtyp.metamethods.H
				W,H = vars.dimIndex[W],vars.dimIndex[H]
				assert(W and H)
				imagewrappers:insert(`imgtyp { W = actualDims[W], H = actualDims[H], impl = @images[i - 3] })
			end
			return `userfn(actualDims[di],actualDims[dj],imagewrappers)
		end
    
		local terra impl(data_ : &opaque, images : &&opt.ImageBinding, params : &opaque)
			var data = [&PlanData](data_)
			var dims = data.dims
			[ emitcalltouserfunction(costdim,dims,images,tbl.cost.fn) ]
			[ emitcalltouserfunction(vars.gradientDim,dims,images,tbl.gradient) ]
		end
		local terra makePlan(actualDims : &uint64) : &opt.Plan
			var pd = PlanData.alloc()
			pd.plan.data = pd
			pd.plan.impl = impl
			pd.dims[0] = 1
			for i = 0,[#dims] do
				pd.dims[i+1] = actualDims[i]
			end
			return &pd.plan
		end
		return Problem:new { makePlan = makePlan }

	elseif kind == "gradientdescentCPU" then
        return gradientDescentCPU(tbl, vars)
	elseif kind == "gradientdescentGPU" then
		return gradientDescentGPU(tbl, vars)
	elseif kind == "conjugateGradientCPU" then
		return conjugateGradientCPU(tbl, vars)
	elseif kind == "linearizedConjugateGradientCPU" then
		return linearizedConjugateGradientCPU(tbl, vars)
	elseif kind == "linearizedConjugateGradientGPU" then
		return linearizedConjugateGradientGPU(tbl, vars)
	elseif kind == "linearizedPreconditionedConjugateGradientCPU" then
		return linearizedPreconditionedConjugateGradientCPU(tbl, vars)
	end
    
end

function opt.ProblemDefineFromTable(tbl, kind, params)
    local p = compileProblem(tbl,kind,params)
    -- store each problem in a table, and assign it an id
    problems[#problems + 1] = p
    p.id = #problems
    return p
end

local function problemDefine(filename, kind, params, pt)
    pt.makePlan = nil
    local success,p = xpcall(function() 
        filename,kind = ffi.string(filename), ffi.string(kind)
        local tbl = assert(terralib.loadfile(filename))() 
        assert(type(tbl) == "table")
        local p = opt.ProblemDefineFromTable(tbl,kind,params)
		pt.id, pt.makePlan = p.id,p.makePlan:getpointer()
		return p
    end,function(err) print(debug.traceback(err,2)) end)
end

struct opt.GradientDescentPlanParams {
    nIterations : uint64
}

struct opt.ImageBinding(S.Object) {
    data : &uint8
    stride : uint64
    elemsize : uint64
}
terra opt.ImageBinding:get(x : uint64, y : uint64) : &uint8
   return self.data + y * self.stride + x * self.elemsize
end

terra opt.ImageBind(data : &opaque, elemsize : uint64, stride : uint64) : &opt.ImageBinding
    var img = opt.ImageBinding.alloc()
    img.data,img.stride,img.elemsize = [&uint8](data),stride,elemsize
    return img
end
terra opt.ImageFree(img : &opt.ImageBinding)
    img:delete()
end

struct opt.Plan(S.Object) {
    impl : {&opaque,&&opt.ImageBinding,&opaque} -> {}
    data : &opaque
} 

struct opt.Problem(S.Object) {
    id : int
    makePlan : &uint64 -> &opt.Plan
}

terra opt.ProblemDefine(filename : rawstring, kind : rawstring, params : &opaque)
    var pt = opt.Problem.alloc()
    problemDefine(filename,kind,params,pt)
	if pt.makePlan == nil then
		pt:delete()
		return nil
	end
    return pt
end 

terra opt.ProblemDelete(p : &opt.Problem)
    -- TODO: deallocate any problem state here (remove from the Lua table,etc.)
    p:delete() 
end

function opt.Dim(name)
    return Dim:new { name = name }
end

local newImage = terralib.memoize(function(typ, W, H)
	local struct Image {
		impl : opt.ImageBinding
		W : uint64
		H : uint64
	}
	function Image.metamethods.__tostring()
	  return string.format("Image(%s,%s,%s)",tostring(typ),W.name, H.name)
	end
	Image.metamethods.__apply = macro(function(self, x, y)
	 return `@[&typ](self.impl:get(x,y))
	end)
	terra Image:initCPU(actualW : int, actualH : int)
		self.W = actualW
		self.H = actualH
		self.impl.data = [&uint8](C.malloc(actualW * actualH * sizeof(typ)))
		self.impl.elemsize = sizeof(typ)
		self.impl.stride = actualW * sizeof(typ)
		
		for h = 0, self.H do
			for w = 0, self.W do
				self(w, h) = 0.0
			end
		end
	end
	terra Image:initGPU(actualW : int, actualH : int)
		self.W = actualW
		self.H = actualH
		var typeSize = sizeof(typ)
		--C.printf("typesize=%d\n", typeSize)
		var cudaError = C.cudaMalloc([&&opaque](&(self.impl.data)), actualW * actualH * typeSize)
		cudaError = C.cudaMemset([&opaque](self.impl.data), 0, actualW * actualH * typeSize)
		self.impl.elemsize = typeSize
		self.impl.stride = actualW * typeSize
	end
	terra Image:debugGPUPrint()
		
		--var cpuImage : type(self)
		--cpuImage:initCPU(self.W, self.H)
		
		--C.cudaMemcpy(dataGPU, cpuImage, sizeof(float) * dimX * dimY, cudaMemcpyHostToDevice)
	end
	Image.metamethods.typ,Image.metamethods.W,Image.metamethods.H = typ, W, H
	return Image
end)

function opt.Image(typ, W, H)
    assert(W == 1 or Dim:is(W))
    assert(H == 1 or Dim:is(H))
    assert(terralib.types.istype(typ))
    return newImage(typ, W, H)
end

terra opt.ProblemPlan(problem : &opt.Problem, dims : &uint64) : &opt.Plan
	return problem.makePlan(dims) -- this is just a wrapper around calling the plan constructor
end 

terra opt.PlanFree(plan : &opt.Plan)
    -- TODO: plan should also have a free implementation
    plan:delete()
end

terra opt.ProblemSolve(plan : &opt.Plan, images : &&opt.ImageBinding, params : &opaque)
	return plan.impl(plan.data, images, params)
end

ad = require("ad")


local accesstovar = {} -- map from image(x,y) to ad variable object that represents that value
local accesstobounds = {} -- map from image(x,y) to ad variable representing the boundary info
local varnames = {}

local nextvar = 1
local function getnextvar(n,x,y)
    local xn,yn = tostring(x):gsub("-","m"),tostring(y):gsub("-","m")
    nextvar = nextvar + 1
    varnames[nextvar] = ("%s_%s_%s"):format(n,xn,yn)
    return ad.v[nextvar]
end
local ImageAccess = terralib.memoize(function(im,x,y)
    local a = { image = im, x = x, y = y}
    accesstovar[a],accesstobounds[a] = getnextvar(im.name,x,y),getnextvar(im.name.."_bounds",x,y)
    return a
end)

local Image = newclass("Image")
-- Z: this will eventually be opt.Image, but that is currently used by our direct methods
-- so this is going in the ad table for now
function ad.Image(name,W,H)
    assert(W == 1 or Dim:is(W))
    assert(H == 1 or Dim:is(H))
    return Image:new { name = tostring(name), W = W, H = H }
end

function Image:inbounds(x,y)
    x,y = assert(tonumber(x)),assert(tonumber(y))
    return assert(accesstobounds[ImageAccess(self,x,y)])
end
function Image:__call(x,y)
    x,y = assert(tonumber(x)),assert(tonumber(y))
    return assert(accesstovar[ImageAccess(self,x,y)])
end

function ad.Cost(dims,images,costexp)
    assert(#images > 0)
    --TODO: check that Dims used in images are in dims list
    --TODO: check images are all images 
    costexp = assert(ad.toexp(costexp))
    --TODO: check all image uses in costexp are bound to images in list
    local unknown = images[1] -- assume for now that the first image is the unknown
    --TODO: code gen cost and gradient
    print(ad.tostrings({assert(costexp)}, varnames))
    
    local realvars = terralib.newlist()
    local rvnames = terralib.newlist()
    for k,v in pairs(accesstovar) do
        local n = varnames[v:N()]
        if n:match("X") then
            realvars:insert(v)
            rvnames:insert(n)
        end
    end
    print(ad.tostrings(costexp:gradient(realvars), varnames))
    print(unpack(rvnames))
    error("TODO: NYI!")
end
return opt