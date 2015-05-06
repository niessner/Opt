
local S = require("std")
local util = require("util")
local C = util.C
local Timer = util.Timer
local positionForValidLane = util.positionForValidLane

local gpuMath = util.gpuMath

solversGPU = {}

local function noHeader(pd)
	return quote end
end

local function noFooter(pd)
	return quote end
end

local FLOAT_EPSILON = `0.000001f
-- GAUSS NEWTON (non-block version)
solversGPU.gaussNewtonGPU = function(problemSpec, vars)

	local struct PlanData(S.Object) {
		plan : opt.Plan
		images : vars.PlanImages
		scratchF : &float
		
		delta : vars.unknownType			--current linear update to be computed -> num vars
		r : vars.unknownType				--residuals -> num vars	--TODO this needs to be a 'residual type'
		z : vars.unknownType				--preconditioned residuals -> num vars	--TODO this needs to be a 'residual type'
		p : vars.unknownType				--decent direction -> num vars
		Ap_X : vars.unknownType				--cache values for next kernel call after A = J^T x J x p -> num vars
		preconditioner : vars.unknownType	--pre-conditioner for linear system -> num vars
		rDotzOld : vars.unknownType			--Old nominator (denominator) of alpha (beta) -> num vars	

		scanAlpha : &float					-- tmp variable for alpha scan
		scanBeta : &float					-- tmp variable for alpha scan
		
		timer : Timer
	}
	
	local specializedKernels = {}
	specializedKernels.PCGInit1 = function(data)
		local terra PCGInit1GPU(pd : &data.PlanData)
			var d = 0.0f -- init for out of bounds lanes
			var w : int, h : int
			if positionForValidLane(pd, "unknown", &w, &h) then
				var residuum = -data.problemSpec.gradient.boundary(w, h, unpackstruct(pd.images))	-- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 
				pd.r(w, h) = residuum

				-- TODO pd.precondition(w,h) needs to computed somehow (ideally in the gradient?
				-- TODO: don't let this be 0
				pd.preconditioner(w, h) = 1 --data.problemSpec.gradientPreconditioner(w, h)	-- TODO fix this hack... the pre-conditioner needs to be the diagonal of JTJ
				--pd.preconditioner(w,h) = data.problemSpec.gradientPreconditioner(w,h)
				var p = pd.preconditioner(w, h)*residuum				   -- apply pre-conditioner M^-1
				pd.p(w, h) = p
			
				d = residuum*p;										   -- x-th term of nominator for computing alpha and denominator for computing beta
			end 
			d = util.warpReduce(d)	
			if (util.laneid() == 0) then
				util.atomicAdd(pd.scanAlpha, d)
			end
		end
		return { kernel = PCGInit1GPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "unknown" }
	end
	
	specializedKernels.PCGInit2 = function(data)
		local terra PCGInit2GPU(pd : &data.PlanData)
			var w : int, h : int
			if positionForValidLane(pd, "unknown", &w, &h) then
				pd.rDotzOld(w,h) = pd.scanAlpha[0]
				pd.delta(w,h) = 0.0f
			end
		end
		return { kernel = PCGInit2GPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "unknown" }
	end
	
	specializedKernels.PCGStep1 = function(data)
		local terra PCGStep1GPU(pd : &data.PlanData)
			var d = 0.0f
			var w : int, h : int
			if positionForValidLane(pd, "unknown", &w, &h) then
				var tmp = data.problemSpec.applyJTJ.boundary(w, h, unpackstruct(pd.images), pd.p) -- A x p_k  => J^T x J x p_k 
				pd.Ap_X(w, h) = tmp								  -- store for next kernel call
				d = pd.p(w, h)*tmp					              -- x-th term of denominator of alpha
			end
			d = util.warpReduce(d)
			if (util.laneid() == 0) then
				util.atomicAdd(pd.scanAlpha, d)
			end
		end
		return { kernel = PCGStep1GPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "unknown" }
	end
	
	specializedKernels.PCGStep2 = function(data)
		local terra PCGStep2GPU(pd : &data.PlanData)
			var b = 0.0f 
			var w : int, h : int
			if positionForValidLane(pd, "unknown", &w, &h) then
				-- sum over block results to compute denominator of alpha
				var dotProduct = pd.scanAlpha[0]
				var alpha = 0.0f
				
				-- update step size alpha
				if dotProduct > FLOAT_EPSILON then alpha = pd.rDotzOld(w, h)/dotProduct end 
			
				pd.delta(w, h) = pd.delta(w, h)+alpha*pd.p(w,h)		-- do a decent step
				
				var r = pd.r(w,h)-alpha*pd.Ap_X(w,h)				-- update residuum
				pd.r(w,h) = r										-- store for next kernel call
			
				var z = pd.preconditioner(w,h)*r						-- apply pre-conditioner M^-1
				pd.z(w,h) = z;										-- save for next kernel call
				
				b = z*r;											-- compute x-th term of the nominator of beta
			end
			b = util.warpReduce(b)
			if (util.laneid() == 0) then
				util.atomicAdd(pd.scanBeta, b)
			end
		end
		return { kernel = PCGStep2GPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "unknown" }
	end
	
	specializedKernels.PCGStep3 = function(data)
		local terra PCGStep3GPU(pd : &data.PlanData)			
			var w : int, h : int
			if positionForValidLane(pd, "unknown", &w, &h) then
				var rDotzNew =  pd.scanBeta[0]									-- get new nominator
				var rDotzOld = pd.rDotzOld(w,h)									-- get old denominator

				var beta = 0.0f														 
				if rDotzOld > FLOAT_EPSILON then beta = rDotzNew/rDotzOld end	-- update step size beta
			
				pd.rDotzOld(w,h) = rDotzNew										-- save new rDotz for next iteration
				pd.p(w,h) = pd.z(w,h)+beta*pd.p(w,h)							-- update decent direction
			end
		end
		return { kernel = PCGStep3GPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "unknown" }
	end
	
	specializedKernels.PCGLinearUpdate = function(data)
		local terra PCGLinearUpdateGPU(pd : &data.PlanData)
			var w : int, h : int
			if positionForValidLane(pd, "unknown", &w, &h) then
				pd.images.unknown(w,h) = pd.images.unknown(w,h) + pd.delta(w,h)
			end
		end
		return { kernel = PCGLinearUpdateGPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "unknown" }
	end

	local gpu = util.makeGPUFunctions(problemSpec, vars, PlanData, specializedKernels)
	
	local terra impl(data_ : &opaque, images : &&opaque, params_ : &opaque)
		var pd = [&PlanData](data_)
		pd.timer:init()

		var params = [&double](params_)

		unpackstruct(pd.images) = [util.getImages(PlanData, images)]

		var nIterations = 30	--non-linear iterations
		var lIterations = 30	--linear iterations
		
		for nIter = 0, nIterations do
            --var startCost = gpu.computeCost(pd, pd.images.unknown)
			--logSolver("iteration %d, cost=%f", nIter, startCost)
			pd.scanAlpha[0] = 0.0	--scan in PCGInit1 requires reset
			gpu.PCGInit1(pd)
			--var a = pd.scanAlpha[0]
			--C.printf("Alpha %15.15f\n", a)
			--break
			gpu.PCGInit2(pd)
			
			for lIter = 0, lIterations do
				pd.scanAlpha[0] = 0.0	--scan in PCGStep1 requires reset
				gpu.PCGStep1(pd)
				pd.scanBeta[0] = 0.0	--scan in PCGStep2 requires reset
				gpu.PCGStep2(pd)
				gpu.PCGStep3(pd)
			end
			
			gpu.PCGLinearUpdate(pd)
		end

		pd.timer:evaluate()
		pd.timer:cleanup()
	end

	local terra makePlan() : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl

		pd.delta:initGPU()
		pd.r:initGPU()
		pd.z:initGPU()
		pd.p:initGPU()
		pd.Ap_X:initGPU()
		pd.preconditioner:initGPU()
		pd.rDotzOld:initGPU()
		
		--TODO make this exclusively GPU
		C.cudaMallocManaged([&&opaque](&(pd.scanAlpha)), sizeof(float), C.cudaMemAttachGlobal)
		C.cudaMallocManaged([&&opaque](&(pd.scanBeta)), sizeof(float), C.cudaMemAttachGlobal)
		return &pd.plan
	end
	return makePlan
end





solversGPU.gradientDescentGPU = function(problemSpec, vars)

	local struct PlanData(S.Object) {
		plan : opt.Plan
		images : vars.PlanImages
		scratchF : &float
		
		gradStore : vars.unknownType

		timer : Timer
	}
	
	local specializedKernels = {}
	specializedKernels.updatePosition = function(data)
		local terra updatePositionGPU(pd : &data.PlanData, learningRate : float)
			var w : int, h : int
			if positionForValidLane(pd, "unknown", &w, &h) then
				var delta = -learningRate * pd.gradStore(w, h)
				pd.images.unknown(w, h) = pd.images.unknown(w, h) + delta
			end
		end
		return { kernel = updatePositionGPU, header = noHeader, footer = noFooter, params = {symbol(float)}, mapMemberName = "unknown" }
	end
	
	local gpu = util.makeGPUFunctions(problemSpec, vars, PlanData, specializedKernels)
	
	local terra impl(data_ : &opaque, images : &&opaque, params_ : &opaque)
		var pd = [&PlanData](data_)
		pd.timer:init()

		var params = [&double](params_)

		unpackstruct(pd.images) = [util.getImages(PlanData, images)]

		-- TODO: parameterize these
		var initialLearningRate = 0.01
		var maxIters = 5000
		var tolerance = 1e-10

		-- Fixed constants (these do not need to be parameterized)
		var learningLoss = 0.8
		var learningGain = 1.1
		var minLearningRate = 1e-25

		var learningRate = initialLearningRate
		
		for iter = 0, maxIters do

			var startCost = gpu.computeCost(pd, pd.images.unknown)
			logSolver("iteration %d, cost=%f, learningRate=%f\n", iter, startCost, learningRate)
			
			gpu.computeGradient(pd, pd.gradStore)
			
			--
			-- move along the gradient by learningRate
			--
			gpu.updatePosition(pd, learningRate)
			
			--
			-- update the learningRate
			--
			var endCost = gpu.computeCost(pd, pd.images.unknown)
			if endCost < startCost then
				learningRate = learningRate * learningGain
			else
				learningRate = learningRate * learningLoss

				if learningRate < minLearningRate then
					break
				end
			end
			pd.timer:nextIteration()
		end
		pd.timer:evaluate()
		pd.timer:cleanup()
	end

	local terra makePlan() : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl

		pd.gradStore:initGPU()
		var err = C.cudaMallocManaged([&&opaque](&(pd.scratchF)), sizeof(float), C.cudaMemAttachGlobal)
		if err ~= 0 then C.printf("cudaMallocManaged error: %d", err) end

		return &pd.plan
	end
	return makePlan
end

solversGPU.conjugateGradientGPU = function(problemSpec, vars)

	local struct PlanData(S.Object) {
		plan : opt.Plan
		images : vars.PlanImages
		scratchF : &float
		
		scratchCost : &float
		scratchNum : &float
		scratchDen : &float
		
		valueStore : vars.unknownType
		
		gradient : vars.unknownType
		prevGradient : vars.unknownType
		
		searchDirection : vars.unknownType

		residualStorage : vars.unknownType
		
		timer : Timer
		timerWall : Timer
	}
	
	local specializedKernels = {}
	
	--
	-- Polak-Ribiere conjugacy
	-- 
	specializedKernels.PRConj = function(data)
		local terra PRConj(pd : &data.PlanData)
			var cost = 0.0f
			var num = 0.0f
			var den = 0.0f
			var w : int, h : int
			if positionForValidLane(pd, "unknown", &w, &h) then
				cost = data.problemSpec.cost.boundary(w, h, unpackstruct(pd.images))
				
				var g = data.problemSpec.gradient.boundary(w, h, unpackstruct(pd.images))
				pd.gradient(w, h) = g
				
				var p = pd.prevGradient(w, h)
				
				num = (-g * (-g + p))
				den = p * p
			end
			
			cost = util.warpReduce(cost)
			num = util.warpReduce(num)
			den = util.warpReduce(den)
			if util.laneid() == 0 then
				util.atomicAdd(pd.scratchCost, cost)
				util.atomicAdd(pd.scratchNum, num)
				util.atomicAdd(pd.scratchDen, den)
			end
			
		end
		return { kernel = PRConj, header = noHeader, footer = noFooter, params = {}, mapMemberName = "unknown" }
	end
	
	specializedKernels.CGDirection = function(data)
		local terra CGDirection(pd : &data.PlanData, beta : float)
			var w : int, h : int
			if positionForValidLane(pd, "unknown", &w, &h) then
				var g = pd.gradient(w, h)
				var addr = &pd.searchDirection(w, h)
				@addr = beta * @addr - g
				pd.prevGradient(w, h) = g
			end
		end
		return { kernel = CGDirection, header = noHeader, footer = noFooter, params = {symbol(float)}, mapMemberName = "unknown" }
	end
	
	local gpu = util.makeGPUFunctions(problemSpec, vars, PlanData, specializedKernels)
	
	local terra impl(data_ : &opaque, images : &&opaque, params_ : &opaque)
		var pd = [&PlanData](data_)
		pd.timerWall:init()
		pd.timer:init()

		var timingInfo : util.TimingInfo
		C.cudaEventCreate(&timingInfo.startEvent)
		C.cudaEventCreate(&timingInfo.endEvent)
		C.cudaEventRecord(timingInfo.startEvent, nil)
		timingInfo.eventName = "wallClock"
		
		var params = [&double](params_)

		unpackstruct(pd.images) = [util.getImages(PlanData, images)]

		var maxIters = 1000
		
		var prevBestAlpha = 0.0f

		for iter = 0, maxIters do

			var curCost = 0.0f
			
			--
			-- compute the search direction
			--
			var beta = 0.0f
			if iter == 0 then
				curCost = gpu.computeCost(pd, pd.images.unknown)
				gpu.computeGradient(pd, pd.gradient)
				gpu.copyImageScale(pd, pd.searchDirection, pd.gradient, -1.0f)
				gpu.copyImage(pd, pd.prevGradient, pd.gradient)
			else
				-- cuda memset this instead
				@pd.scratchCost = 0.0f
				@pd.scratchNum = 0.0f
				@pd.scratchDen = 0.0f
				
				gpu.PRConj(pd)
				
				curCost = @pd.scratchCost
				
				-- ask zach about beta stack vs. beta dynamic
				beta = util.max(@pd.scratchNum / @pd.scratchDen, 0.0f)
				
				gpu.CGDirection(pd, beta)
			end
			
			var bestAlpha = gpu.lineSearchQuadraticFallback(pd, pd.images.unknown, pd.residualStorage, curCost, pd.searchDirection, pd.valueStore, prevBestAlpha)
			
			gpu.addImage(pd, pd.images.unknown, pd.searchDirection, bestAlpha)
			
			prevBestAlpha = bestAlpha
			
			logSolver("iteration %d, cost=%f\n", iter, curCost)
			logSolver("alpha=%12.12f, beta=%12.12f\n\n", bestAlpha, beta)
			
			if bestAlpha == 0.0 and beta == 0.0 then
				break
			end
			
			pd.timer:nextIteration()
		end
		
		C.cudaEventRecord(timingInfo.endEvent, nil)
		pd.timerWall.timingInfo:insert(timingInfo)
		
		pd.timerWall:evaluate()
		pd.timer:evaluate()
		
		pd.timerWall:cleanup()
		pd.timer:cleanup()
	end

	local terra makePlan() : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl

		pd.valueStore:initGPU()
		pd.gradient:initGPU()
		pd.prevGradient:initGPU()
		pd.searchDirection:initGPU()
		pd.residualStorage:initGPU()
		
		C.cudaMallocManaged([&&opaque](&(pd.scratchF)), sizeof(float), C.cudaMemAttachGlobal)
		C.cudaMallocManaged([&&opaque](&(pd.scratchCost)), sizeof(float), C.cudaMemAttachGlobal)
		C.cudaMallocManaged([&&opaque](&(pd.scratchNum)), sizeof(float), C.cudaMemAttachGlobal)
		C.cudaMallocManaged([&&opaque](&(pd.scratchDen)), sizeof(float), C.cudaMemAttachGlobal)

		return &pd.plan
	end
	return makePlan
end

-- http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
solversGPU.adaDeltaGPU = function(problemSpec, vars)

	local momentum = 0.95
	local epsilon = 0.01
	local annealingA = 1.0
	local annealingB = 0.7
	local annealingCutoff = 100
	local struct PlanData(S.Object) {
		plan : opt.Plan
		images : vars.PlanImages
		scratchF : &float
		
		gradient : vars.unknownType
		Eg2 : vars.unknownType
		Ex2 : vars.unknownType
		xNext : vars.unknownType

		timer : Timer
	}
	
	local specializedKernels = {}
	
	specializedKernels.updatePositionA = function(data)
		local terra updatePosition(pd : &data.PlanData)
			var w : int, h : int
			if positionForValidLane(pd, "unknown", &w, &h) then
				var Eg2 = 0.0f
				var Ex2 = 0.0f
				for i = 0, 10 do
					var g = data.problemSpec.gradient.boundary(w, h, pd.images.unknown, unpackstruct(pd.images, 2))
					Eg2 = momentum * Eg2 + (1.0f - momentum) * g * g
					var learningRate = -annealingA * gpuMath.sqrt((Ex2 + epsilon) / (Eg2 + epsilon))
					var delta = learningRate * g
					Ex2 = momentum * Ex2 + (1.0f - momentum) * delta * delta
					pd.images.unknown(w, h) = pd.images.unknown(w, h) + delta
					--pd.xNext(w, h) = pd.images.unknown(w, h) + delta
				end
			end
		end
		return { kernel = updatePosition, header = noHeader, footer = noFooter, params = {}, mapMemberName = "unknown" }
	end
	
	specializedKernels.updatePositionB = function(data)
		local terra updatePosition(pd : &data.PlanData)
			var w : int, h : int
			if positionForValidLane(pd, "unknown", &w, &h) then
				var g = data.problemSpec.gradient.boundary(w, h, pd.images.unknown, unpackstruct(pd.images, 2))
				var Eg2val = momentum * pd.Eg2(w, h) + (1.0f - momentum) * g * g
				pd.Eg2(w, h) = Eg2val
				var Ex2val = pd.Ex2(w, h)
				var learningRate = -annealingB * gpuMath.sqrt((Ex2val + epsilon) / (Eg2val + epsilon))
				var delta = learningRate * g
				--var delta = -0.01 * g
				pd.Ex2(w, h) = momentum * Ex2val + (1.0f - momentum) * delta * delta
				pd.images.unknown(w, h) = pd.images.unknown(w, h) + delta
				--pd.xNext(w, h) = pd.images.unknown(w, h) + delta
			end
		end
		return { kernel = updatePosition, header = noHeader, footer = noFooter, params = {}, mapMemberName = "unknown" }
	end
	
	local gpu = util.makeGPUFunctions(problemSpec, vars, PlanData, specializedKernels)
	
	local terra impl(data_ : &opaque, images : &&opaque, params_ : &opaque)
		var pd = [&PlanData](data_)
		pd.timer:init()

		var params = [&double](params_)

		unpackstruct(pd.images) = [util.getImages(PlanData, images)]

		-- TODO: parameterize these
		var maxIters = 10000
		var tolerance = 1e-10
		
		var file = C.fopen("C:/code/run.txt", "wb")

		for iter = 0, maxIters do

			var startCost = gpu.computeCost(pd, pd.images.unknown)
			logSolver("iteration %d, cost=%f\n", iter, startCost)
			C.fprintf(file, "%d\t%15.15f\n", iter, startCost)
			
			if iter < annealingCutoff then
				gpu.updatePositionA(pd)
			else
				gpu.updatePositionB(pd)
			end
			
			--gpu.copyImage(pd, pd.images.unknown, pd.xNext)
			
			if iter == 2000 then
				--gpu.copyImageScale(pd, pd.Eg2, pd.Eg2, 0.0f)
				--gpu.copyImageScale(pd, pd.Ex2, pd.Ex2, 0.0f)
			end
			
			pd.timer:nextIteration()
		end
		
		C.fclose(file)
		
		pd.timer:evaluate()
		pd.timer:cleanup()
	end

	local terra makePlan() : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl

		pd.gradient:initGPU()
		pd.Eg2:initGPU()
		pd.Ex2:initGPU()
		pd.xNext:initGPU()
		
		C.cudaMallocManaged([&&opaque](&(pd.scratchF)), sizeof(float), C.cudaMemAttachGlobal)

		return &pd.plan
	end
	return makePlan
end

-- vector-free L-BFGS using two-loop recursion: http://papers.nips.cc/paper/5333-large-scale-l-bfgs-using-mapreduce.pdf
solversGPU.vlbfgsGPU = function(problemSpec, vars)

	local maxIters = 1000
	local m = 3
	local b = 2 * m + 1
	
	local bDim = opt.InternalDim("b", b)
	local dpmType = opt.InternalImage(float, bDim, bDim)
	
	local struct GPUStore {
		-- These all live on the CPU!
		dotProductMatrix : dpmType
		dotProductMatrixStorage : dpmType
		alphaList : opt.InternalImage(float, bDim, 1)
		imageList : vars.unknownType[b]
		coefficients : float[b]
	}

	-- TODO: alphaList must be a custom image!
	local struct PlanData(S.Object) {
		plan : opt.Plan
		images : vars.PlanImages
		scratchF : &float
		
		gradient : vars.unknownType
		prevGradient : vars.unknownType

		p : vars.unknownType
		
		timer : Timer

		sList : vars.unknownType[m]
		yList : vars.unknownType[m]
		
		-- variables used for line search
		currentValues : vars.unknownType
		currentResiduals : vars.unknownType
		
		gpuStore : GPUStore
	}
		
	local terra imageFromIndex(pd : &PlanData, index : int)
		if index < m then
			return pd.sList[index]
		elseif index < 2 * m then
			return pd.yList[index - m]
		else
			return pd.gradient
		end
	end
	
	local terra nextCoefficientIndex(index : int)
		if index == m - 1 or index == 2 * m - 1 or index == 2 * m then
			return -1
		end
		return index + 1
	end
	
	local makeDotProductPairs = function()
		local pairs = terralib.newlist()
		local insertAllPairs = function(j)
			for i = 0, b do
				pairs:insert( {j, i} )
			end
		end
		
		insertAllPairs(m - 1)
		insertAllPairs(2 * m - 1)
		insertAllPairs(2 * m)
		
		-- TODO: computing 3 unnecessary dot products
		return pairs
	end

	local terra atomicReduce(a : float, b : &float) -- NYI
	end

	--[[local function makeDotProducts(dps, nImages, imageType)
		local nDotProducts = #dps
		local localOut = util.symTable(float, nDotProducts, "localOut")
		local es = util.symTable(float, nImages, "e")
		local terra outKernel(input : (&float)[nImages], out : dpmType, N : int)
			var I = util.ceilingDivide(N, blockDim.x * gridDim.x)
			escape
				for i,l in ipairs(localOut) do
					emit quote var [l] = 0.f end
				end 
			end
			for i = 0,I do
				var idx = blockIdx.x*blockDim.x*I + blockDim.x*i + threadIdx.x
				if idx < N then
					escape
						for i,e in ipairs(es) do
							emit quote var [e] = input[ [i-1] ][idx] end
						end
						for i,dp in ipairs(dps) do
							--print(dp[1],dp[2],unpack(es))
							emit quote
								[localOut[i] ] = [localOut[i] ] + [es[dp[1] + 1] ] * [es[dp[2] + 1] ]
							end
						end
					end
				end
			end
			escape
				for i,dp in ipairs(dps) do
					emit quote atomicReduce([localOut[i] ],&out([dp[1] ], [dp[2] ])) end
				end
			end
		end
		return outKernel
	end

	local test = { {1,1}, {2,1}, {1,3} }
	local r = makeDotProducts(test,3)
	r:printpretty()]]

	local specializedKernels = {}
	
	local gpu = util.makeGPUFunctions(problemSpec, vars, PlanData, {})
	local cpu = util.makeCPUFunctions(problemSpec, vars, PlanData)
	
	local dotPairs = makeDotProductPairs()
		
	local terra impl(data_ : &opaque, images : &&opaque, params_ : &opaque)
		
		var pd = [&PlanData](data_)
		var params = [&double](params_)
		pd.timer:init()
		unpackstruct(pd.images) = [util.getImages(PlanData, images)]

		var k = 0
		
		-- using an initial guess of alpha means that it will invoke quadratic optimization on the first iteration,
		-- which is only sometimes a good idea.
		var prevBestAlpha = 0.0

		gpu.computeGradient(pd, pd.gradient)

		for iter = 0, maxIters - 1 do
		
			var iterStartCost = gpu.computeCost(pd, pd.images.unknown)
			
			logSolver("iteration %d, cost=%f\n", iter, iterStartCost)
			
			--
			-- compute the search direction p
			--
			if k == 0 then
				gpu.copyImageScale(pd, pd.p, pd.gradient, -1.0f)
			else
				-- note that much of this happens on the CPU!
				
				for i = 0, b do
					pd.gpuStore.imageList[i] = imageFromIndex(pd, i)
				end
				
				-- compute the top half of the dot product matrix
				--cpu.copyImage(pd.gpuStore.dotProductMatrixStorage, pd.gpuStore.dotProductMatrix)
				for i = 0, b do
					for j = 0, b do
						pd.gpuStore.dotProductMatrixStorage(i, j) = pd.gpuStore.dotProductMatrix(i, j)
					end
				end
				
				for i = 0, b do
					for j = i, b do
						var prevI = nextCoefficientIndex(i)
						var prevJ = nextCoefficientIndex(j)
						if prevI == -1 or prevJ == -1 then
							pd.gpuStore.dotProductMatrix(i, j) = gpu.innerProduct(pd, pd.gpuStore.imageList[i], pd.gpuStore.imageList[j])
							--C.printf("%d dot %d\n", i, j)
						else
							pd.gpuStore.dotProductMatrix(i, j) = pd.gpuStore.dotProductMatrixStorage(prevI, prevJ)
						end
					end
				end
				
				-- compute the bottom half of the dot product matrix
				for i = 1, b do
					for j = 0, i - 1 do
						pd.gpuStore.dotProductMatrix(i, j) = pd.gpuStore.dotProductMatrix(j, i)
					end
				end
			
				for i = 0, 2 * m do pd.gpuStore.coefficients[i] = 0.0 end
				pd.gpuStore.coefficients[2 * m] = -1.0
				
				for i = k - 1, k - m - 1, -1 do
					if i < 0 then break end
					var j = i - (k - m)
					
					var num = 0.0
					for q = 0, b do
						num = num + pd.gpuStore.coefficients[q] * pd.gpuStore.dotProductMatrix(q, j)
					end
					var den = pd.gpuStore.dotProductMatrix(j, j + m)
					pd.gpuStore.alphaList(i, 0) = num / den
					pd.gpuStore.coefficients[j + m] = pd.gpuStore.coefficients[j + m] - pd.gpuStore.alphaList(i, 0)
				end
				
				var scale = pd.gpuStore.dotProductMatrix(m - 1, 2 * m - 1) / pd.gpuStore.dotProductMatrix(2 * m - 1, 2 * m - 1)
				for i = 0, b do
					pd.gpuStore.coefficients[i] = pd.gpuStore.coefficients[i] * scale
				end
				
				for i = k - m, k do
					if i >= 0 then
						var j = i - (k - m)
						var num = 0.0
						for q = 0, b do
							num = num + pd.gpuStore.coefficients[q] * pd.gpuStore.dotProductMatrix(q, m + j)
						end
						var den = pd.gpuStore.dotProductMatrix(j, j + m)
						var beta = num / den
						pd.gpuStore.coefficients[j] = pd.gpuStore.coefficients[j] + (pd.gpuStore.alphaList(i, 0) - beta)
					end
				end
				
				-- reconstruct p from basis vectors
				gpu.copyImageScale(pd, pd.p, pd.p, 0.0f)
				for i = 0, b do
					var image = imageFromIndex(pd, i)
					var coefficient = pd.gpuStore.coefficients[i]
					gpu.addImage(pd, pd.p, image, coefficient)
				end
			end
			
			--
			-- line search
			--
			gpu.copyImage(pd, pd.currentValues, pd.images.unknown)
			--gpu.computeResiduals(pd, pd.currentResiduals, pd.currentValues)
			
			var bestAlpha = gpu.lineSearchQuadraticFallback(pd, pd.currentValues, pd.currentResiduals, iterStartCost, pd.p, pd.images.unknown, prevBestAlpha)
			
			-- cycle the oldest s and y
			var yListStore = pd.yList[0]
			var sListStore = pd.sList[0]
			for i = 0, m - 1 do
				pd.yList[i] = pd.yList[i + 1]
				pd.sList[i] = pd.sList[i + 1]
			end
			pd.yList[m - 1] = yListStore
			pd.sList[m - 1] = sListStore
			
			-- compute new x and s
			gpu.copyImageScale(pd, pd.sList[m - 1], pd.p, bestAlpha)
			gpu.combineImage(pd, pd.images.unknown, pd.currentValues, pd.sList[m - 1], 1.0f)
			
			gpu.copyImage(pd, pd.prevGradient, pd.gradient)
			
			gpu.computeGradient(pd, pd.gradient)
			
			-- compute new y
			gpu.combineImage(pd, pd.yList[m - 1], pd.gradient, pd.prevGradient, -1.0f)
			
			prevBestAlpha = bestAlpha
		
			
			k = k + 1
			
			logSolver("alpha=%12.12f\n\n", bestAlpha)
			if bestAlpha == 0.0 then
				break
			end
		end
		pd.timer:evaluate()
		pd.timer:cleanup()
	end
	
	local terra makePlan() : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl

		pd.gradient:initGPU()
		pd.prevGradient:initGPU()
		
		pd.currentValues:initGPU()
		pd.currentResiduals:initGPU()
		
		pd.p:initGPU()
		
		for i = 0, m do
			pd.sList[i]:initGPU()
			pd.yList[i]:initGPU()
		end
		
		C.cudaMallocManaged([&&opaque](&(pd.scratchF)), sizeof(float), C.cudaMemAttachGlobal)
		
		-- CPU!
		pd.gpuStore.dotProductMatrix:initCPU()
		pd.gpuStore.dotProductMatrixStorage:initCPU()
		pd.gpuStore.alphaList:initCPU()
		--pd.alphaList:initCPU(maxIters, 1)
		

		return &pd.plan
	end
	return makePlan
end

return solversGPU