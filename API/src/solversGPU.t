
local S = require("std")
local util = require("util")
local C = util.C
local Timer = util.Timer

solversGPU = {}

local function noHeader(pd)
	return quote end
end

local function noFooter(pd)
	return quote end
end


-- GAUSS NEWTON (non-block version)
solversGPU.gaussNewtonGPU = function(problemSpec, vars)

	local struct PlanData(S.Object) {
		plan : opt.Plan
		images : vars.PlanImages
		scratchF : &float
		
		r : vars.unknownType				--residuals -> num vars	--TODO this needs to be a 'residual type'
		p : vars.unknownType				--decent direction -> num vars
		preconditioner : vars.unknownType	--preconditioner for linear system -> num vars
		rDotZOld : &float					--Old nominator (denominator) of alpha (beta)				
		timer : Timer
		
		--TODO allocate the data in makePlan
	}
	
	local specializedKernels = {}
	specializedKernels.PCGInit1 = function(data)
		local terra PCGInit1GPU(pd : &data.PlanData, w : int, h : int)
			var residuum = -data.problemSpec.gradient.boundary(w, h, unpackstruct(pd.images))	-- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 
			pd.r(w, h) = residuum

			-- TODO pd.precondition(w,h) needs to computed somehow (ideally in the gradient?
			pd.preconditioner(w, h) = data.problemSpec.gradientPreconditioner(w, h)	-- TODO fix this hack... the preconditioner needs to be the diagonal of JTJ
			var p = pd.preconditioner(w, h)*residuum				   -- apply preconditioner M^-1
			pd.p(w, h) = p
		
			var d = residuum*p;										   -- x-th term of nomimator for computing alpha and denominator for computing beta
			
			d = util.warpReduce(d)	--TODO check for sizes != 32
			if (util.laneid() == 0) then
				util.atomicAdd(pd.rDotZOld, d)
			end
		end
		return { kernel = PCGInit1GPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "unknown" }
	end
	

	local gpu = util.makeGPUFunctions(problemSpec, vars, PlanData, specializedKernels)
	
	local terra impl(data_ : &opaque, images : &&opaque, params_ : &opaque)
		var pd = [&PlanData](data_)
		pd.timer:init()

		var params = [&double](params_)

		unpackstruct(pd.images) = [util.getImages(PlanData, images)]

		var maxIters = 5000
		
		for iter = 0, maxIters do
			--init
			@pd.rDotZOld = 0.0
			gpu.PCGInit1(pd)
		end
		--[[
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

			var startCost = gpu.computeCost(pd)
			logSolver("iteration %d, cost=%f, learningRate=%f\n", iter, startCost, learningRate)
			
			
			--
			-- move along the gradient by learningRate
			--
			gpu.updatePosition(pd, learningRate)
			
			--
			-- update the learningRate
			--
			var endCost = gpu.computeCost(pd)
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
		]]--
		pd.timer:evaluate()
		pd.timer:cleanup()
	end

	local terra makePlan() : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl

		--pd.gradStore:initGPU()
		C.cudaMallocManaged([&&opaque](&(pd.rDotZOld)), sizeof(float), C.cudaMemAttachGlobal)

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
		local terra updatePositionGPU(pd : &data.PlanData, w : int, h : int, learningRate : float)
			var delta = -learningRate * pd.gradStore(w, h)
			pd.images.unknown(w, h) = pd.images.unknown(w, h) + delta
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
							C.printf("%d dot %d\n", i, j)
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