
local S = require("std")
local util = require("util")
local C = util.C
local Timer = util.Timer
local positionForValidLane = util.positionForValidLane

local gpuMath = util.gpuMath

local function noHeader(pd)
	return quote end
end

local function noFooter(pd)
	return quote end
end

return function(problemSpec, vars)

	local struct PlanData(S.Object) {
		plan : opt.Plan
		parameters : problemSpec:ParameterType(false)	--get the non-blocked version
		scratchF : &float
		
		gradStore : problemSpec:UnknownType()

		timer : Timer
	}
	
	local kernels = {}
	kernels.updatePosition = function(data)
		local terra updatePositionGPU(pd : &data.PlanData, learningRate : float)
			var w : int, h : int
			if positionForValidLane(pd, "X", &w, &h) then
				var delta = -learningRate * pd.gradStore(w, h)
				pd.parameters.X(w, h) = pd.parameters.X(w, h) + delta
			end
		end
		return { kernel = updatePositionGPU, header = noHeader, footer = noFooter, params = {symbol(float)}, mapMemberName = "X" }
	end
	
	kernels.computeGradient = function(data)
		local terra computeGradientGPU(pd : &data.PlanData,  gradientOut : data.imageType)
			var w : int, h : int
			if positionForValidLane(pd, "X", &w, &h) then
				gradientOut(w, h) = data.problemSpec.functions.gradient.boundary(w, h, w, h, pd.parameters)
			end
		end
		return { kernel = computeGradientGPU, header = noHeader, footer = noFooter, params = {symbol(data.imageType)}, mapMemberName = "X" }
	end
	
	
	local gpu = util.makeGPUFunctions(problemSpec, vars, PlanData, kernels)
	
	local terra impl(data_ : &opaque, images : &&opaque, edgeValues : &&opaque, params_ : &&opaque)
		var pd = [&PlanData](data_)
		pd.timer:init()

		var params = [&double](params_)

		--unpackstruct(pd.images) = [util.getImages(PlanData, images)]
		pd.parameters = [util.getParameters(problemSpec, images, edgeValues,params_)]

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

			var startCost = gpu.computeCost(pd, pd.parameters.X)
			logSolver("iteration %d, cost=%f, learningRate=%f\n", iter, startCost, learningRate)

			gpu.computeGradient(pd, pd.gradStore)
			--
			-- move along the gradient by learningRate
			--
			gpu.updatePosition(pd, learningRate)

			
			--
			-- update the learningRate
			--
			var endCost = gpu.computeCost(pd, pd.parameters.X)
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
