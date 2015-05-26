
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
		learningRate : float
		iter : int
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
				--gradientOut(w, h) = data.problemSpec.functions.gradient.boundary(w, h, w, h, pd.parameters)
				gradientOut(w, h) = data.problemSpec.functions.evalJTF.boundary(w, h, w, h, pd.parameters)._0
			end
		end
		return { kernel = computeGradientGPU, header = noHeader, footer = noFooter, params = {symbol(data.imageType)}, mapMemberName = "X" }
	end
	
	
	local gpu = util.makeGPUFunctions(problemSpec, vars, PlanData, kernels)
	
	local learningLoss, learningGain, minLearningRate = .8,1.1,1e-25

    --TODO: parameterize these
    local initialLearningRate, maxIters, tolerance = 0.01, 5000, 1e-10
	
	local terra init(data_ : &opaque, images : &&opaque, edgeValues : &&opaque, params_ : &&opaque, solverparams : &&opaque)
		var pd = [&PlanData](data_)
		pd.timer:init()
		pd.parameters = [util.getParameters(problemSpec, images, edgeValues,params_)]
        pd.learningRate = initialLearningRate
        pd.iter = 0
	end
	
	local terra step(data_ : &opaque, images : &&opaque, edgeValues : &&opaque, params_ : &&opaque, solverparams : &&opaque)
	    var pd = [&PlanData](data_)
	    pd.parameters = [util.getParameters(problemSpec, images, edgeValues,params_)]
	    
	    if pd.iter < maxIters then
	        var startCost = gpu.computeCost(pd, pd.parameters.X)
			logSolver("iteration %d, cost=%f, learningRate=%f\n", pd.iter, startCost, pd.learningRate)
			

			gpu.computeGradient(pd, pd.gradStore)
			--
			-- move along the gradient by learningRate
			--
			gpu.updatePosition(pd, pd.learningRate)
			--
			-- update the learningRate
			--
			var endCost = gpu.computeCost(pd, pd.parameters.X)
			if endCost < startCost then
				pd.learningRate = pd.learningRate * learningGain
			else
				pd.learningRate = pd.learningRate * learningLoss

				if pd.learningRate < minLearningRate then
					goto exit
				end
			end
			pd.iter = pd.iter + 1
			pd.timer:nextIteration()
	        return 1
	    end
	    
	    ::exit::
	    pd.timer:evaluate()
		pd.timer:cleanup()
		return 0
	end

	local terra makePlan() : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.init,pd.plan.step = init,step

		pd.gradStore:initGPU()
		var err = C.cudaMallocManaged([&&opaque](&(pd.scratchF)), sizeof(float), C.cudaMemAttachGlobal)
		if err ~= 0 then C.printf("cudaMallocManaged error: %d", err) end

		return &pd.plan
	end
	return makePlan
end
