local S = require("std")
local util = require("util")
local dbg = require("dbg")
local C = util.C
local Timer = util.Timer

local getValidUnknown = util.getValidUnknown

local gpuMath = util.gpuMath

opt.BLOCK_SIZE = 16
local BLOCK_SIZE =  opt.BLOCK_SIZE

local FLOAT_EPSILON = `0.000001f 
-- GAUSS NEWTON (non-block version)
return function(problemSpec)

	local unknownElement = problemSpec:UnknownType().metamethods.typ
	local unknownType = problemSpec:UnknownType()
    local isGraph = #problemSpec.functions.cost.graphfunctions > 0
    local struct PlanData(S.Object) {
		plan : opt.Plan
		parameters : problemSpec:ParameterType()
		scratchF : &float
		debugDumpImage : &float
		debugCostImage : &float
		debugJTJImage : &float
		debugJTFImage : &float
		debugPreImage : &float


		delta : problemSpec:UnknownType()	--current linear update to be computed -> num vars
		r : problemSpec:UnknownType()		--residuals -> num vars	--TODO this needs to be a 'residual type'
		z : problemSpec:UnknownType()		--preconditioned residuals -> num vars	--TODO this needs to be a 'residual type'
		p : problemSpec:UnknownType()		--descent direction -> num vars
		Ap_X : problemSpec:UnknownType()	--cache values for next kernel call after A = J^T x J x p -> num vars
		preconditioner : problemSpec:UnknownType() --pre-conditioner for linear system -> num vars
		
		
		scanAlphaNumerator : &float
		scanAlphaDenominator : &float
		scanBetaNumerator : &float
		scanBetaDenominator : &float
		
		timer : Timer
		endSolver : util.TimerEvent
		nIter : int				--current non-linear iter counter
		nIterations : int		--non-linear iterations
		lIterations : int		--linear iterations
	}
	
	local guardedInvert = macro(function(p)
	    local pt = p:gettype()
	    if util.isvectortype(pt) then
	        return quote
	                    var invp = p
	                    for i = 0, invp:size() do
	                        invp(i) = terralib.select(invp(i) > FLOAT_EPSILON, 1.f / invp(i),invp(i))
	                    end
	               in invp end
	    else
	        return `terralib.select(p > FLOAT_EPSILON, 1.f / p, p)
	    end
	end)

    local kernels = {}
    terra kernels.PCGInit1(pd : PlanData)
        var d = 0.0f -- init for out of bounds lanes
        var w : int, h : int
        if getValidUnknown(pd, &w, &h) then
		
            -- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0                            
            var residuum : unknownElement = 0.0f
            var pre : unknownElement = 0.0f	
			
            if (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then 
                
				pd.delta(w,h) = 0.0f    
				
                residuum, pre = problemSpec.functions.evalJTF.unknownfunction(w, h, w, h, pd.parameters)
                residuum = -residuum
                pd.r(w, h) = residuum
				
				if not problemSpec.usepreconditioner then
					pre = 1.0f
				end
            end        
            
			if not isGraph then				
				pre = guardedInvert(pre)
				var p = pre*residuum	-- apply pre-conditioner M^-1			   
				pd.p(w, h) = p
				
				--d = residuum*p		-- x-th term of nominator for computing alpha and denominator for computing beta
				d = util.Dot(residuum,p) 
			end
			
			pd.preconditioner(w, h) = pre
			
        end 
		if not isGraph then
			d = util.warpReduce(d)	
			if (util.laneid() == 0) then				
				util.atomicAdd(pd.scanAlphaNumerator, d)
			end
		end
    end
    
    terra kernels.PCGInit1_Graph(pd : PlanData)
		var tIdx = 0 	
		escape 
	    	for i,func in ipairs(problemSpec.functions.evalJTF.graphfunctions) do
				local name,implementation = func.graphname,func.implementation
				emit quote 
	    			if util.getValidGraphElement(pd,[name],&tIdx) then
						implementation(tIdx, pd.parameters, pd.r, pd.preconditioner)
	    			end 
				end
    		end
    	end
    end

    terra kernels.PCGInit1_Finish(pd : PlanData)	--only called for graphs
    	var d = 0.0f -- init for out of bounds lanes
        var w : int, h : int
        if getValidUnknown(pd, &w, &h) then
        	var residuum = pd.r(w, h)			
			var pre = pd.preconditioner(w, h)
			
			pre = guardedInvert(pre)
			
			if not problemSpec.usepreconditioner then
				pre = 1.0f
			end
			
			var p = pre*residuum	-- apply pre-conditioner M^-1			   
			pd.p(w, h) = p
        	d = util.Dot(residuum, p)
        end

		d = util.warpReduce(d)	
        if (util.laneid() == 0) then
            util.atomicAdd(pd.scanAlphaNumerator, d)
        end
	end
	
    terra kernels.PCGStep1(pd : PlanData)
        var d = 0.0f
        var w : int, h : int
        if getValidUnknown(pd, &w, &h) and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then
            var tmp : unknownElement = 0.0f
             -- A x p_k  => J^T x J x p_k 
            tmp = problemSpec.functions.applyJTJ.unknownfunction(w, h, w, h, pd.parameters, pd.p)
            pd.Ap_X(w, h) = tmp					 -- store for next kernel call
            d = util.Dot(pd.p(w,h),tmp)			 -- x-th term of denominator of alpha
        end
        d = util.warpReduce(d)
        if (util.laneid() == 0) then
            util.atomicAdd(pd.scanAlphaDenominator, d)
        end
    end
	
	terra kernels.PCGStep1_Graph(pd : PlanData)
		var d = 0.0f
		var tIdx = 0 	
        escape 
			for i,func in ipairs(problemSpec.functions.applyJTJ.graphfunctions) do
				local name,implementation = func.graphname,func.implementation
				emit quote 
				    if util.getValidGraphElement(pd,[name],&tIdx) then
				        d = d + implementation(tIdx, pd.parameters, pd.p, pd.Ap_X)
				    end 
				end
			end
		end
		d = util.warpReduce(d)
        if (util.laneid() == 0) then
            util.atomicAdd(pd.scanAlphaDenominator, d)
        end
    end

	
	terra kernels.PCGStep2(pd : PlanData)
        var b = 0.0f 
        var w : int, h : int
        if getValidUnknown(pd, &w, &h)  and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then
            -- sum over block results to compute denominator of alpha
            var alphaDenominator : float = pd.scanAlphaDenominator[0]
			var alphaNumerator : float = pd.scanAlphaNumerator[0]
                        
            -- update step size alpha
			var alpha = 0.0f
            if alphaDenominator > FLOAT_EPSILON then 
				alpha = alphaNumerator/alphaDenominator 
			end 
        
            pd.delta(w, h) = pd.delta(w, h)+alpha*pd.p(w,h)		-- do a decent step
            
            var r = pd.r(w,h)-alpha*pd.Ap_X(w,h)				-- update residuum
            pd.r(w,h) = r										-- store for next kernel call
        
			var pre = pd.preconditioner(w,h)
			if not problemSpec.usepreconditioner then
				pre = 1.0f
			end
			
			if isGraph then
				pre = guardedInvert(pre)
			end
			
            var z = pre*r										-- apply pre-conditioner M^-1
            pd.z(w,h) = z;										-- save for next kernel call
            
            b = util.Dot(z,r)									-- compute x-th term of the numerator of beta
        end
        b = util.warpReduce(b)
        if (util.laneid() == 0) then
            util.atomicAdd(pd.scanBetaNumerator, b)
        end
    end
	
    terra kernels.PCGStep3(pd : PlanData)			
        var w : int, h : int
		
        if getValidUnknown(pd, &w, &h) and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then
			
			var rDotzNew : float = pd.scanBetaNumerator[0]				-- get new numerator
			var rDotzOld : float = pd.scanAlphaNumerator[0]				-- get old denominator

			var beta : float = 0.0f		                    
			if rDotzOld > FLOAT_EPSILON then beta = rDotzNew/rDotzOld end	-- update step size beta
			pd.p(w,h) = pd.z(w,h)+beta*pd.p(w,h)							-- update decent direction

        end
    end
	
	terra kernels.PCGLinearUpdate(pd : PlanData)
        var w : int, h : int
        if getValidUnknown(pd, &w, &h) and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then
            pd.parameters.X(w,h) = pd.parameters.X(w,h) + pd.delta(w,h)
        end
    end	
	
	terra kernels.computeCost(pd : PlanData)			
        var cost : float = 0.0f
        var w : int, h : int
        if util.getValidUnknown(pd, &w, &h)  and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then
            var params = pd.parameters				
            cost = cost + [float](problemSpec.functions.cost.unknownfunction(w, h, w, h, params))
        end

        cost = util.warpReduce(cost)
        if (util.laneid() == 0) then
        util.atomicAdd(pd.scratchF, cost)
        end
    end
	
	terra kernels.computeCost_Graph(pd : PlanData)			
        var cost : float = 0.0f
		
		var tIdx = 0 	
        escape 
			for i,func in ipairs(problemSpec.functions.cost.graphfunctions) do
				local name,implementation = func.graphname,func.implementation
				emit quote 
				    if util.getValidGraphElement(pd,[name],&tIdx) then
				        cost = cost + implementation(tIdx, pd.parameters)
				    end 
				end
			end
		end
		
        cost = util.warpReduce(cost)
        if (util.laneid() == 0) then
            util.atomicAdd(pd.scratchF, cost)
        end
    end
    
    terra kernels.precomputeImages(pd : PlanData)
        var w : int, h : int
        if util.getValidUnknown(pd,&w,&h) then
            escape
                if problemSpec.functions.precompute then
                    emit quote 
                        problemSpec.functions.precompute.unknownfunction(w,h,w,h,pd.parameters)
                    end
                end
            end
        end
    end
    
	local gpu = util.makeGPUFunctions(problemSpec, PlanData, kernels)

	local terra init(data_ : &opaque, params_ : &&opaque, solverparams : &&opaque)
	   var pd = [&PlanData](data_)
	   pd.timer:init()
	   pd.timer:startEvent("overall",nil,&pd.endSolver)
       [util.initParameters(`pd.parameters,problemSpec,params_,true)]
	   pd.nIter = 0
	   pd.nIterations = @[&int](solverparams[0])
	   pd.lIterations = @[&int](solverparams[1])
	end
    local terra computeCost(pd : &PlanData) : float
        C.cudaMemset(pd.scratchF, 0, sizeof(float))
        gpu.computeCost(pd)
        gpu.computeCost_Graph(pd)
        var f : float
        C.cudaMemcpy(&f, pd.scratchF, sizeof(float), C.cudaMemcpyDeviceToHost)
        return f
    end

	local terra step(data_ : &opaque, params_ : &&opaque, solverparams : &&opaque)
		var pd = [&PlanData](data_)
		[util.initParameters(`pd.parameters,problemSpec, params_,false)]
        gpu.precomputeImages(pd)    
		if pd.nIter < pd.nIterations then
		    var startCost = computeCost(pd)
			logSolver("iteration %d, cost=%f\n", pd.nIter, startCost)

			C.cudaMemset(pd.scanAlphaNumerator, 0, sizeof(float))	--scan in PCGInit1 requires reset
			C.cudaMemset(pd.scanAlphaDenominator, 0, sizeof(float))	--scan in PCGInit1 requires reset
			C.cudaMemset(pd.scanBetaNumerator, 0, sizeof(float))	--scan in PCGInit1 requires reset
			C.cudaMemset(pd.scanBetaDenominator, 0, sizeof(float))	--scan in PCGInit1 requires reset
			
			gpu.PCGInit1(pd)
			
			if isGraph then
				gpu.PCGInit1_Graph(pd)	
				gpu.PCGInit1_Finish(pd)	
			end

			for lIter = 0, pd.lIterations do				

                C.cudaMemset(pd.scanAlphaDenominator, 0, sizeof(float))
				
				gpu.PCGStep1(pd)

				if isGraph then
					gpu.PCGStep1_Graph(pd)	
				end
				
				C.cudaMemset(pd.scanBetaNumerator, 0, sizeof(float))
				
				gpu.PCGStep2(pd)
				gpu.PCGStep3(pd)

				-- save new rDotz for next iteration
				C.cudaMemcpy(pd.scanAlphaNumerator, pd.scanBetaNumerator, sizeof(float), C.cudaMemcpyDeviceToDevice)	
				
			end
			
			gpu.PCGLinearUpdate(pd)
		    pd.nIter = pd.nIter + 1
		    return 1
		else
			var finalCost = computeCost(pd)
			logSolver("final cost=%f\n", finalCost)
		    pd.timer:endEvent(nil,pd.endSolver)
		    pd.timer:evaluate()
		    pd.timer:cleanup()
		    return 0
		end
	end

	local terra makePlan() : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.init,pd.plan.step = init,step

		pd.delta:initGPU()
		pd.r:initGPU()
		pd.z:initGPU()
		pd.p:initGPU()
		pd.Ap_X:initGPU()
		pd.preconditioner:initGPU()

		[util.initPrecomputedImages(`pd.parameters,problemSpec)]
		
		C.cudaMalloc([&&opaque](&(pd.scanAlphaNumerator)), sizeof(float))
		C.cudaMalloc([&&opaque](&(pd.scanBetaNumerator)), sizeof(float))
		C.cudaMalloc([&&opaque](&(pd.scanAlphaDenominator)), sizeof(float))
		C.cudaMalloc([&&opaque](&(pd.scanBetaDenominator)), sizeof(float))
		
		C.cudaMalloc([&&opaque](&(pd.scratchF)), sizeof(float))
		return &pd.plan
	end
	return makePlan
end
