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

    local UnknownType = problemSpec:UnknownType()
	local unknownElement = UnknownType.images[1].imagetype:ElementType()
	local TUnknownType = UnknownType:terratype()
	local UnknownIndexSpace = UnknownType:IndexSpaces()[1]
    local Index = UnknownIndexSpace:indextype()
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


		delta : TUnknownType	--current linear update to be computed -> num vars
		r : TUnknownType		--residuals -> num vars	--TODO this needs to be a 'residual type'
		z : TUnknownType		--preconditioned residuals -> num vars	--TODO this needs to be a 'residual type'
		p : TUnknownType		--descent direction -> num vars
		Ap_X : TUnknownType	--cache values for next kernel call after A = J^T x J x p -> num vars
		preconditioner : TUnknownType --pre-conditioner for linear system -> num vars
		
		
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
	
	local terra guardedInvert(p : unknownElement)
	    var invp = p
	    for i = 0, invp:size() do
	        invp(i) = terralib.select(invp(i) > FLOAT_EPSILON, 1.f / invp(i),invp(i))
	    end
	    return invp
	end
	
    local kernels = {}
    terra kernels.PCGInit1(pd : PlanData)
        var d = 0.0f -- init for out of bounds lanes
        
        var idx : Index
        if idx:initFromCUDAParams() then
        
            -- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0                            
            var residuum : unknownElement = 0.0f
            var pre : unknownElement = 0.0f	
			
            if (not [problemSpec:EvalExclude(idx,`pd.parameters)]) then 
                
				pd.delta(idx) = 0.0f    
				
                residuum, pre = problemSpec.functions.evalJTF.unknownfunction(idx, pd.parameters)
                residuum = -residuum
                pd.r(idx) = residuum
				
				if not problemSpec.usepreconditioner then
					pre = 1.0f
				end
            end        
            
			if not isGraph then				
				pre = guardedInvert(pre)
				var p = pre*residuum	-- apply pre-conditioner M^-1			   
				pd.p(idx) = p
				
				--d = residuum*p		-- x-th term of nominator for computing alpha and denominator for computing beta
				d = residuum:dot(p) 
			end
			
			pd.preconditioner(idx) = pre
			
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
        var idx : Index
        if idx:initFromCUDAParams() then
        	var residuum = pd.r(idx)			
			var pre = pd.preconditioner(idx)
			
			pre = guardedInvert(pre)
			
			if not problemSpec.usepreconditioner then
				pre = 1.0f
			end
			
			var p = pre*residuum	-- apply pre-conditioner M^-1			   
			pd.p(idx) = p
        	d = residuum:dot(p)
        end

		d = util.warpReduce(d)	
        if (util.laneid() == 0) then
            util.atomicAdd(pd.scanAlphaNumerator, d)
        end
	end
	
    terra kernels.PCGStep1(pd : PlanData)
        var d = 0.0f
        var idx : Index
        if idx:initFromCUDAParams() and (not [problemSpec:EvalExclude(idx,`pd.parameters)]) then
            var tmp : unknownElement = 0.0f
             -- A x p_k  => J^T x J x p_k 
            tmp = problemSpec.functions.applyJTJ.unknownfunction(idx, pd.parameters, pd.p.X)
            pd.Ap_X(idx) = tmp					 -- store for next kernel call
            d = pd.p(idx):dot(tmp)			 -- x-th term of denominator of alpha
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
        var idx : Index
        if idx:initFromCUDAParams() and (not [problemSpec:EvalExclude(idx,`pd.parameters)]) then
            -- sum over block results to compute denominator of alpha
            var alphaDenominator : float = pd.scanAlphaDenominator[0]
			var alphaNumerator : float = pd.scanAlphaNumerator[0]
                        
            -- update step size alpha
			var alpha = 0.0f
            if alphaDenominator > FLOAT_EPSILON then 
				alpha = alphaNumerator/alphaDenominator 
			end 
        
            pd.delta(idx) = pd.delta(idx)+alpha*pd.p(idx)		-- do a decent step
            
            var r = pd.r(idx)-alpha*pd.Ap_X(idx)				-- update residuum
            pd.r(idx) = r										-- store for next kernel call
        
			var pre = pd.preconditioner(idx)
			if not problemSpec.usepreconditioner then
				pre = 1.0f
			end
			
			if isGraph then
				pre = guardedInvert(pre)
			end
			
            var z = pre*r										-- apply pre-conditioner M^-1
            pd.z(idx) = z;										-- save for next kernel call
            
            b = z:dot(r)									-- compute x-th term of the numerator of beta
        end
        b = util.warpReduce(b)
        if (util.laneid() == 0) then
            util.atomicAdd(pd.scanBetaNumerator, b)
        end
    end
	
    terra kernels.PCGStep3(pd : PlanData)			
        var idx : Index
        if idx:initFromCUDAParams() and (not [problemSpec:EvalExclude(idx,`pd.parameters)])  then
			
			var rDotzNew : float = pd.scanBetaNumerator[0]				-- get new numerator
			var rDotzOld : float = pd.scanAlphaNumerator[0]				-- get old denominator

			var beta : float = 0.0f		                    
			if rDotzOld > FLOAT_EPSILON then beta = rDotzNew/rDotzOld end	-- update step size beta
			pd.p(idx) = pd.z(idx)+beta*pd.p(idx)							-- update decent direction

        end
    end
	
	terra kernels.PCGLinearUpdate(pd : PlanData)
        var idx : Index
        if idx:initFromCUDAParams() and (not [problemSpec:EvalExclude(idx,`pd.parameters)])  then
            pd.parameters.X(idx) = pd.parameters.X(idx) + pd.delta(idx)
        end
    end	
	
	terra kernels.computeCost(pd : PlanData)			
        var cost : float = 0.0f
        var idx : Index
        if idx:initFromCUDAParams() and (not [problemSpec:EvalExclude(idx,`pd.parameters)])  then
            var params = pd.parameters				
            cost = cost + [float](problemSpec.functions.cost.unknownfunction(idx, params))
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
        var idx : Index
        if idx:initFromCUDAParams() then
            escape
                if problemSpec.functions.precompute then
                    emit quote 
                        problemSpec.functions.precompute.unknownfunction(idx,pd.parameters)
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
