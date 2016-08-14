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
	local TUnknownType = UnknownType:terratype()	
	
    local isGraph = problemSpec:UsesGraphs() 
    
    local struct PlanData(S.Object) {
		plan : opt.Plan
		parameters : problemSpec:ParameterType()
		scratchF : &float

		delta : TUnknownType	--current linear update to be computed -> num vars
		r : TUnknownType		--residuals -> num vars	--TODO this needs to be a 'residual type'
		z : TUnknownType		--preconditioned residuals -> num vars	--TODO this needs to be a 'residual type'
		p : TUnknownType		--descent direction -> num vars
		Ap_X : TUnknownType	--cache values for next kernel call after A = J^T x J x p -> num vars
		preconditioner : TUnknownType --pre-conditioner for linear system -> num vars
		g : TUnknownType		--gradient of F(x): g = -2J'F -> num vars
		
		scanAlphaNumerator : &float
		scanAlphaDenominator : &float
		scanBetaNumerator : &float
		scanBetaDenominator : &float

        modelCostChange : &float    -- modelCostChange = L(0) - L(delta) where L(h) = F' F + 2 h' J' F + h' J' J h
        maxDiagJTJ : &float    -- maximum value in the diagonal of JTJ 
		
		timer : Timer
		endSolver : util.TimerEvent
		nIter : int				--current non-linear iter counter
		nIterations : int		--non-linear iterations
		lIterations : int		--linear iterations
	    prevCost : float
	}
	
	local delegate = {}
	
	function delegate.CenterFunctions(UnknownIndexSpace,fmap)
	    local kernels = {}
	    local unknownElement = UnknownType:VectorTypeForIndexSpace(UnknownIndexSpace)
	    local Index = UnknownIndexSpace:indextype()

        local terra guardedInvert(p : unknownElement)
            var invp = p
            for i = 0, invp:size() do
                invp(i) = terralib.select(invp(i) > FLOAT_EPSILON, 1.f / invp(i),invp(i))
            end
            return invp
        end
	
        terra kernels.PCGInit1(pd : PlanData)
            var d = 0.0f -- init for out of bounds lanes
        
            var idx : Index
            if idx:initFromCUDAParams() then
        
                -- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0                            
                var residuum : unknownElement = 0.0f
                var pre : unknownElement = 0.0f	
            
                if not fmap.exclude(idx,pd.parameters) then 
                
                    pd.delta(idx) = 0.0f    
                
                    residuum, pre = fmap.evalJTF(idx, pd.parameters)
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
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                var tmp : unknownElement = 0.0f
                 -- A x p_k  => J^T x J x p_k 
                tmp = fmap.applyJTJ(idx, pd.parameters, pd.p)
                pd.Ap_X(idx) = tmp					 -- store for next kernel call
                d = pd.p(idx):dot(tmp)			 -- x-th term of denominator of alpha
            end
            d = util.warpReduce(d)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.scanAlphaDenominator, d)
            end
        end
        terra kernels.PCGStep2(pd : PlanData)
            var b = 0.0f 
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
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
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
            
                var rDotzNew : float = pd.scanBetaNumerator[0]				-- get new numerator
                var rDotzOld : float = pd.scanAlphaNumerator[0]				-- get old denominator

                var beta : float = 0.0f		                    
                if rDotzOld > FLOAT_EPSILON then beta = rDotzNew/rDotzOld end	-- update step size beta
                pd.p(idx) = pd.z(idx)+beta*pd.p(idx)							-- update decent direction

            end
        end
    
        terra kernels.PCGLinearUpdate(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.parameters.X(idx) = pd.parameters.X(idx) + pd.delta(idx)
            end
        end	
        
        terra kernels.PCGLinearUpdateRevert(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.parameters.X(idx) = pd.parameters.X(idx) - pd.delta(idx)
            end
        end	

        terra kernels.computeCost(pd : PlanData)			
            var cost : float = 0.0f
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                var params = pd.parameters				
                cost = cost + [float](fmap.cost(idx, params))
            end

            cost = util.warpReduce(cost)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.scratchF, cost)
            end
        end

        if fmap.precompute then
            terra kernels.precompute(pd : PlanData)
                var idx : Index
                if idx:initFromCUDAParams() then
                   fmap.precompute(idx,pd.parameters)
                end
            end
        end

        terra kernels.computeModelCostChangeStep1(pd : PlanData)
            var d = 0.0f -- init for out of bounds lanes
        
            var idx : Index
            if idx:initFromCUDAParams() then
        
                -- grad_F = - 2J'F                           
                var g : unknownElement = 0.0f
            
                if not fmap.exclude(idx,pd.parameters) then 
                    g = fmap.evalJTFsimple(idx, pd.parameters)  -- 2J'F  
                    -- g = -g
                end        
                if not isGraph then
                    var dd = pd.delta(idx)
                    d = g:dot(dd)  -- delta'(-2J'F) 
                end
            end
            if not isGraph then
                d = util.warpReduce(d)	
                if (util.laneid() == 0) then				
                    util.atomicAdd(pd.modelCostChange, d)
                end
            end
        end
        terra kernels.computeModelCostChangeStep1_Finish(pd : PlanData)	--only called for graphs
            var d = 0.0f -- init for out of bounds lanes
            var idx : Index
            if idx:initFromCUDAParams() then
                var gradF = pd.g(idx)   -- gradF = -2J'F
                var dd = pd.delta(idx)
                d = gradF:dot(dd) -- delta'(-2J'F) 
            end
            d = util.warpReduce(d)	
            if (util.laneid() == 0) then
                util.atomicAdd(pd.modelCostChange, d)
            end
        end
        terra kernels.computeModelCostChangeStep2(pd : PlanData)
            var d = 0.0f -- init for out of bounds lanes
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                var jtjd : unknownElement = 0.0f
 
                jtjd = fmap.applyJTJsimple(idx, pd.parameters, pd.delta)   -- J'J delta
                -- jtjd = -jtjd
                d = pd.delta(idx):dot(jtjd)			 -- delta'(-J'J delta) 
            end
            d = util.warpReduce(d)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.modelCostChange, d)
            end
        end
        terra kernels.LMLambdaInit(pd : PlanData)
            var maxJTJ = 0.0f -- maximum diag(JTJ)
            var idx : Index
            if idx:initFromCUDAParams() then	
                var pre : unknownElement = 1.0f	
                
                if not fmap.exclude(idx,pd.parameters) then 
                    pre = fmap.evalDiagJTJ(idx, pd.parameters)
                end

                if not isGraph then
                    maxJTJ = pre:max()
                end
            end

            if not isGraph then
                maxJTJ = util.warpMaxReduce(maxJTJ)	
                if (util.laneid() == 0) then				
                    util.atomicMax(pd.maxDiagJTJ, maxJTJ)
                end
            end
        end
	    return kernels
	end
	
	function delegate.GraphFunctions(graphname,fmap)
	    local kernels = {}
        terra kernels.PCGInit1_Graph(pd : PlanData)
            var tIdx = 0
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
                fmap.evalJTF(tIdx, pd.parameters, pd.r, pd.preconditioner)
            end
        end    
        
    	terra kernels.PCGStep1_Graph(pd : PlanData)
            var d = 0.0f
            var tIdx = 0 
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
               d = d + fmap.applyJTJ(tIdx, pd.parameters, pd.p, pd.Ap_X)
            end 
            d = util.warpReduce(d)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.scanAlphaDenominator, d)
            end
        end
        terra kernels.computeCost_Graph(pd : PlanData)			
            var cost : float = 0.0f
            var tIdx = 0
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
                cost = cost + fmap.cost(tIdx, pd.parameters)
            end 
            cost = util.warpReduce(cost)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.scratchF, cost)
            end
        end
        terra kernels.computeModelCostChangeStep1_Graph(pd : PlanData)
            var tIdx = 0
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
                fmap.evalJTFsimple(tIdx, pd.parameters, pd.g)
            end
        end   
    	terra kernels.computeModelCostChangeStep2_Graph(pd : PlanData)
            var d = 0.0f
            var tIdx = 0 
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
               d = d + fmap.applyJTJsimple(tIdx, pd.parameters, pd.delta)
            end 
            d = util.warpReduce(d)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.modelCostChange, d)
            end
        end
        -- terra kernels.LMLambdaInit_Graph(pd : PlanData)
        --     var tIdx = 0
        --     if util.getValidGraphElement(pd,[graphname],&tIdx) then
        --         fmap.evalJTF(tIdx, pd.parameters, pd.preconditioner)
        --     end
        -- end   
	    return kernels
	end
	
	local gpu = util.makeGPUFunctions(problemSpec, PlanData, delegate, {"PCGInit1",
                                                                        "PCGInit1_Finish",
                                                                        "PCGStep1",
                                                                        "PCGStep2",
                                                                        "PCGStep3",
                                                                        "PCGLinearUpdate",
                                                                        "PCGLinearUpdateRevert",
                                                                        "computeCost",
                                                                        "precompute",
                                                                        "computeModelCostChangeStep1",
                                                                        "computeModelCostChangeStep1_Finish",
                                                                        "computeModelCostChangeStep2",
                                                                        "LMLambdaInit",
                                                                        "PCGInit1_Graph",
                                                                        "PCGStep1_Graph",
                                                                        "computeCost_Graph",
                                                                        "computeModelCostChangeStep1_Graph",
                                                                        "computeModelCostChangeStep2_Graph"
                                                                        })

    
    local terra computeCost(pd : &PlanData) : float
        C.cudaMemset(pd.scratchF, 0, sizeof(float))
        gpu.computeCost(pd)
        gpu.computeCost_Graph(pd)
        var f : float
        C.cudaMemcpy(&f, pd.scratchF, sizeof(float), C.cudaMemcpyDeviceToHost)
        return f
    end

    local terra computeModelCostChange(pd : &PlanData) : float
        C.cudaMemset(pd.modelCostChange, 0, sizeof(float))

        -- model_cost_change = delta' * (-2 * J' * F) + delta' * (-J' * J * delta)
        gpu.computeModelCostChangeStep1(pd)
        gpu.computeModelCostChangeStep2(pd)
        if isGraph then
            gpu.computeModelCostChangeStep1_Graph(pd)	
            gpu.computeModelCostChangeStep1_Finish(pd)
            gpu.computeModelCostChangeStep2_Graph(pd)
        end

        var model_cost_change : float
        C.cudaMemcpy(&model_cost_change, pd.modelCostChange, sizeof(float), C.cudaMemcpyDeviceToHost)

        return model_cost_change
    end

    local terra initLambda(pd : &PlanData)
        -- Init lambda based on the maximum value on the diagonal of JTJ
        C.cudaMemset(pd.maxDiagJTJ, 0, sizeof(float))

        -- lambda = tau * max{a_ii} where A = JTJ
        var tau = 1e-6f
        gpu.LMLambdaInit(pd);
        var maxDiagJTJ : float
        C.cudaMemcpy(&maxDiagJTJ, pd.maxDiagJTJ, sizeof(float), C.cudaMemcpyDeviceToHost)
        pd.parameters.lambda = tau * maxDiagJTJ
    end

	local terra init(data_ : &opaque, params_ : &&opaque, solverparams : &&opaque)
	   var pd = [&PlanData](data_)
	   pd.timer:init()
	   pd.timer:startEvent("overall",nil,&pd.endSolver)
       [util.initParameters(`pd.parameters,problemSpec,params_,true)]
	   pd.nIter = 0
	   pd.nIterations = @[&int](solverparams[0])
	   pd.lIterations = @[&int](solverparams[1])
       escape if problemSpec:UsesLambda() then
	    --   emit quote pd.parameters.lambda = 0.001f end
          emit quote 
            initLambda(pd)
          end
          emit quote pd.parameters.lambda_increase_factor = 2.0f end
	   end end
	   gpu.precompute(pd)
	   pd.prevCost = computeCost(pd)
	end
	
	local terra step(data_ : &opaque, params_ : &&opaque, solverparams : &&opaque)
		var pd = [&PlanData](data_)
		[util.initParameters(`pd.parameters,problemSpec, params_,false)]
		if pd.nIter < pd.nIterations then
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
			gpu.precompute(pd)
			var newCost = computeCost(pd)
			
			logSolver("\t%d: prev=%f new=%f ", pd.nIter, pd.prevCost,newCost)
			
			escape if problemSpec:UsesLambda() then
			    -- lm version
			    emit quote
                    logSolver(" lambda=%f ",pd.parameters.lambda)

                    var cost_change = pd.prevCost - newCost
                    var model_cost_change = computeModelCostChange(pd)
                    var relative_decrease = cost_change / model_cost_change

                    logSolver(" cost_change=%f ", cost_change)
                    logSolver(" model_cost_change=%f ", model_cost_change)
                    logSolver(" relative_decrease=%f ", relative_decrease)

                    var min_relative_decrease = 1e-3f

                    if cost_change < 0 or relative_decrease < min_relative_decrease then	--in this case we revert
                        gpu.PCGLinearUpdateRevert(pd)

                        -- lambda = lambda * nu
                        pd.parameters.lambda = pd.parameters.lambda * pd.parameters.lambda_increase_factor
                        -- nu = 2 * nu
                        pd.parameters.lambda_increase_factor = 2.0f * pd.parameters.lambda_increase_factor

                        logSolver("REVERT\n")
                        gpu.precompute(pd)
                    else 
                        -- lambda = lambda * max{1/3, 1 - (2 * relative_decrease - 1)^3}
                        var min_factor = 1.0f/3.0f
                        var tmp_factor = 1.0f - util.cpuMath.pow(2.0f * relative_decrease - 1.0f, 3.0f)
                        pd.parameters.lambda = pd.parameters.lambda * util.cpuMath.fmax(min_factor, tmp_factor)
                        pd.parameters.lambda_increase_factor = 2.0f

                        logSolver("\n")
                        pd.prevCost = newCost
                    end
                    var min_lm_diagonal = 1e-6f
                    var max_lm_diagonal = 1e32f
                    pd.parameters.lambda = util.cpuMath.fmax(min_lm_diagonal, pd.parameters.lambda)
                    pd.parameters.lambda = util.cpuMath.fmin(max_lm_diagonal, pd.parameters.lambda)
                end
            else
                emit quote
                    logSolver("\n")
                    pd.prevCost = newCost 
                end
            end end
			pd.nIter = pd.nIter + 1
			return 1
		else
			logSolver("final cost=%f\n", pd.prevCost)
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
		pd.g:initGPU()
		[util.initPrecomputedImages(`pd.parameters,problemSpec)]	
		C.cudaMalloc([&&opaque](&(pd.scanAlphaNumerator)), sizeof(float))
		C.cudaMalloc([&&opaque](&(pd.scanBetaNumerator)), sizeof(float))
		C.cudaMalloc([&&opaque](&(pd.scanAlphaDenominator)), sizeof(float))
		C.cudaMalloc([&&opaque](&(pd.scanBetaDenominator)), sizeof(float))
		C.cudaMalloc([&&opaque](&(pd.modelCostChange)), sizeof(float))
		C.cudaMalloc([&&opaque](&(pd.maxDiagJTJ)), sizeof(float))
		
		C.cudaMalloc([&&opaque](&(pd.scratchF)), sizeof(float))
		return &pd.plan
	end
	return makePlan
end
