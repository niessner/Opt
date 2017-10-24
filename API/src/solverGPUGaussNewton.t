local S = require("std")
local util = require("util")
require("precision")

local ffi = require("ffi")

local C = util.C
local Timer = util.Timer

local getValidUnknown = util.getValidUnknown

local GuardedInvertType = { CERES = {}, MODIFIED_CERES = {}, EPSILON_ADD = {} }

-- CERES default, ONCE_PER_SOLVE
local JacobiScalingType = { NONE = {}, ONCE_PER_SOLVE = {}, EVERY_ITERATION = {}}


local initialization_parameters = {
    use_cusparse = false,
    use_fused_jtj = false,
    guardedInvertType = GuardedInvertType.CERES,
    jacobiScaling = JacobiScalingType.ONCE_PER_SOLVE
}

local solver_parameter_defaults = {
    residual_reset_period = 10,
    min_relative_decrease = 1e-3,
    min_trust_region_radius = 1e-32,
    max_trust_region_radius = 1e16,
    q_tolerance = 0.0001,
    function_tolerance = 0.000001,
    trust_region_radius = 1e4,
    radius_decrease_factor = 2.0,
    min_lm_diagonal = 1e-6,
    max_lm_diagonal = 1e32,
    nIterations = 10,
    lIterations = 10
}


local multistep_alphaDenominator_compute = initialization_parameters.use_cusparse

local cd = macro(function(apicall) 
    local apicallstr = tostring(apicall)
    local filename = debug.getinfo(1,'S').source
    return quote
        var str = [apicallstr]
        var r = apicall
        if r ~= 0 then  
            C.printf("Cuda reported error %d: %s\n",r, C.cudaGetErrorString(r))
            C.printf("In call: %s", str)
            C.printf("In file: %s\n", filename)
            C.exit(r)
        end
    in
        r
    end end)


local gpuMath = util.gpuMath

local FLOAT_EPSILON = `[opt_float](0.00000001f) 
-- GAUSS NEWTON (or LEVENBERG-MARQUADT)
return function(problemSpec)
    local UnknownType = problemSpec:UnknownType()
    local TUnknownType = UnknownType:terratype()	
    
    local isGraph = problemSpec:UsesGraphs() 
    
    local struct SolverParameters {
        min_relative_decrease : float
        min_trust_region_radius : float
        max_trust_region_radius : float
        q_tolerance : float
        function_tolerance : float
        trust_region_radius : float
        radius_decrease_factor : float
        min_lm_diagonal : float
        max_lm_diagonal : float

        residual_reset_period : int
        nIter : int             --current non-linear iter counter
        nIterations : int       --non-linear iterations
        lIterations : int       --linear iterations
    }

    local struct PlanData {
        plan : opt.Plan
        parameters : problemSpec:ParameterType()
        solverparameters : SolverParameters
        gpuDims : &uint32
        cpuDims : &uint32
        scratch : &opt_float

        delta : TUnknownType	--current linear update to be computed -> num vars
        r : TUnknownType		--residuals -> num vars	--TODO this needs to be a 'residual type'
        b : TUnknownType        --J^TF. Constant during inner iterations, only used to recompute r to counteract drift -> num vars --TODO this needs to be a 'residual type'
        Adelta : TUnknownType       -- (A'A+D'D)delta TODO this needs to be a 'residual type'
        z : TUnknownType		--preconditioned residuals -> num vars	--TODO this needs to be a 'residual type'
        p : TUnknownType		--descent direction -> num vars
        Ap_X : TUnknownType	--cache values for next kernel call after A = J^T x J x p -> num vars
        CtC : TUnknownType -- The diagonal matrix C'C for the inner linear solve (J'J+C'C)x = J'F Used only by LM
        preconditioner : TUnknownType --pre-conditioner for linear system -> num vars
        SSq : TUnknownType -- Square of jacobi scaling diagonal
        g : TUnknownType		--gradient of F(x): g = -2J'F -> num vars

        prevX : TUnknownType -- Place to copy unknowns to before speculatively updating. Avoids hassle when (X + delta) - delta != X 

        scanAlphaNumerator : &opt_float
        scanAlphaDenominator : &opt_float
        scanBetaNumerator : &opt_float

        modelCost : &opt_float    -- modelCost = L(delta) where L(h) = F' F + 2 h' J' F + h' J' J h
        q : &opt_float -- Q value for zeta calculation (see CERES)

        timer : Timer
        endSolver : util.TimerEvent

        prevCost : opt_float
        
    }
    S.Object(PlanData)

    assert(not initialization_parameters.use_cusparse)
	
	local delegate = {}
	function delegate.CenterFunctions(UnknownIndexSpace,fmap)
	    local kernels = {}
	    local unknownElement = UnknownType:VectorTypeForIndexSpace(UnknownIndexSpace)
	    local Index = UnknownIndexSpace:indextype()

        local unknownWideReduction = macro(function(idx,val,reductionTarget) return quote
            val = util.warpReduce(val)
            if (util.laneid() == 0) then                
                util.atomicAdd(reductionTarget, val)
            end
        end end)

        local terra square(x : opt_float) : opt_float
            return x*x
        end

        local terra guardedInvert(p : unknownElement)
            escape 
                if initialization_parameters.guardedInvertType == GuardedInvertType.CERES then
                    emit quote
                        var invp = p
                        for i = 0, invp:size() do
                            invp(i) = [opt_float](1.f) / square(opt_float(1.f) + util.gpuMath.sqrt(invp(i)))
                        end
                        return invp
                    end
                elseif initialization_parameters.guardedInvertType == GuardedInvertType.MODIFIED_CERES then
                    emit quote
                        var invp = p
                        for i = 0, invp:size() do
                             invp(i) = [opt_float](1.f) / (opt_float(1.f) + invp(i))
                        end
                        return invp
                    end
                elseif initialization_parameters.guardedInvertType == GuardedInvertType.EPSILON_ADD then
                    emit quote
                        var invp = p
                        for i = 0, invp:size() do
                            invp(i) = [opt_float](1.f) / (FLOAT_EPSILON + invp(i))
                        end
                        return invp
                    end
                end
            end
        end

        local terra clamp(x : unknownElement, minVal : unknownElement, maxVal : unknownElement) : unknownElement
            var result = x
            for i = 0, result:size() do
                result(i) = util.gpuMath.fmin(util.gpuMath.fmax(x(i), minVal(i)), maxVal(i))
            end
            return result
        end

        terra kernels.PCGInit1(pd : PlanData)
            var d : opt_float = opt_float(0.0f) -- init for out of bounds lanes
            var idx : Index
            var dims = pd.gpuDims
            if idx:initFromCUDAParams(dims) then
        
                -- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0                            
                var residuum : unknownElement = 0.0f
                var pre : unknownElement = 0.0f	
                
                if not fmap.exclude(idx,pd.parameters,pd.gpuDims) then 
                
                    pd.delta(idx,dims) = opt_float(0.0f)   
                
                    residuum, pre = fmap.evalJTF(idx, pd.parameters, pd.gpuDims)
                    residuum = -residuum
                    pd.r(idx,dims) = residuum
                
                    if not problemSpec.usepreconditioner then
                        pre = opt_float(1.0f)
                    end
                end        
            
                if (not fmap.exclude(idx,pd.parameters,pd.gpuDims)) and (not isGraph) then		
                    pre = guardedInvert(pre)
                    var p = pre*residuum	-- apply pre-conditioner M^-1			   
                    pd.p(idx,dims) = p
                
                    d = residuum:dot(p) 
                end
            
                pd.preconditioner(idx,dims) = pre
            end 
            if not isGraph then
                unknownWideReduction(idx,d,pd.scanAlphaNumerator)
            end
        end
    
        terra kernels.PCGInit1_Finish(pd : PlanData)	--only called for graphs
            var d : opt_float = opt_float(0.0f) -- init for out of bounds lanes
            var idx : Index
            var dims = pd.gpuDims
            if idx:initFromCUDAParams(dims) then
                var residuum = pd.r(idx,dims)			
                var pre = pd.preconditioner(idx,dims)
            
                pre = guardedInvert(pre)
            
                if not problemSpec.usepreconditioner then
                    pre = opt_float(1.0f)
                end
            
                var p = pre*residuum	-- apply pre-conditioner M^-1
                pd.preconditioner(idx,dims) = pre
                pd.p(idx,dims) = p
                d = residuum:dot(p)
            end

            unknownWideReduction(idx,d,pd.scanAlphaNumerator)
        end

        terra kernels.PCGStep1(pd : PlanData)
            var d : opt_float = opt_float(0.0f)
            var idx : Index
            var dims = pd.gpuDims
            if idx:initFromCUDAParams(dims) and not fmap.exclude(idx,pd.parameters,pd.gpuDims) then
                var tmp : unknownElement = 0.0f
                 -- A x p_k  => J^T x J x p_k 
                tmp = fmap.applyJTJ(idx, pd.parameters, dims, pd.p, pd.CtC)
                pd.Ap_X(idx,dims) = tmp					 -- store for next kernel call
                d = pd.p(idx,dims):dot(tmp)			 -- x-th term of denominator of alpha
            end
            if not [multistep_alphaDenominator_compute] then
                unknownWideReduction(idx,d,pd.scanAlphaDenominator)
            end
        end
        if multistep_alphaDenominator_compute then
            terra kernels.PCGStep1_Finish(pd : PlanData)
                var d : opt_float = opt_float(0.0f)
                var idx : Index
                var dims = pd.gpuDims
                if idx:initFromCUDAParams(dims) and not fmap.exclude(idx,pd.parameters,pd.gpuDims) then
                    d = pd.p(idx,dims):dot(pd.Ap_X(idx,dims))           -- x-th term of denominator of alpha
                end
                unknownWideReduction(idx,d,pd.scanAlphaDenominator)
            end
        end

        terra kernels.PCGStep2(pd : PlanData)
            var betaNum = opt_float(0.0f) 
            var q = opt_float(0.0f) -- Only used if LM
            var idx : Index
            var dims = pd.gpuDims
            if idx:initFromCUDAParams(dims) and not fmap.exclude(idx,pd.parameters,pd.gpuDims) then
                -- sum over block results to compute denominator of alpha
                var alphaDenominator : opt_float = pd.scanAlphaDenominator[0]
                var alphaNumerator : opt_float = pd.scanAlphaNumerator[0]

                -- update step size alpha
                var alpha = opt_float(0.0f)
                alpha = alphaNumerator/alphaDenominator 
    
                var delta = pd.delta(idx,dims)+alpha*pd.p(idx,dims)       -- do a descent step
                pd.delta(idx,dims) = delta

                var r = pd.r(idx,dims)-alpha*pd.Ap_X(idx,dims)				-- update residuum
                pd.r(idx,dims) = r										-- store for next kernel call

                var pre = pd.preconditioner(idx,dims)
                if not problemSpec.usepreconditioner then
                    pre = opt_float(1.0f)
                end
        
                var z = pre*r										-- apply pre-conditioner M^-1
                pd.z(idx,dims) = z;										-- save for next kernel call

                betaNum = z:dot(r)									-- compute x-th term of the numerator of beta

                if [problemSpec:UsesLambda()] then
                    -- computeQ    
                    -- Right side is -2 of CERES versions, left is just negative version, 
                    --  so after the dot product, just need to multiply by 2 to recover value identical to CERES  
                    q = 0.5*(delta:dot(r + pd.b(idx,dims))) 
                end
            end
            
            unknownWideReduction(idx,betaNum,pd.scanBetaNumerator)
            if [problemSpec:UsesLambda()] then
                unknownWideReduction(idx,q,pd.q)
            end
        end

        terra kernels.PCGStep2_1stHalf(pd : PlanData)
            var idx : Index
            var dims = pd.gpuDims
            if idx:initFromCUDAParams(dims) and not fmap.exclude(idx,pd.parameters,dims) then
                var alphaDenominator : opt_float = pd.scanAlphaDenominator[0]
                var alphaNumerator : opt_float = pd.scanAlphaNumerator[0]
                -- update step size alpha
                var alpha = alphaNumerator/alphaDenominator 
                pd.delta(idx,dims) = pd.delta(idx,dims)+alpha*pd.p(idx,dims)       -- do a descent step
            end
        end

        terra kernels.PCGStep2_2ndHalf(pd : PlanData)
            var betaNum = opt_float(0.0f) 
            var q = opt_float(0.0f) 
            var idx : Index
            var dims = pd.gpuDims
            if idx:initFromCUDAParams(dims) and not fmap.exclude(idx,pd.parameters,pd.gpuDims) then
                -- Recompute residual
                var Ax = pd.Adelta(idx,dims)
                var b = pd.b(idx,dims)
                var r = b - Ax
                pd.r(idx,dims) = r

                var pre = pd.preconditioner(idx,dims)
                if not problemSpec.usepreconditioner then
                    pre = opt_float(1.0f)
                end
                var z = pre*r       -- apply pre-conditioner M^-1
                pd.z(idx,dims) = z;      -- save for next kernel call
                betaNum = z:dot(r)        -- compute x-th term of the numerator of beta
                if [problemSpec:UsesLambda()] then
                    -- computeQ    
                    -- Right side is -2 of CERES versions, left is just negative version, 
                    --  so after the dot product, just need to multiply by 2 to recover value identical to CERES  
                    q = 0.5*(pd.delta(idx,dims):dot(r + b)) 
                end
            end
            unknownWideReduction(idx,betaNum,pd.scanBetaNumerator) 
            if [problemSpec:UsesLambda()] then
                unknownWideReduction(idx,q,pd.q)
            end
        end


        terra kernels.PCGStep3(pd : PlanData)			
            var idx : Index
            var dims = pd.gpuDims
            if idx:initFromCUDAParams(dims) and not fmap.exclude(idx,pd.parameters,dims) then
            
                var rDotzNew : opt_float = pd.scanBetaNumerator[0]	-- get new numerator
                var rDotzOld : opt_float = pd.scanAlphaNumerator[0]	-- get old denominator

                var beta : opt_float = opt_float(0.0f)
                beta = rDotzNew/rDotzOld
                pd.p(idx,dims) = pd.z(idx,dims)+beta*pd.p(idx,dims)			    -- update decent direction
            end
        end
    
        terra kernels.PCGLinearUpdate(pd : PlanData)
            var idx : Index
            var dims = pd.gpuDims
            if idx:initFromCUDAParams(dims)and not fmap.exclude(idx,pd.parameters,dims) then
                pd.parameters.X(idx,dims) = pd.parameters.X(idx,dims) + pd.delta(idx,dims)
            end
        end	
        
        terra kernels.revertUpdate(pd : PlanData)
            var idx : Index
            var dims = pd.gpuDims
            if idx:initFromCUDAParams(dims) and not fmap.exclude(idx,pd.parameters,dims) then
                pd.parameters.X(idx,dims) = pd.prevX(idx,dims)
            end
        end	

        terra kernels.computeAdelta(pd : PlanData)
            var idx : Index
            var dims = pd.gpuDims
            if idx:initFromCUDAParams(dims) and not fmap.exclude(idx,pd.parameters,dims) then
                pd.Adelta(idx,dims) = fmap.applyJTJ(idx, pd.parameters, dims, pd.delta, pd.CtC)
            end
        end

        terra kernels.savePreviousUnknowns(pd : PlanData)
            var idx : Index
            var dims = pd.gpuDims
            if idx:initFromCUDAParams(dims) and not fmap.exclude(idx,pd.parameters,dims) then
                pd.prevX(idx,dims) = pd.parameters.X(idx,dims)
            end
        end 

        terra kernels.computeCost(pd : PlanData)
            var cost : opt_float = opt_float(0.0f)
            var idx : Index
            var dims = pd.gpuDims
            if idx:initFromCUDAParams(dims) and not fmap.exclude(idx,pd.parameters,dims) then
                cost = fmap.cost(idx, pd.parameters, dims)
                printf("%f\n", cost)
            end

            cost = util.warpReduce(cost)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.scratch, cost)
            end
        end

        if fmap.precompute then
            terra kernels.precompute(pd : PlanData)
                var idx : Index
                var dims = pd.gpuDims
                if idx:initFromCUDAParams(dims) then
                   fmap.precompute(idx,pd.parameters,dims)
                end
            end
        end
        if problemSpec:UsesLambda() then
            terra kernels.PCGComputeCtC(pd : PlanData)
                var idx : Index
                var dims = pd.gpuDims
                if idx:initFromCUDAParams(dims) and not fmap.exclude(idx,pd.parameters,pd.gpuDims) then 
                    var CtC = fmap.computeCtC(idx, pd.parameters, dims)
                    pd.CtC(idx,dims) = CtC    
                end 
            end

            terra kernels.PCGSaveSSq(pd : PlanData)
                var idx : Index
                var dims = pd.gpuDims
                if idx:initFromCUDAParams(dims) and not fmap.exclude(idx,pd.parameters,pd.gpuDims) then 
                    pd.SSq(idx,dims) = pd.preconditioner(idx,dims)       
                end 
            end

            terra kernels.PCGFinalizeDiagonal(pd : PlanData)
                var idx : Index
                var d = opt_float(0.0f)
                var q = opt_float(0.0f)
                var dims = pd.gpuDims
                if idx:initFromCUDAParams(dims) and not fmap.exclude(idx,pd.parameters,pd.gpuDims) then 
                    var unclampedCtC = pd.CtC(idx)
                    var invS_iiSq : unknownElement = opt_float(1.0f)
                    if [initialization_parameters.jacobiScaling == JacobiScalingType.ONCE_PER_SOLVE] then
                        invS_iiSq = opt_float(1.0f) / pd.SSq(idx,dims)
                    elseif [initialization_parameters.jacobiScaling == JacobiScalingType.EVERY_ITERATION] then 
                        invS_iiSq = opt_float(1.0f) / pd.preconditioner(idx,dims)
                    end -- else if  [initialization_parameters.jacobiScaling == JacobiScalingType.NONE] then invS_iiSq == 1
                    var clampMultiplier = invS_iiSq / pd.parameters.trust_region_radius
                    var minVal = pd.parameters.min_lm_diagonal * clampMultiplier
                    var maxVal = pd.parameters.max_lm_diagonal * clampMultiplier
                    var CtC = clamp(unclampedCtC, minVal, maxVal)
                    pd.CtC(idx,dims) = CtC
                    
                    -- Calculate true preconditioner, taking into account the diagonal
                    var pre = opt_float(1.0f) / (CtC+pd.parameters.trust_region_radius*unclampedCtC) 
                    pd.preconditioner(idx,dims) = pre
                    var residuum = pd.r(idx,dims)
                    pd.b(idx,dims) = residuum -- copy over to b
                    var p = pre*residuum    -- apply pre-conditioner M^-1
                    pd.p(idx,dims) = p
                    d = residuum:dot(p)
                    -- computeQ    
                    -- Right side is -2 of CERES versions, left is just negative version, 
                    --  so after the dot product, just need to multiply by 2 to recover value identical to CERES  
                    q = 0.5*(pd.delta(idx,dims):dot(residuum + residuum)) 
                end    
                unknownWideReduction(idx,q,pd.q)
                unknownWideReduction(idx,d,pd.scanAlphaNumerator)
            end

            terra kernels.computeModelCost(pd : PlanData)            
                var cost : opt_float = opt_float(0.0f)
                var idx : Index
                var dims = pd.gpuDims
                if idx:initFromCUDAParams(dims) and not fmap.exclude(idx,pd.parameters,pd.gpuDims) then
                    var params = pd.parameters              
                    cost = fmap.modelcost(idx, params, dims, pd.delta)
                end

                cost = util.warpReduce(cost)
                if (util.laneid() == 0) then
                    util.atomicAdd(pd.modelCost, cost)
                end
            end

        end -- :UsesLambda()
	    return kernels
	end
	
	function delegate.GraphFunctions(graphname,fmap,ES)
	    --print("ES-graph",fmap.derivedfrom)
	    local kernels = {}
        terra kernels.PCGInit1_Graph(pd : PlanData)
            var tIdx = 0
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
                fmap.evalJTF(tIdx, pd.parameters, pd.gpuDims, pd.r, pd.preconditioner)
            end
        end    
        
    	terra kernels.PCGStep1_Graph(pd : PlanData)
            var d = opt_float(0.0f)
            var tIdx = 0 
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
               d = d + fmap.applyJTJ(tIdx, pd.parameters, pd.gpuDims, pd.p, pd.Ap_X)
            end 
            if not [multistep_alphaDenominator_compute] then
                d = util.warpReduce(d)
                if (util.laneid() == 0) then
                    util.atomicAdd(pd.scanAlphaDenominator, d)
                end
            end
        end

        terra kernels.computeAdelta_Graph(pd : PlanData)
            var tIdx = 0 
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
                fmap.applyJTJ(tIdx, pd.parameters, pd.gpuDims, pd.delta, pd.Adelta)
            end
        end

        terra kernels.computeCost_Graph(pd : PlanData)
            var cost : opt_float = opt_float(0.0f)
            var tIdx = 0
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
                cost = fmap.cost(tIdx, pd.parameters, pd.gpuDims)
            end 
            cost = util.warpReduce(cost)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.scratch, cost)
            end
        end

        if problemSpec:UsesLambda() then
            terra kernels.PCGComputeCtC_Graph(pd : PlanData)
                var tIdx = 0
                if util.getValidGraphElement(pd,[graphname],&tIdx) then
                    fmap.computeCtC(tIdx, pd.parameters, pd.gpuDims, pd.CtC)
                end
            end    

            terra kernels.computeModelCost_Graph(pd : PlanData)          
                var cost : opt_float = opt_float(0.0f)
                var tIdx = 0
                if util.getValidGraphElement(pd,[graphname],&tIdx) then
                    cost = fmap.modelcost(tIdx, pd.parameters, pd.gpuDims, pd.delta)
                end 
                cost = util.warpReduce(cost)
                if (util.laneid() == 0) then
                    util.atomicAdd(pd.modelCost, cost)
                end
            end
        end

	    return kernels
	end
	
	local gpu = util.makeGPUFunctions(problemSpec, PlanData, delegate, {"PCGInit1",
                                                                        "PCGInit1_Finish",
                                                                        "PCGComputeCtC",
                                                                        "PCGFinalizeDiagonal",
                                                                        "PCGStep1",
                                                                        "PCGStep1_Finish",
                                                                        "PCGStep2",
                                                                        "PCGStep2_1stHalf",
                                                                        "PCGStep2_2ndHalf",
                                                                        "PCGStep3",
                                                                        "PCGLinearUpdate",
                                                                        "revertUpdate",
                                                                        "savePreviousUnknowns",
                                                                        "computeCost",
                                                                        "PCGSaveSSq",
                                                                        "precompute",
                                                                        "computeAdelta",
                                                                        "computeAdelta_Graph",
                                                                        "PCGInit1_Graph",
                                                                        "PCGComputeCtC_Graph",
                                                                        "PCGStep1_Graph",
                                                                        "computeCost_Graph",
                                                                        "computeModelCost",
                                                                        "computeModelCost_Graph",
                                                                        "saveJToCRS",
                                                                        "saveJToCRS_Graph"
                                                                        })

    local terra computeCost(pd : &PlanData) : opt_float
        C.cudaMemset(pd.scratch, 0, sizeof(opt_float))
        gpu.computeCost(pd)
        gpu.computeCost_Graph(pd)
        var f : opt_float
        C.cudaMemcpy(&f, pd.scratch, sizeof(opt_float), C.cudaMemcpyDeviceToHost)
        return f
    end

    local terra computeModelCost(pd : &PlanData) : opt_float
        C.cudaMemset(pd.modelCost, 0, sizeof(opt_float))
        gpu.computeModelCost(pd)
        gpu.computeModelCost_Graph(pd)
        var f : opt_float
        C.cudaMemcpy(&f, pd.modelCost, sizeof(opt_float), C.cudaMemcpyDeviceToHost)
        return f
    end

    local sqrtf = util.cpuMath.sqrt

    local terra fetchQ(pd : &PlanData) : opt_float
        var f : opt_float
        C.cudaMemcpy(&f, pd.q, sizeof(opt_float), C.cudaMemcpyDeviceToHost)
        return f
    end

    local computeModelCostChange
    
    if problemSpec:UsesLambda() then
        terra computeModelCostChange(pd : &PlanData) : opt_float
            var cost = pd.prevCost
            var model_cost = computeModelCost(pd)
            logSolver(" cost=%f \n",cost)
            logSolver(" model_cost=%f \n",model_cost)
            var model_cost_change = cost - model_cost
            logSolver(" model_cost_change=%f \n",model_cost_change)
            return model_cost_change
        end
    end

    local terra GetToHost(ptr : &opaque, N : int) : &int
        var r = [&int](C.malloc(sizeof(int)*N))
        C.cudaMemcpy(r,ptr,N*sizeof(int),C.cudaMemcpyDeviceToHost)
        return r
    end

	local terra init(data_ : &opaque, params_ : &&opaque)
	   var pd = [&PlanData](data_)
	   pd.timer:init()
	   pd.timer:startEvent("overall",nil,&pd.endSolver)
       [util.initParameters(`pd.parameters,problemSpec,`pd.cpuDims,params_,true)]
	   pd.solverparameters.nIter = 0
       escape 
            if problemSpec:UsesLambda() then
              emit quote 
                pd.parameters.trust_region_radius       = pd.solverparameters.trust_region_radius
                pd.parameters.radius_decrease_factor    = pd.solverparameters.radius_decrease_factor
                pd.parameters.min_lm_diagonal           = pd.solverparameters.min_lm_diagonal
                pd.parameters.max_lm_diagonal           = pd.solverparameters.max_lm_diagonal
              end
	        end 
       end
	   gpu.precompute(pd)
	   pd.prevCost = computeCost(pd)
	end

	local terra cleanup(pd : &PlanData)
        logSolver("final cost=%f\n", pd.prevCost)
        pd.timer:endEvent(nil,pd.endSolver)
        pd.timer:evaluate()
        pd.timer:cleanup()
    end

	local terra step(data_ : &opaque, params_ : &&opaque)
        var pd = [&PlanData](data_)
        var residual_reset_period : int         = pd.solverparameters.residual_reset_period
        var min_relative_decrease : opt_float   = pd.solverparameters.min_relative_decrease
        var min_trust_region_radius : opt_float = pd.solverparameters.min_trust_region_radius
        var max_trust_region_radius : opt_float = pd.solverparameters.max_trust_region_radius
        var q_tolerance : opt_float             = pd.solverparameters.q_tolerance
        var function_tolerance : opt_float      = pd.solverparameters.function_tolerance
        var Q0 : opt_float
        var Q1 : opt_float
		[util.initParameters(`pd.parameters,problemSpec,`pd.cpuDims, params_,false)]
		if pd.solverparameters.nIter < pd.solverparameters.nIterations then
			C.cudaMemset(pd.scanAlphaNumerator, 0, sizeof(opt_float))	--scan in PCGInit1 requires reset
			C.cudaMemset(pd.scanAlphaDenominator, 0, sizeof(opt_float))	--scan in PCGInit1 requires reset
			C.cudaMemset(pd.scanBetaNumerator, 0, sizeof(opt_float))	--scan in PCGInit1 requires reset

			gpu.PCGInit1(pd)
			if isGraph then
				gpu.PCGInit1_Graph(pd)	
				gpu.PCGInit1_Finish(pd)	
			end

            escape 
                if problemSpec:UsesLambda() then
                    emit quote
                        C.cudaMemset(pd.scanAlphaNumerator, 0, sizeof(opt_float))
                        C.cudaMemset(pd.q, 0, sizeof(opt_float))
                        if [initialization_parameters.jacobiScaling == JacobiScalingType.ONCE_PER_SOLVE] and pd.solverparameters.nIter == 0 then
                            gpu.PCGSaveSSq(pd)
                        end
                        gpu.PCGComputeCtC(pd)
                        gpu.PCGComputeCtC_Graph(pd)
                        -- This also computes Q
                        gpu.PCGFinalizeDiagonal(pd)
                        Q0 = fetchQ(pd)
                    end
                end
            end
            for lIter = 0, pd.solverparameters.lIterations do				

                C.cudaMemset(pd.scanAlphaDenominator, 0, sizeof(opt_float))
                C.cudaMemset(pd.q, 0, sizeof(opt_float))

                
				gpu.PCGStep1(pd)
				if isGraph then
					gpu.PCGStep1_Graph(pd)
				end
                

                if multistep_alphaDenominator_compute then
                    gpu.PCGStep1_Finish(pd)
                end
				
				C.cudaMemset(pd.scanBetaNumerator, 0, sizeof(opt_float))
				
				if [problemSpec:UsesLambda()] and ((lIter + 1) % residual_reset_period) == 0 then
                    gpu.PCGStep2_1stHalf(pd)
                    gpu.computeAdelta(pd)
                    if isGraph then
                        gpu.computeAdelta_Graph(pd)
                    end
                    gpu.PCGStep2_2ndHalf(pd)
                else
                    gpu.PCGStep2(pd)
                end
                gpu.PCGStep3(pd)

				-- save new rDotz for next iteration
				C.cudaMemcpy(pd.scanAlphaNumerator, pd.scanBetaNumerator, sizeof(opt_float), C.cudaMemcpyDeviceToDevice)	
				
				if [problemSpec:UsesLambda()] then
	                Q1 = fetchQ(pd)
	                var zeta = [opt_float](lIter+1)*(Q1 - Q0) / Q1 
                    --logSolver("%d: Q0(%g) Q1(%g), zeta(%g)\n", lIter, Q0, Q1, zeta)
	                if zeta < q_tolerance then
                        logSolver("zeta=%.18g, breaking at iteration: %d\n", zeta, (lIter+1))
	                    break
	                end
	                Q0 = Q1
				end
			end
			

            var model_cost_change : opt_float

            escape if problemSpec:UsesLambda() then
                emit quote 
                    model_cost_change = computeModelCostChange(pd)
                    gpu.savePreviousUnknowns(pd)
                end
            end end

			gpu.PCGLinearUpdate(pd)    
			gpu.precompute(pd)
			var newCost = computeCost(pd)

			escape 
                if problemSpec:UsesLambda() then
                    emit quote
                        var cost_change = pd.prevCost - newCost
                        
                        
                        -- See CERES's TrustRegionStepEvaluator::StepAccepted() for a more complicated version of this
                        var relative_decrease = cost_change / model_cost_change
                        if cost_change >= 0 and relative_decrease > min_relative_decrease then
                            var absolute_function_tolerance = pd.prevCost * function_tolerance
                            if cost_change <= absolute_function_tolerance then
                                logSolver("\nFunction tolerance reached, exiting\n")
                                cleanup(pd)
                                return 0
                            end

                            var step_quality = relative_decrease
                            var min_factor = 1.0/3.0
                            var tmp_factor = 1.0 - util.cpuMath.pow(2.0 * step_quality - 1.0, 3.0)
                            pd.parameters.trust_region_radius = pd.parameters.trust_region_radius / util.cpuMath.fmax(min_factor, tmp_factor)
                            pd.parameters.trust_region_radius = util.cpuMath.fmin(pd.parameters.trust_region_radius, max_trust_region_radius)
                            pd.parameters.radius_decrease_factor = 2.0

                            pd.prevCost = newCost
                        else 
                            gpu.revertUpdate(pd)

                            pd.parameters.trust_region_radius = pd.parameters.trust_region_radius / pd.parameters.radius_decrease_factor
                            logSolver(" trust_region_radius=%f \n", pd.parameters.trust_region_radius)
                            pd.parameters.radius_decrease_factor = 2.0 * pd.parameters.radius_decrease_factor
                            if pd.parameters.trust_region_radius <= min_trust_region_radius then
                                logSolver("\nTrust_region_radius is less than the min, exiting\n")
                                cleanup(pd)
                                return 0
                            end
                            logSolver("REVERT\n")
                            gpu.precompute(pd)
                        end
                    end
                else
                    emit quote
                        pd.prevCost = newCost 
                    end
                end 
            end

            --[[ 
            To match CERES we would check for termination:
            iteration_summary_.gradient_max_norm <= options_.gradient_tolerance
            ]]

            pd.solverparameters.nIter = pd.solverparameters.nIter + 1
            return 1
        else
            cleanup(pd)
            return 0
        end
    end

    local terra cost(data_ : &opaque) : double
        var pd = [&PlanData](data_)
        return [double](pd.prevCost)
    end

    local terra initializeSolverParameters(params : &SolverParameters)
        escape
            -- for each value in solver_parameter_defaults, assign to params
            for name,value in pairs(solver_parameter_defaults) do
                local foundVal = false
                -- TODO, more elegant solution to this
                for _,entry in ipairs(SolverParameters.entries) do
                    if entry.field == name then
                        foundVal = true
                        emit quote params.[name] = [entry.type]([value])
                        end
                        break
                    end
                end
                if not foundVal then
                    print("Tried to initialize "..name.." but not found")
                end
            end
        end
    end

    local terra setSolverParameter(data_ : &opaque, name : rawstring, value : &opaque) 
        var pd = [&PlanData](data_)
        var success = false
        escape
            -- Instead of building a table datastructure, 
            -- explicitly emit an if-statement chain for setting the parameter
            for _,entry in ipairs(SolverParameters.entries) do
                emit quote
                    if C.strcmp([entry.field],name)==0 then
                        pd.solverparameters.[entry.field] = @[&entry.type]([value])
                        return
                    end
                end
            end
        end
        logSolver("Warning: tried to set nonexistent solver parameter %s\n", name)
    end

    local terra free(data_ : &opaque)
        var pd = [&PlanData](data_)

        cd(C.cudaFree([&opaque](pd.gpuDims)))
        C.free(pd.cpuDims)

        pd.delta:freeData()
        pd.r:freeData()
        pd.b:freeData()
        pd.Adelta:freeData()
        pd.z:freeData()
        pd.p:freeData()
        pd.Ap_X:freeData()
        pd.CtC:freeData()
        pd.SSq:freeData()
        pd.preconditioner:freeData()
        pd.g:freeData()
        pd.prevX:freeData()

        [util.freePrecomputedImages(`pd.parameters,problemSpec)]

        cd(C.cudaFree([&opaque](pd.scanAlphaNumerator)))
        cd(C.cudaFree([&opaque](pd.scanBetaNumerator)))
        cd(C.cudaFree([&opaque](pd.scanAlphaDenominator)))
        cd(C.cudaFree([&opaque](pd.modelCost)))

        cd(C.cudaFree([&opaque](pd.scratch)))
        cd(C.cudaFree([&opaque](pd.q)))
        
    end

	local terra makePlan(dims : &uint32, maxDimIndex : uint32) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.init,pd.plan.step,pd.plan.cost,pd.plan.setsolverparameter,pd.plan.free = init,step,cost,setSolverParameter,free
		
        var dimByteCount = sizeof(uint32)*maxDimIndex
        C.cudaMalloc([&&opaque](&(pd.gpuDims)), dimByteCount)
        C.cudaMemcpy(pd.gpuDims,dims, dimByteCount, C.cudaMemcpyHostToDevice)
        pd.cpuDims = [&uint32](C.malloc(dimByteCount))
        C.memcpy(pd.cpuDims, dims, dimByteCount)

        var cpuD = pd.cpuDims
        pd.delta:initGPU(cpuD)
		pd.r:initGPU(cpuD)
        pd.b:initGPU(cpuD)
        pd.Adelta:initGPU(cpuD)
		pd.z:initGPU(cpuD)
		pd.p:initGPU(cpuD)
		pd.Ap_X:initGPU(cpuD)
        pd.CtC:initGPU(cpuD)
        pd.SSq:initGPU(cpuD)
		pd.preconditioner:initGPU(cpuD)
		pd.g:initGPU(cpuD)
        pd.prevX:initGPU(cpuD)

        initializeSolverParameters(&pd.solverparameters)
		
		[util.initPrecomputedImages(`pd.parameters,problemSpec,`pd.cpuDims)]	
		C.cudaMalloc([&&opaque](&(pd.scanAlphaNumerator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.scanBetaNumerator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.scanAlphaDenominator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.modelCost)), sizeof(opt_float))
		
		C.cudaMalloc([&&opaque](&(pd.scratch)), sizeof(opt_float))
        C.cudaMalloc([&&opaque](&(pd.q)), sizeof(opt_float))

		return &pd.plan
	end

	return makePlan
end
