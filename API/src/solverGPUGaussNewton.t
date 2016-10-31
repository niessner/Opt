local S = require("std")
local util = require("util")
local dbg = require("dbg")
require("precision")
local C = util.C
local Timer = util.Timer

local getValidUnknown = util.getValidUnknown
local use_dump_j = false

local gpuMath = util.gpuMath

opt.BLOCK_SIZE = 16
local BLOCK_SIZE =  opt.BLOCK_SIZE

local FLOAT_EPSILON = `[opt_float](0.0) 
-- GAUSS NEWTON (non-block version)
return function(problemSpec)
    local UnknownType = problemSpec:UnknownType()
    local TUnknownType = UnknownType:terratype()	
    local imagename_to_unknown_offset = {}
    local energyspec_to_residual_offset_exp = {}
    local energyspec_to_rowidx_offset_exp = {}
    local nUnknowns,nResidualsExp,nnzExp = 0,`0,`0
    local parametersSym = symbol(&problemSpec:ParameterType(),"parameters")
    local function numberofelements(ES)
        if ES.kind.kind == "CenteredFunction" then
            return ES.kind.ispace:cardinality()
        else
            return `parametersSym.[ES.kind.graphname].N
        end
    end
    if problemSpec.energyspecs then
        for i,image in ipairs(UnknownType.images) do
            imagename_to_unknown_offset[image.name] = nUnknowns
            print(("image %s has offset %d"):format(image.name,nUnknowns))
            nUnknowns = nUnknowns + image.imagetype.ispace:cardinality()*image.imagetype.channelcount
        end
        for i,es in ipairs(problemSpec.energyspecs) do
            print("ES",i,nResidualsExp,nnzExp)
            energyspec_to_residual_offset_exp[es] = nResidualsExp
            energyspec_to_rowidx_offset_exp[es] = nnzExp
            
            local residuals_per_element = #es.residuals
            nResidualsExp = `nResidualsExp + [numberofelements(es)]*residuals_per_element
            local nentries = 0
            for i,r in ipairs(es.residuals) do
                nentries = nentries + #r.unknowns
            end
            nnzExp = `nnzExp + [numberofelements(es)]*nentries
        end
        print("nUnknowns = ",nUnknowns)
        print("nResiduals = ",nResidualsExp)
        print("nnz = ",nnzExp)
    end
    
    local isGraph = problemSpec:UsesGraphs() 
    
    local struct PlanData(S.Object) {
		plan : opt.Plan
		parameters : problemSpec:ParameterType()
		scratchF : &opt_float

		delta : TUnknownType	--current linear update to be computed -> num vars
		r : TUnknownType		--residuals -> num vars	--TODO this needs to be a 'residual type'
        b : TUnknownType        --J^TF. Constant during inner iterations, only used to recompute r to counteract drift -> num vars --TODO this needs to be a 'residual type'
        Adelta : TUnknownType       -- (A'A+D'D)delta TODO this needs to be a 'residual type'
		z : TUnknownType		--preconditioned residuals -> num vars	--TODO this needs to be a 'residual type'
		p : TUnknownType		--descent direction -> num vars
		Ap_X : TUnknownType	--cache values for next kernel call after A = J^T x J x p -> num vars
        DtD : TUnknownType -- The diagonal matrix D'D for the inner linear solve (A'A+D'D)x = A'b (A = J, b = residuals). Used only by LM
		preconditioner : TUnknownType --pre-conditioner for linear system -> num vars
		g : TUnknownType		--gradient of F(x): g = -2J'F -> num vars
		
        prevX : TUnknownType -- Place to copy unknowns to before speculatively updating. Avoids hassle when (X + delta) - delta != X 

		scanAlphaNumerator : &opt_float
		scanAlphaDenominator : &opt_float
		scanBetaNumerator : &opt_float
		scanBetaDenominator : &opt_float

        modelCostChange : &opt_float    -- modelCostChange = L(0) - L(delta) where L(h) = F' F + 2 h' J' F + h' J' J h
		
		timer : Timer
		endSolver : util.TimerEvent
		nIter : int				--current non-linear iter counter
		nIterations : int		--non-linear iterations
		lIterations : int		--linear iterations
	    prevCost : opt_float
	    
	    J_values : &opt_float
	    J_colindex : &int
	    J_rowptr : &int
	}
	
    local function generateDumpJ(ES,dumpJ,idx,pd)
        local nnz_per_entry = 0
        for i,r in ipairs(ES.residuals) do
            nnz_per_entry = nnz_per_entry + #r.unknowns
        end
        local base_rowidx = energyspec_to_rowidx_offset_exp[ES]
        local base_residual = energyspec_to_residual_offset_exp[ES]
        local idx_offset
        print(terralib.type(idx))
        if idx.type == int then
            idx_offset = idx
        else    
            idx_offset = `idx:tooffset()
        end
        local local_rowidx = `base_rowidx + idx_offset*nnz_per_entry
        local local_residual = `base_residual + idx_offset*[#ES.residuals]
        local function GetOffset(idx,index)
            if index.kind == "Offset" then
                return `idx([{unpack(index.data)}]):tooffset()
            else
                return `parametersSym.[index.graph.name].[index.element][idx]:tooffset()
            end
        end
        return quote
            var rhs = dumpJ(idx,pd.parameters)
            escape                
                local nnz = 0
                local residual = 0
                for i,r in ipairs(ES.residuals) do
                    emit quote
                        pd.J_rowptr[local_residual+residual] = local_rowidx + nnz
                    end
                    for i,u in ipairs(r.unknowns) do
                        local image_offset = imagename_to_unknown_offset[u.image.name]
                        local nchannels = u.image.type.channelcount
                        local uidx = GetOffset(idx,u.index)
                        local unknown_index = `image_offset + nchannels*uidx + u.channel
                        emit quote
                            pd.J_values[local_rowidx + nnz] = rhs.["_"..tostring(nnz)]
                            pd.J_colindex[local_rowidx + nnz] = unknown_index
                        end
                        nnz = nnz + 1
                    end
                    residual = residual + 1
                    -- write next entry as well to ensure the final entry is correct
                    -- wasteful but less chance for error
                    emit quote
                        pd.J_rowptr[local_residual+residual] = local_rowidx + nnz
                    end
                end
            end
        end
	end
	
	local delegate = {}
	function delegate.CenterFunctions(UnknownIndexSpace,fmap)
	    --print("ES",fmap.derivedfrom)
	    local kernels = {}
	    local unknownElement = UnknownType:VectorTypeForIndexSpace(UnknownIndexSpace)
	    local Index = UnknownIndexSpace:indextype()

        local CERES_style_guardedInvert = true

        local terra square(x : opt_float) : opt_float
            return x*x
        end

        local terra guardedInvert(p : unknownElement)
            escape 
                if CERES_style_guardedInvert then
                    emit quote
                        var invp = p
                        for i = 0, invp:size() do
                            --invp(i) = [opt_float](1.f) / ([opt_float](1.f) + 2*util.gpuMath.sqrt(invp(i)) + invp(i))
                            invp(i) = [opt_float](1.f) / square([opt_float](1.f) + util.gpuMath.sqrt(invp(i)))
                        end
                        return invp
                    end
                else
                    emit quote
                        var invp = p
                        for i = 0, invp:size() do
                            invp(i) = terralib.select(invp(i) > FLOAT_EPSILON, [opt_float](1.f) / invp(i),invp(i))
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
                pd.preconditioner(idx) = pre
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
                tmp = fmap.applyJTJ(idx, pd.parameters, pd.p, pd.DtD)
                pd.Ap_X(idx) = tmp					 -- store for next kernel call
                d = pd.p(idx):dot(tmp)			 -- x-th term of denominator of alpha
            end
            d = util.warpReduce(d)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.scanAlphaDenominator, d)
            end
        end
        terra kernels.PCGStep2(pd : PlanData)
            var b = opt_float(0.0f) 
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                -- sum over block results to compute denominator of alpha
                var alphaDenominator : opt_float = pd.scanAlphaDenominator[0]
                var alphaNumerator : opt_float = pd.scanAlphaNumerator[0]

                -- update step size alpha
                var alpha = opt_float(0.0f)
                if alphaDenominator > FLOAT_EPSILON then 
                    alpha = alphaNumerator/alphaDenominator 
                else
                    --printf("WARNING: Invalid alphaDenominator: %f\n", alphaDenominator)
                end 
    
                pd.delta(idx) = pd.delta(idx)+alpha*pd.p(idx)		-- do a descent step

                var r = pd.r(idx)-alpha*pd.Ap_X(idx)				-- update residuum
                pd.r(idx) = r										-- store for next kernel call
    
                var pre = pd.preconditioner(idx)
                if not problemSpec.usepreconditioner then
                    pre = opt_float(1.0f)
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

        terra kernels.PCGStep2_1stHalf(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                -- sum over block results to compute denominator of alpha
                var alphaDenominator : opt_float = pd.scanAlphaDenominator[0]
                var alphaNumerator : opt_float = pd.scanAlphaNumerator[0]

                -- update step size alpha
                var alpha = opt_float(0.0f)
                if alphaDenominator > FLOAT_EPSILON then 
                    alpha = alphaNumerator/alphaDenominator 
                else
                    --printf("WARNING: Invalid alphaDenominator: %f\n", alphaDenominator)
                end 
    
                pd.delta(idx) = pd.delta(idx)+alpha*pd.p(idx)       -- do a descent step
            end
        end

        terra kernels.PCGStep2_2ndHalf(pd : PlanData)
            var b = opt_float(0.0f) 
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                var r = pd.r(idx)
                var pre = pd.preconditioner(idx)
                if not problemSpec.usepreconditioner then
                    pre = opt_float(1.0f)
                end
                var z = pre*r                                       -- apply pre-conditioner M^-1
                pd.z(idx) = z;                                      -- save for next kernel call
                b = z:dot(r)                                    -- compute x-th term of the numerator of beta
            end
            b = util.warpReduce(b)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.scanBetaNumerator, b)
            end
        end


        terra kernels.PCGStep3(pd : PlanData)			
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
            
                var rDotzNew : opt_float = pd.scanBetaNumerator[0]				-- get new numerator
                var rDotzOld : opt_float = pd.scanAlphaNumerator[0]				-- get old denominator

                var beta : opt_float = [opt_float](0.0f)		                    
                if rDotzOld > FLOAT_EPSILON then 
                    beta = rDotzNew/rDotzOld 
                else
                    --printf("WARNING: Invalid rDotzOld: %f\n", rDotzOld)
                end	-- update step size beta
                pd.p(idx) = pd.z(idx)+beta*pd.p(idx)							-- update decent direction

            end
        end
    
        terra kernels.PCGLinearUpdate(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.parameters.X(idx) = pd.parameters.X(idx) + pd.delta(idx)
                --printf("delta*10000: %f %f %f\n", pd.delta(idx)(0)*10000, pd.delta(idx)(1)*10000, pd.delta(idx)(2)*10000)            
            end
        end	
        
        terra kernels.revertUpdate(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.parameters.X(idx) = pd.prevX(idx)
            end
        end	

        terra kernels.copyResidualsToB(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.b(idx) = pd.r(idx)
            end
        end

        terra kernels.computeAdelta(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.Adelta(idx) = fmap.applyJTJ(idx, pd.parameters, pd.delta, pd.DtD)
            end
        end

        terra kernels.recomputeResiduals(pd : PlanData)
            var d = 0.0f
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                var Ax = pd.Adelta(idx)
                var b = pd.b(idx)
                var newR = b - Ax
                --var diff = newR - pd.r(idx)
                --printf("\ndiff*10000: %f %f %f\n", diff(0)*10000, diff(1)*10000, diff(2)*10000)
                pd.r(idx) = newR
            end
        end

        terra kernels.savePreviousUnknowns(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.prevX(idx) = pd.parameters.X(idx)
            end
        end 

        terra kernels.computeCost(pd : PlanData)			
            var cost : opt_float = [opt_float](0.0f)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                var params = pd.parameters				
                cost = cost + [opt_float](fmap.cost(idx, params))
            end

            cost = util.warpReduce(cost)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.scratchF, cost)
            end
        end
        if not fmap.dumpJ then
            terra kernels.saveJToCRS(pd : PlanData)
            end
        else
            terra kernels.saveJToCRS(pd : PlanData)
                var idx : Index
                var [parametersSym] = &pd.parameters
                if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                    [generateDumpJ(fmap.derivedfrom,fmap.dumpJ,idx,pd)]
                end
            end
            print(kernels.saveJToCRS)
        end
    

        if fmap.precompute then
            terra kernels.precompute(pd : PlanData)
                var idx : Index
                if idx:initFromCUDAParams() then
                   fmap.precompute(idx,pd.parameters)
                end
            end
        end
        if problemSpec:UsesLambda() then
            terra kernels.PCGComputeDtD(pd : PlanData)
                var idx : Index
                if idx:initFromCUDAParams() then                         
                    if not fmap.exclude(idx,pd.parameters) then 
                        var DtD = fmap.computeDtD(idx, pd.parameters, pd.preconditioner)
                        pd.DtD(idx) = DtD
                    end        
                end 
            end


            terra kernels.PCGClampDiagonal(pd : PlanData)
                var idx : Index
                if idx:initFromCUDAParams() then
                    if not fmap.exclude(idx,pd.parameters) then 
                        var DtD = pd.DtD(idx)        
                        var invS_iiSq = 1.0 / pd.preconditioner(idx)
                        var minVal = square(pd.parameters.min_lm_diagonal) * invS_iiSq
                        var maxVal = square(pd.parameters.max_lm_diagonal) * invS_iiSq
                        pd.DtD(idx) = clamp(DtD, minVal, maxVal)
                    end        
                end 
            end

            terra kernels.DebugDumpDtD(pd : PlanData)

                var idx : Index
                if idx:initFromCUDAParams() then                         
                    var CtC : unknownElement = pd.DtD(idx)
                    printf("\nC'C: %d: %f %f %f\n", idx, CtC(0), CtC(1), CtC(2))
                    var JtJ_ii : unknownElement = pd.DtD(idx) * pd.parameters.trust_region_radius
                    printf("\nJ'J_ii: %d, %f %f %f\n", idx, JtJ_ii(0), JtJ_ii(1), JtJ_ii(2))
                    var DtD : unknownElement = CtC * pd.preconditioner(idx)
                    printf("\nD'D: %d: %f %f %f\n", idx, DtD(0), DtD(1), DtD(2))
                    var M1 : unknownElement = pd.preconditioner(idx)
                    printf("\nE^-1*100: %d: %f %f %f\n", idx, util.gpuMath.sqrt(M1(0))*100, util.gpuMath.sqrt(M1(1))*100, util.gpuMath.sqrt(M1(2))*100)
                    printf("\nM^-1*10000: %d: %f %f %f\n", idx, M1(0)*10000, M1(1)*10000, M1(2)*10000)
                end 
            end

            terra kernels.computeModelCost(pd : PlanData)            
                var cost : opt_float = [opt_float](0.0f)
                var idx : Index
                if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                    var params = pd.parameters              
                    cost = cost + [opt_float](fmap.modelcost(idx, params, pd.delta))
                end

                cost = util.warpReduce(cost)
                if (util.laneid() == 0) then
                    util.atomicAdd(pd.scratchF, cost)
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
                fmap.evalJTF(tIdx, pd.parameters, pd.r, pd.preconditioner)
            end
        end    
        
    	terra kernels.PCGStep1_Graph(pd : PlanData)
            var d = 0.0f
            var tIdx = 0 
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
               d = d + fmap.applyJTJ(tIdx, pd.parameters, pd.p, pd.Ap_X, pd.DtD)
            end 
            d = util.warpReduce(d)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.scanAlphaDenominator, d)
            end
        end

        terra kernels.computeAdelta_Graph(pd : PlanData)
            var tIdx = 0 
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
                fmap.applyJTJ(tIdx, pd.parameters, pd.delta, pd.Adelta, pd.DtD)
            end
        end

        terra kernels.computeCost_Graph(pd : PlanData)			
            var cost : opt_float = [opt_float](0.0f)
            var tIdx = 0
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
                cost = cost + fmap.cost(tIdx, pd.parameters)
            end 
            cost = util.warpReduce(cost)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.scratchF, cost)
            end
        end
        if not fmap.dumpJ then
            terra kernels.saveJToCRS_Graph(pd : PlanData)
            end
        else
            terra kernels.saveJToCRS_Graph(pd : PlanData)
                var tIdx = 0
                var [parametersSym] = &pd.parameters
                if util.getValidGraphElement(pd,[graphname],&tIdx) then
                    [generateDumpJ(fmap.derivedfrom,fmap.dumpJ,tIdx,pd)]
                end
            end
            print(kernels.saveJToCRS_Graph)
        end
        if problemSpec:UsesLambda() then
            terra kernels.PCGComputeDtD_Graph(pd : PlanData)
                var tIdx = 0
                if util.getValidGraphElement(pd,[graphname],&tIdx) then
                    fmap.computeDtD(tIdx, pd.parameters, pd.DtD, pd.preconditioner)
                end
            end    

            terra kernels.computeModelCost_Graph(pd : PlanData)          
                var cost : opt_float = [opt_float](0.0f)
                var tIdx = 0
                if util.getValidGraphElement(pd,[graphname],&tIdx) then
                    cost = cost + fmap.modelcost(tIdx, pd.parameters, pd.delta)
                end 
                cost = util.warpReduce(cost)
                if (util.laneid() == 0) then
                    util.atomicAdd(pd.scratchF, cost)
                end
            end
        end

	    return kernels
	end
	
	local gpu = util.makeGPUFunctions(problemSpec, PlanData, delegate, {"PCGInit1",
                                                                        "PCGInit1_Finish",
                                                                        "PCGComputeDtD",
                                                                        "PCGClampDiagonal",
                                                                        "PCGStep1",
                                                                        "PCGStep2",
                                                                        "PCGStep2_1stHalf",
                                                                        "PCGStep2_2ndHalf",
                                                                        "PCGStep3",
                                                                        "PCGLinearUpdate",
                                                                        "revertUpdate",
                                                                        "savePreviousUnknowns",
                                                                        "computeCost",
                                                                        "precompute",
                                                                        "copyResidualsToB",
                                                                        "computeAdelta",
                                                                        "computeAdelta_Graph",
                                                                        "recomputeResiduals",
                                                                        "PCGInit1_Graph",
                                                                        "PCGComputeDtD_Graph",
                                                                        "PCGStep1_Graph",
                                                                        "computeCost_Graph",
                                                                        "computeModelCost",
                                                                        "computeModelCost_Graph",
                                                                        "saveJToCRS",
                                                                        "DebugDumpDtD",
                                                                        "saveJToCRS_Graph"
                                                                        })

    local terra computeCost(pd : &PlanData) : opt_float
        C.cudaMemset(pd.scratchF, 0, sizeof(opt_float))
        gpu.computeCost(pd)
        gpu.computeCost_Graph(pd)
        var f : opt_float
        C.cudaMemcpy(&f, pd.scratchF, sizeof(opt_float), C.cudaMemcpyDeviceToHost)
        return f
    end

    local terra computeModelCost(pd : &PlanData) : opt_float
        C.cudaMemset(pd.scratchF, 0, sizeof(opt_float))
        gpu.computeModelCost(pd)
        gpu.computeModelCost_Graph(pd)
        var f : opt_float
        C.cudaMemcpy(&f, pd.scratchF, sizeof(opt_float), C.cudaMemcpyDeviceToHost)
        return f
    end

    local initLambda,computeModelCostChange
    
    if problemSpec:UsesLambda() then
        
        terra computeModelCostChange(pd : &PlanData) : opt_float
            var cost = computeCost(pd)
            var model_cost = computeModelCost(pd)
            logSolver(" cost=%f \n",cost)
            logSolver(" model_cost=%f \n",model_cost)
            var model_cost_change = cost - model_cost
            --logSolver(" model_cost_change=%f \n",model_cost_change)
            return model_cost_change
        end

        terra initLambda(pd : &PlanData)
            pd.parameters.trust_region_radius = 1e4

            --[[
            C.cudaMemset(pd.maxDiagJTJ, 0, sizeof(opt_float))

            -- lambda = tau * max{a_ii} where A = JTJ
            var tau = 1e-6f
            gpu.LMLambdaInit(pd);
            var maxDiagJTJ : opt_float
            C.cudaMemcpy(&maxDiagJTJ, pd.maxDiagJTJ, sizeof(opt_float), C.cudaMemcpyDeviceToHost)
            pd.parameters.lambda = tau * maxDiagJTJ
        --]]
        end

    end

	local terra init(data_ : &opaque, params_ : &&opaque, solverparams : &&opaque)
	   var pd = [&PlanData](data_)
	   pd.timer:init()
	   pd.timer:startEvent("overall",nil,&pd.endSolver)
       [util.initParameters(`pd.parameters,problemSpec,params_,true)]
       var [parametersSym] = &pd.parameters
        if use_dump_j and pd.J_values == nil then
            logSolver("nnz = %s\n",[tostring(nnzExp)])
            logSolver("nResiduals = %s\n",[tostring(nResidualsExp)])
            logSolver("nnz = %d, nResiduals = %d\n",int(nnzExp),int(nResidualsExp))
            C.cudaMalloc([&&opaque](&(pd.J_values)), sizeof(opt_float)*nnzExp)
            C.cudaMalloc([&&opaque](&(pd.J_colindex)), sizeof(int)*nnzExp)
            C.cudaMalloc([&&opaque](&(pd.J_rowptr)), sizeof(int)*(nResidualsExp+1))
        end
	   pd.nIter = 0
	   pd.nIterations = @[&int](solverparams[0])
	   pd.lIterations = @[&int](solverparams[1])
       escape 
            if problemSpec:UsesLambda() then
              emit quote 
                initLambda(pd)
                pd.parameters.radius_decrease_factor = 2.0
                pd.parameters.min_lm_diagonal = 1e-6;
                pd.parameters.max_lm_diagonal = 1e32;
              end
	        end 
       end
	   gpu.precompute(pd)
	   pd.prevCost = computeCost(pd)
	end

	
	local terra step(data_ : &opaque, params_ : &&opaque, solverparams : &&opaque)


        --TODO: make parameters
        var residual_reset_period = 10
        var min_relative_decrease = 1e-3f
        var min_trust_region_radius = 1e-32;
        var max_trust_region_radius = 1e16;

		var pd = [&PlanData](data_)
		[util.initParameters(`pd.parameters,problemSpec, params_,false)]
		if pd.nIter < pd.nIterations then
			C.cudaMemset(pd.scanAlphaNumerator, 0, sizeof(opt_float))	--scan in PCGInit1 requires reset
			C.cudaMemset(pd.scanAlphaDenominator, 0, sizeof(opt_float))	--scan in PCGInit1 requires reset
			C.cudaMemset(pd.scanBetaNumerator, 0, sizeof(opt_float))	--scan in PCGInit1 requires reset
			C.cudaMemset(pd.scanBetaDenominator, 0, sizeof(opt_float))	--scan in PCGInit1 requires reset

			gpu.PCGInit1(pd)
			--gpu.saveJToCRS(pd)
			if isGraph then
				gpu.PCGInit1_Graph(pd)	
				gpu.PCGInit1_Finish(pd)	
			end
            escape 
                if problemSpec:UsesLambda() then
                    emit quote
                        --logSolver(" trust_region_radius=%f ",pd.parameters.trust_region_radius)
                        gpu.PCGComputeDtD(pd)
                        gpu.PCGComputeDtD_Graph(pd)
                        gpu.PCGClampDiagonal(pd)
                        --gpu.DebugDumpDtD(pd)
                        gpu.copyResidualsToB(pd)
                        --gpu.recomputeResiduals(pd)
                    end
                end
            end
            if use_dump_j then
                logSolver("saving J\n")
                gpu.saveJToCRS(pd)
                if isGraph then
                    gpu.saveJToCRS_Graph(pd)
                end
                logSolver("saving J2\n")
            end
            for lIter = 0, pd.lIterations do				
--[[
                var v : double[3]

                C.cudaMemcpy(&v, pd.r.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\nlinIter: %d\n\t r %f %f %f\n", lIter, v[0], v[1], v[2])

                C.cudaMemcpy(&v, pd.z.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t z %f %f %f\n", v[0], v[1], v[2])

                C.cudaMemcpy(&v, pd.p.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t p %f %f %f\n", v[0], v[1], v[2])

                C.cudaMemcpy(&v, pd.Ap_X.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t Ap_X %f %f %f\n", v[0], v[1], v[2])


                C.cudaMemcpy(&v, pd.delta.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\tdelta %f %f %f\n", v[0], v[1], v[2])

                C.cudaMemcpy(&v, pd.scanAlphaNumerator, sizeof(double), C.cudaMemcpyDeviceToHost)
                logSolver("\tscanAlphaNumerator %f\n", v[0])
                C.cudaMemcpy(&v, pd.scanAlphaDenominator, sizeof(double), C.cudaMemcpyDeviceToHost)
                logSolver("\tscanAlphaDenominator %f\n", v[0])

                var numerator : double
                C.cudaMemcpy(&numerator, pd.scanAlphaNumerator, sizeof(double), C.cudaMemcpyDeviceToHost)
                logSolver("\talpha %f\n", numerator/v[0])


                C.cudaMemcpy(&v, pd.scanBetaNumerator, sizeof(double), C.cudaMemcpyDeviceToHost)
                logSolver("\tscanBetaNumerator %f\n", v[0])
--]]


                C.cudaMemset(pd.scanAlphaDenominator, 0, sizeof(opt_float))

				gpu.PCGStep1(pd)

				if isGraph then
					gpu.PCGStep1_Graph(pd)	
				end
				
				C.cudaMemset(pd.scanBetaNumerator, 0, sizeof(opt_float))
				
				

                if [problemSpec:UsesLambda()] and ((lIter + 1) % residual_reset_period) == 0 then
                    gpu.PCGStep2_1stHalf(pd)
                    gpu.computeAdelta(pd)
                    -- TODO: merge these?
                    gpu.computeAdelta_Graph(pd)
                    --TODO: Should we redo then end of step 2 (thats where the residual is changed regularly)
                    gpu.recomputeResiduals(pd)
                    gpu.PCGStep2_2ndHalf(pd)
                else
                    gpu.PCGStep2(pd)
                end
                gpu.PCGStep3(pd)


				-- save new rDotz for next iteration
				C.cudaMemcpy(pd.scanAlphaNumerator, pd.scanBetaNumerator, sizeof(opt_float), C.cudaMemcpyDeviceToDevice)	
				
			end
			

            var model_cost_change : opt_float
            escape 
                if problemSpec:UsesLambda() then
                    emit quote
                        model_cost_change = computeModelCostChange(pd)
                        gpu.savePreviousUnknowns(pd)
                    end
                end
            end

			gpu.PCGLinearUpdate(pd)    
			gpu.precompute(pd)
			var newCost = computeCost(pd)
			
			logSolver("\t%d: prev=%f new=%f ", pd.nIter, pd.prevCost,newCost)
			
            -- TODO: Remove
            if newCost > 1000000 then
                logSolver("Shitty cost!")

                
                var v : double[3]

                C.cudaMemcpy(&v, pd.r.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t r %f %f %f\n", v[0], v[1], v[2])

                C.cudaMemcpy(&v, pd.z.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t z %f %f %f\n", v[0], v[1], v[2])

                C.cudaMemcpy(&v, pd.p.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t p %f %f %f\n", v[0], v[1], v[2])

                C.cudaMemcpy(&v, pd.Ap_X.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t Ap_X %f %f %f\n", v[0], v[1], v[2])

                C.cudaMemcpy(&v, pd.preconditioner.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t preconditioner %f %f %f\n", v[0], v[1], v[2])

                C.cudaMemcpy(&v, pd.delta.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\tdelta %f %f %f\n", v[0], v[1], v[2])

                C.cudaMemcpy(&v, pd.parameters.X.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\tunknown %f %f %f\n", v[0], v[1], v[2])
                

                C.cudaMemcpy(&v, pd.scanAlphaNumerator, sizeof(double), C.cudaMemcpyDeviceToHost)
                logSolver("\tscanAlphaNumerator %f\n", v[0])
                C.cudaMemcpy(&v, pd.scanAlphaDenominator, sizeof(double), C.cudaMemcpyDeviceToHost)
                logSolver("\tscanAlphaDenominator %f\n", v[0])

                var numerator : double
                C.cudaMemcpy(&numerator, pd.scanAlphaNumerator, sizeof(double), C.cudaMemcpyDeviceToHost)
                logSolver("\talpha %f\n", numerator/v[0])


                C.cudaMemcpy(&v, pd.scanBetaNumerator, sizeof(double), C.cudaMemcpyDeviceToHost)
                logSolver("\tscanBetaNumerator %f\n", v[0])
                --]]
                return 0
            end

			escape 
                if problemSpec:UsesLambda() then
                    emit quote
                        logSolver(" trust_region_radius=%f ",pd.parameters.trust_region_radius)

                        var cost_change = pd.prevCost - newCost
                        
                        
                        -- See TrustRegionStepEvaluator::StepAccepted() for a more complicated version of this
                        var relative_decrease = cost_change / model_cost_change

                        logSolver(" cost_change=%f ", cost_change)
                        logSolver(" model_cost_change=%f ", model_cost_change)
                        logSolver(" relative_decrease=%f ", relative_decrease)

                        if cost_change >= 0 and relative_decrease > min_relative_decrease then	--in this case we revert
                            --[[
                                radius_ = radius_ / std::max(1.0 / 3.0,
                                                           1.0 - pow(2.0 * step_quality - 1.0, 3));
                                radius_ = std::min(max_radius_, radius_);
                                decrease_factor_ = 2.0;
                            --]]
                            var step_quality = relative_decrease
                            var min_factor = 1.0/3.0
                            var tmp_factor = 1.0 - util.cpuMath.pow(2.0 * step_quality - 1.0, 3.0)
                            pd.parameters.trust_region_radius = pd.parameters.trust_region_radius / util.cpuMath.fmax(min_factor, tmp_factor)
                            pd.parameters.trust_region_radius = util.cpuMath.fmin(pd.parameters.trust_region_radius, max_trust_region_radius)
                            pd.parameters.radius_decrease_factor = 2.0

                            logSolver("\n")
                            pd.prevCost = newCost
                        else 
                            gpu.revertUpdate(pd)

                            pd.parameters.trust_region_radius = pd.parameters.trust_region_radius / pd.parameters.radius_decrease_factor
                            pd.parameters.radius_decrease_factor = 2.0f * pd.parameters.radius_decrease_factor
                            if pd.parameters.trust_region_radius <= min_trust_region_radius then
                                logSolver("\nTrust_region_radius is less than the min, exiting\n")
                                logSolver("final cost=%f\n", pd.prevCost)
                                pd.timer:endEvent(nil,pd.endSolver)
                                pd.timer:evaluate()
                                pd.timer:cleanup()
                                return 0
                            end
                            logSolver("REVERT\n")
                            gpu.precompute(pd)
                        end
                    end
                else
                    emit quote
                        logSolver("\n")
                        pd.prevCost = newCost 
                    end
                end 
            end

            --[[ 
            To match CERES we would check for termination:
            iteration_summary_.gradient_max_norm <= options_.gradient_tolerance
            iteration_summary_.trust_region_radius <= options_.min_trust_region_radius
            ]]

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

    local terra cost(data_ : &opaque) : float
        var pd = [&PlanData](data_)
        return [float](pd.prevCost)
    end

	local terra makePlan() : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.init,pd.plan.step,pd.plan.cost = init,step,cost

		pd.delta:initGPU()
		pd.r:initGPU()
        pd.b:initGPU()
        pd.Adelta:initGPU()
		pd.z:initGPU()
		pd.p:initGPU()
		pd.Ap_X:initGPU()
        pd.DtD:initGPU()
		pd.preconditioner:initGPU()
		pd.g:initGPU()
        pd.prevX:initGPU()
		
		[util.initPrecomputedImages(`pd.parameters,problemSpec)]	
		C.cudaMalloc([&&opaque](&(pd.scanAlphaNumerator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.scanBetaNumerator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.scanAlphaDenominator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.scanBetaDenominator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.modelCostChange)), sizeof(opt_float))
		
		C.cudaMalloc([&&opaque](&(pd.scratchF)), sizeof(opt_float))
		pd.J_values = nil
		return &pd.plan
	end
	return makePlan
end
