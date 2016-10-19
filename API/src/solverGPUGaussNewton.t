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

local FLOAT_EPSILON = `[opt_float](0.000001) 
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
		z : TUnknownType		--preconditioned residuals -> num vars	--TODO this needs to be a 'residual type'
		p : TUnknownType		--descent direction -> num vars
		Ap_X : TUnknownType	--cache values for next kernel call after A = J^T x J x p -> num vars
        DtD : TUnknownType -- The diagonal matrix D'D for the inner linear solve (A'A+D'D)x = A'b (A = J, b = residuals). Used only by LM
		preconditioner : TUnknownType --pre-conditioner for linear system -> num vars
		g : TUnknownType		--gradient of F(x): g = -2J'F -> num vars
		
		scanAlphaNumerator : &opt_float
		scanAlphaDenominator : &opt_float
		scanBetaNumerator : &opt_float
		scanBetaDenominator : &opt_float

        modelCostChange : &opt_float    -- modelCostChange = L(0) - L(delta) where L(h) = F' F + 2 h' J' F + h' J' J h
        maxDiagJTJ : &opt_float    -- maximum value in the diagonal of JTJ 
		
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
        local terra guardedInvert(p : unknownElement)
            escape 
                if CERES_style_guardedInvert then
                    emit quote
                        var invp = p
                        for i = 0, invp:size() do
                            invp(i) = [opt_float](1.f) / ([opt_float](1.f) + 2*util.gpuMath.sqrt(invp(i)) + invp(i))
                            --invp(i) = [opt_float](1.f) / ([opt_float](1.f) + invp(i))
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
        terra kernels.PCGStep3(pd : PlanData)			
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
            
                var rDotzNew : opt_float = pd.scanBetaNumerator[0]				-- get new numerator
                var rDotzOld : opt_float = pd.scanAlphaNumerator[0]				-- get old denominator

                var beta : opt_float = [opt_float](0.0f)		                    
                if rDotzOld > FLOAT_EPSILON then beta = rDotzNew/rDotzOld end	-- update step size beta
                pd.p(idx) = pd.z(idx)+beta*pd.p(idx)							-- update decent direction

            end
        end
    
        terra kernels.PCGLinearUpdate(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.parameters.X(idx) = pd.parameters.X(idx) + pd.delta(idx)
                printf("delta: %f %f\n", pd.delta(idx)(0), pd.delta(idx)(1))
            end
        end	
        
        terra kernels.PCGLinearUpdateRevert(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.parameters.X(idx) = pd.parameters.X(idx) - pd.delta(idx)
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
                    -- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0                            
                    var DtD : unknownElement = 0.0f
                
                    if not fmap.exclude(idx,pd.parameters) then 
                        pd.delta(idx) = 0.0f    
                        DtD = fmap.computeDtD(idx, pd.parameters, pd.preconditioner)
                        pd.DtD(idx) = DtD
                    end        
                end 
            end

            terra kernels.DebugDumpDtD(pd : PlanData)
                var idx : Index
                if idx:initFromCUDAParams() then                         
                    var DtD : unknownElement = pd.DtD(idx)
                    printf("\nD: %d: %f %f\n", idx, DtD(0), DtD(1))
                end 
            end

            --[[
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
            --]]
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
   --[[
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
                    --]]
        end

	    return kernels
	end
	
	local gpu = util.makeGPUFunctions(problemSpec, PlanData, delegate, {"PCGInit1",
                                                                        "PCGInit1_Finish",
                                                                        "PCGComputeDtD",
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
                                                                        "PCGComputeDtD_Graph",
                                                                        "PCGStep1_Graph",
                                                                        "computeCost_Graph",
                                                                        "computeModelCostChangeStep1_Graph",
                                                                        "computeModelCostChangeStep2_Graph",
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

    local initLambda,computeModelCostChange
    
    if problemSpec:UsesLambda() then
        --[[
        terra computeModelCostChange(pd : &PlanData) : opt_float
            C.cudaMemset(pd.modelCostChange, 0, sizeof(opt_float))

            -- model_cost_change = delta' * (-2 * J' * F) + delta' * (-J' * J * delta)
            gpu.computeModelCostChangeStep1(pd)
            gpu.computeModelCostChangeStep2(pd)
            if isGraph then
                gpu.computeModelCostChangeStep1_Graph(pd)	
                gpu.computeModelCostChangeStep1_Finish(pd)
                gpu.computeModelCostChangeStep2_Graph(pd)
            end

            var model_cost_change : opt_float
            C.cudaMemcpy(&model_cost_change, pd.modelCostChange, sizeof(opt_float), C.cudaMemcpyDeviceToHost)

            return model_cost_change
        end
                    --]]
        terra initLambda(pd : &PlanData)
            pd.parameters.trust_region_radius = 1e-16
            -- TODO: remove. Just for testing
            pd.parameters.trust_region_radius = 10
            --pd.parameters.trust_region_radius = 0.33333333333333333

            -- Init lambda based on the maximum value on the diagonal of JTJ
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
    	       --   emit quote pd.parameters.lambda = 0.001f end
              emit quote 
                initLambda(pd)
              end
              emit quote pd.parameters.radius_decrease_factor = 2.0f end
	        end 
       end
	   gpu.precompute(pd)
	   pd.prevCost = computeCost(pd)
	end
	
	local terra step(data_ : &opaque, params_ : &&opaque, solverparams : &&opaque)
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
                        logSolver(" trust_region_radius=%f ",pd.parameters.trust_region_radius)
                        gpu.PCGComputeDtD(pd)
                        gpu.PCGComputeDtD_Graph(pd)
                        gpu.DebugDumpDtD(pd)
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

                C.cudaMemset(pd.scanAlphaDenominator, 0, sizeof(opt_float))
				
				gpu.PCGStep1(pd)

				if isGraph then
					gpu.PCGStep1_Graph(pd)	
				end
				
				C.cudaMemset(pd.scanBetaNumerator, 0, sizeof(opt_float))
				
				gpu.PCGStep2(pd)
				gpu.PCGStep3(pd)

				-- save new rDotz for next iteration
				C.cudaMemcpy(pd.scanAlphaNumerator, pd.scanBetaNumerator, sizeof(opt_float), C.cudaMemcpyDeviceToDevice)	
				
			end
			
			gpu.PCGLinearUpdate(pd)    
			gpu.precompute(pd)
			var newCost = computeCost(pd)
			
			logSolver("\t%d: prev=%f new=%f ", pd.nIter, pd.prevCost,newCost)
			
			escape 
                if problemSpec:UsesLambda() then
                --[[
                
                  // Compute a scaling vector that is used to improve the
                  // conditioning of the Jacobian.
                  //
                  // jacobian_scaling_ = diag(J'J)^{-1}
                  jacobian_->SquaredColumnNorm(jacobian_scaling_.data());
                  for (int i = 0; i < jacobian_->num_cols(); ++i) {
                    // Add one to the denominator to prevent division by zero.
                    jacobian_scaling_[i] = 1.0 / (1.0 + sqrt(jacobian_scaling_[i]));
                  }


                    // A linear operator which takes a matrix A and a diagonal vector D and
                    // performs products of the form
                    //
                    //   (A^T A + D^T D)x
                    //
                    // This is used to implement iterative general sparse linear solving with
                    // conjugate gradients, where A is the Jacobian and D is a regularizing
                    // parameter. A brief proof that D^T D is the correct regularizer:
                    //
                    // Given a regularized least squares problem:
                    //
                    //   min  ||Ax - b||^2 + ||Dx||^2
                    //    x
                    //
                    // First expand into matrix notation:
                    //
                    //   (Ax - b)^T (Ax - b) + xD^TDx
                    //
                    // Then multiply out to get:
                    //
                    //   = xA^TAx - 2b^T Ax + b^Tb + xD^TDx
                    //
                    // Take the derivative:
                    //
                    //   0 = 2A^TAx - 2A^T b + 2 D^TDx
                    //   0 = A^TAx - A^T b + D^TDx
                    //   0 = (A^TA + D^TD)x - A^T b
                    //
                    // Thus, the symmetric system we need to solve for CGNR is
                    //
                    //   Sx = z
                    //
                    // with S = A^TA + D^TD
                    //  and z = A^T b

                Loop

                      

                        // jacobian = jacobian * diag(J'J) ^{-1}
                        jacobian_->ScaleColumns(jacobian_scaling_.data());



                    ComputeTrustRegionStep()

                        jacobian->SquaredColumnNorm(diagonal_.data());
                        for (int i = 0; i < num_parameters; ++i) {
                          diagonal_[i] = std::min(std::max(diagonal_[i], min_diagonal_),
                                                  max_diagonal_);
                        }
                        lm_diagonal_ = (diagonal_ / radius_).array().sqrt();
                        solve_options.D = lm_diagonal_.data();
                        q_tolerance = eta = 0.1 // Used for termination in the CG steps 

            
                        LinearSolver::Summary linear_solver_summary =
                              linear_solver_->Solve(jacobian, residuals, solve_options, step);
                            preconditioner_->Update(*A, per_solve_options.D);
                            // A = jacobian, b = residuals, x = step
                            // Solve (AtA + DtD)x = z (= Atb).
                        step *= -1.0f;
                        // trust_region_step_ = step
                        
                          // new_model_cost
                          //  = 1/2 [f + J * step]^2
                          //  = 1/2 [ f'f + 2f'J * step + step' * J' * J * step ]
                          // model_cost_change
                          //  = cost - new_model_cost
                          //  = f'f/2  - 1/2 [ f'f + 2f'J * step + step' * J' * J * step]
                          //  = -f'J * step - step' * J' * J * step / 2
                          //  = -(J * step)'(f + J * step / 2)
                          model_residuals_.setZero();
                          jacobian_->RightMultiply(trust_region_step_.data(), model_residuals_.data());
                          model_cost_change_ =
                              -model_residuals_.dot(residuals_ + model_residuals_ / 2.0);

                          // TODO(sameeragarwal)
                          //
                          //  1. What happens if model_cost_change_ = 0
                          //  2. What happens if -epsilon <= model_cost_change_ < 0 for some
                          //     small epsilon due to round off error.
                          iteration_summary_.step_is_valid = (model_cost_change_ > 0.0);
                          if (iteration_summary_.step_is_valid) {
                            // Undo the Jacobian column scaling.
                            delta_ = (trust_region_step_.array() * jacobian_scaling_.array()).matrix();
                            num_consecutive_invalid_steps_ = 0;
                          }

                    ComputeCandidatePointAndEvaluateCost()
                        This is our ApplyLinearUpdate, but stores result in "candidate" result

                    Two termination checks
                    ParameterToleranceReached() {
                      // Compute the norm of the step in the ambient space.
                      iteration_summary_.step_norm = (x_ - candidate_x_).norm();
                          options_.parameter_tolerance * (x_norm_ + options_.parameter_tolerance);

                      if (iteration_summary_.step_norm > step_size_tolerance) {
                        return false;
                      }
                    FunctionToleranceReached() {
                      iteration_summary_.cost_change = x_cost_ - candidate_cost_;
                      const double absolute_function_tolerance =
                          options_.function_tolerance * x_cost_;

                      if (fabs(iteration_summary_.cost_change) > absolute_function_tolerance) {
                        return false;
                      }


                    IsStepSuccessful() {
                      iteration_summary_.relative_decrease =
                          step_evaluator_->StepQuality(candidate_cost_, model_cost_change_);

                            double TrustRegionStepEvaluator::StepQuality(
                                const double cost,
                                const double model_cost_change) const {
                              const double relative_decrease = (current_cost_ - cost) / model_cost_change;
                              const double historical_relative_decrease =
                                  (reference_cost_ - cost) /
                                  (accumulated_reference_model_cost_change_ + model_cost_change);
                              return std::max(relative_decrease, historical_relative_decrease);
                            }

                      return iteration_summary_.relative_decrease > options_.min_relative_decrease;
                    
                    HandleSuccessfulStep() {
                      x_ = candidate_x_;
                      x_norm_ = x_.norm();

                      if (!EvaluateGradientAndJacobian()) {
                        return false;
                      }

                      iteration_summary_.step_is_successful = true;
                      strategy_->StepAccepted(iteration_summary_.relative_decrease);
                      step_evaluator_->StepAccepted(candidate_cost_, model_cost_change_);
                      return true;
                    }
                        void LevenbergMarquardtStrategy::StepAccepted(double step_quality) {
                          CHECK_GT(step_quality, 0.0);
                          radius_ = radius_ / std::max(1.0 / 3.0,
                                                       1.0 - pow(2.0 * step_quality - 1.0, 3));
                          radius_ = std::min(max_radius_, radius_);
                          decrease_factor_ = 2.0;
                          reuse_diagonal_ = false;
                        }

                        void TrustRegionStepEvaluator::StepAccepted(
                            const double cost,
                            const double model_cost_change) {
                          // Algorithm 10.1.2 from Trust Region Methods by Conn, Gould &
                          // Toint.
                          //
                          // Step 3a
                          current_cost_ = cost;
                          accumulated_candidate_model_cost_change_ += model_cost_change;
                          accumulated_reference_model_cost_change_ += model_cost_change;

                          // Step 3b.
                          if (current_cost_ < minimum_cost_) {
                            minimum_cost_ = current_cost_;
                            num_consecutive_nonmonotonic_steps_ = 0;
                            candidate_cost_ = current_cost_;
                            accumulated_candidate_model_cost_change_ = 0.0;
                          } else {
                            // Step 3c.
                            ++num_consecutive_nonmonotonic_steps_;
                            if (current_cost_ > candidate_cost_) {
                              candidate_cost_ = current_cost_;
                              accumulated_candidate_model_cost_change_ = 0.0;
                            }
                          }

                          // Step 3d.
                          //
                          // At this point we have made too many non-monotonic steps and
                          // we are going to reset the value of the reference iterate so
                          // as to force the algorithm to descend.
                          //
                          // Note: In the original algorithm by Toint, this step was only
                          // executed if the step was non-monotonic, but that would not handle
                          // the case of max_consecutive_nonmonotonic_steps = 0. The small
                          // modification of doing this always handles that corner case
                          // correctly.
                          if (num_consecutive_nonmonotonic_steps_ ==
                              max_consecutive_nonmonotonic_steps_) {
                            reference_cost_ = candidate_cost_;
                            accumulated_reference_model_cost_change_ =
                                accumulated_candidate_model_cost_change_;
                          }
                        }

                    HandleUnsuccessfulStep() {
                      iteration_summary_.step_is_successful = false;
                      strategy_->StepRejected(iteration_summary_.relative_decrease);
                      iteration_summary_.cost = candidate_cost_ + solver_summary_->fixed_cost;
                    }
                        void LevenbergMarquardtStrategy::StepRejected(double step_quality) {
                          radius_ = radius_ / decrease_factor_;
                          decrease_factor_ *= 2.0;
                          reuse_diagonal_ = true;
                        }


                --]]




			    -- lm version

                --[[
                    logSolver(" trust_region_radius=%f ",pd.parameters.trust_region_radius)

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
                        pd.parameters.trust_region_radius = pd.parameters.trust_region_radius / pd.parameters.radius_decrease_factor
                        -- nu = 2 * nu
                        pd.parameters.radius_decrease_factor = 2.0f * pd.parameters.radius_decrease_factor

                        logSolver("REVERT\n")
                        gpu.precompute(pd)
                    else 
                        --]]


                        --FOR NOW, ASSUME ALWAYS ACCEPT

--[[
                        radius_ = radius_ / std::max(1.0 / 3.0,
                                                       1.0 - pow(2.0 * step_quality - 1.0, 3));
                      radius_ = std::min(max_radius_, radius_);
                      decrease_factor_ = 2.0;
--]]
                    emit quote
                        -- TODO: compute relative_decrease 
                        var relative_decrease = 1.0f
                        var min_factor = 1.0f/3.0f
                        var tmp_factor = 1.0f - util.cpuMath.pow(2.0f * relative_decrease - 1.0f, 3.0f)
                        pd.parameters.trust_region_radius = pd.parameters.trust_region_radius / util.cpuMath.fmax(min_factor, tmp_factor)
                        pd.parameters.radius_decrease_factor = 2.0f

                        logSolver("\n")
                        pd.prevCost = newCost

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
		pd.z:initGPU()
		pd.p:initGPU()
		pd.Ap_X:initGPU()
        pd.DtD:initGPU()
		pd.preconditioner:initGPU()
		pd.g:initGPU()
		
		[util.initPrecomputedImages(`pd.parameters,problemSpec)]	
		C.cudaMalloc([&&opaque](&(pd.scanAlphaNumerator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.scanBetaNumerator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.scanAlphaDenominator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.scanBetaDenominator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.modelCostChange)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.maxDiagJTJ)), sizeof(opt_float))
		
		C.cudaMalloc([&&opaque](&(pd.scratchF)), sizeof(opt_float))
		pd.J_values = nil
		return &pd.plan
	end
	return makePlan
end
