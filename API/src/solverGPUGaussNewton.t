local S = require("std")
local util = require("util")
local dbg = require("dbg")
require("precision")
local C = util.C
local Timer = util.Timer

local getValidUnknown = util.getValidUnknown
local use_dump_j = false
local use_fused_jtj = false


local CERES_style_guardedInvert = true
local CERES_style_preconditioning = true

local cd = macro(function(apicall) return quote
        var r = apicall
        if r ~= 0 then  
            C.printf("Cuda reported error %d: %s\n",r, C.cudaGetErrorString(r))
            C.exit(r)
        end
    in
        r
    end end)
if use_dump_j then
    local cusparsepath = "/usr/local/cuda"
    terralib.linklibrary(cusparsepath.."/lib64/libcusparse.so")
    terralib.includepath = terralib.includepath..";"..
                           cusparsepath.."/include/"
    CUsp = terralib.includecstring [[
        #include <cusparse_v2.h>
    ]]
end


local gpuMath = util.gpuMath

opt.BLOCK_SIZE = 16
local BLOCK_SIZE =  opt.BLOCK_SIZE

local FLOAT_EPSILON = `[opt_float](0.0) 
-- GAUSS NEWTON (non-block version)
return function(problemSpec)
    local UnknownType = problemSpec:UnknownType()
    local TUnknownType = UnknownType:terratype()	
    -- start of the unknowns that correspond to this image
    -- for each entry there are a constant number of unknowns
    -- corresponds to the col dim of the J matrix
    local imagename_to_unknown_offset = {}
    
    -- start of the rows of residuals for this energy spec
    -- corresponds to the row dim of the J matrix
    local energyspec_to_residual_offset_exp = {}
    
    -- start of the block of non-zero entries that correspond to this energy spec
    -- the total dimension here adds up to the number of non-zeros
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
            --print(("image %s has offset %d"):format(image.name,nUnknowns))
            nUnknowns = nUnknowns + image.imagetype.ispace:cardinality()*image.imagetype.channelcount
        end
        for i,es in ipairs(problemSpec.energyspecs) do
            --print("ES",i,nResidualsExp,nnzExp)
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
    
    local struct PlanData {
		plan : opt.Plan
		parameters : problemSpec:ParameterType()
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
		g : TUnknownType		--gradient of F(x): g = -2J'F -> num vars
		
        prevX : TUnknownType -- Place to copy unknowns to before speculatively updating. Avoids hassle when (X + delta) - delta != X 

		scanAlphaNumerator : &opt_float
		scanAlphaDenominator : &opt_float
		scanBetaNumerator : &opt_float

        modelCostChange : &opt_float    -- modelCostChange = L(0) - L(delta) where L(h) = F' F + 2 h' J' F + h' J' J h
		
		timer : Timer
		endSolver : util.TimerEvent
		nIter : int				--current non-linear iter counter
		nIterations : int		--non-linear iterations
		lIterations : int		--linear iterations
	    prevCost : opt_float
	    
	    J_csrValA : &opt_float
	    J_csrColIndA : &int
	    J_csrRowPtrA : &int
	    
	    JT_csrValA : &float
		JT_csrRowPtrA : &int
		JT_csrColIndA : &int
		
		JTJ_csrValA : &float
		JTJ_csrRowPtrA : &int
		JTJ_csrColIndA : &int
		
		JTJ_nnz : int
	    
	    Jp : &float
	}
	if use_dump_j then
	    PlanData.entries:insert {"handle", CUsp.cusparseHandle_t }
	    PlanData.entries:insert {"desc", CUsp.cusparseMatDescr_t }
	end
	S.Object(PlanData)
	local terra swapCol(pd : &PlanData, a : int, b : int)
	    pd.J_csrValA[a],pd.J_csrColIndA[a],pd.J_csrValA[b],pd.J_csrColIndA[b] =
	        pd.J_csrValA[b],pd.J_csrColIndA[b],pd.J_csrValA[a],pd.J_csrColIndA[a]
	end
	local terra sortCol(pd : &PlanData, s : int, e : int)
	    for i = s,e do
            var minidx = i
            var min = pd.J_csrColIndA[i]
            for j = i+1,e do
                if pd.J_csrColIndA[j] < min then
                    min = pd.J_csrColIndA[j]
                    minidx = j
                end
            end
            swapCol(pd,i,minidx)
        end
	end
    local terra wrap(c : int, v : float)
        if c < 0 then
            if v ~= 0.f then
                printf("wrap a non-zero? %d %f\n",c,v)
            end
            c = c + nUnknowns
        end
        if c >= nUnknowns then
            if v ~= 0.f then
                printf("wrap a non-zero? %d %f\n",c,v)
            end
            c = c - nUnknowns
        end
        return c
    end
    local function generateDumpJ(ES,dumpJ,idx,pd)
        local nnz_per_entry = 0
        for i,r in ipairs(ES.residuals) do
            nnz_per_entry = nnz_per_entry + #r.unknowns
        end
        local base_rowidx = energyspec_to_rowidx_offset_exp[ES]
        local base_residual = energyspec_to_residual_offset_exp[ES]
        local idx_offset
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
                        pd.J_csrRowPtrA[local_residual+residual] = local_rowidx + nnz
                    end
                    local begincolumns = nnz
                    for i,u in ipairs(r.unknowns) do
                        local image_offset = imagename_to_unknown_offset[u.image.name]
                        local nchannels = u.image.type.channelcount
                        local uidx = GetOffset(idx,u.index)
                        local unknown_index = `image_offset + nchannels*uidx + u.channel
                        emit quote
                            pd.J_csrValA[local_rowidx + nnz] = rhs.["_"..tostring(nnz)]
                            pd.J_csrColIndA[local_rowidx + nnz] = wrap(unknown_index,rhs.["_"..tostring(nnz)])
                        end
                        nnz = nnz + 1
                    end
                    -- sort the columns
                    emit quote
                        sortCol(&pd, local_rowidx + begincolumns, local_rowidx + nnz)
                    end
                    residual = residual + 1
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

        terra kernels.computeQ(pd : PlanData)
            var q = 0.0f
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then 
                -- Right side is -2 of CERES versions, left is just negative version, 
                --  so after the dot product, just need to multiply by 2 to recover value identical to CERES  
                q = 0.5*(pd.delta(idx):dot(pd.r(idx) + pd.b(idx))) 
            end
            unknownWideReduction(idx,q,pd.scratch)
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
                unknownWideReduction(idx,d,pd.scanAlphaNumerator)
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

            unknownWideReduction(idx,d,pd.scanAlphaNumerator)
        end

        terra kernels.PCGStep1(pd : PlanData)
            var d = 0.0f
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                var tmp : unknownElement = 0.0f
                 -- A x p_k  => J^T x J x p_k 
                tmp = fmap.applyJTJ(idx, pd.parameters, pd.p, pd.CtC)
                pd.Ap_X(idx) = tmp					 -- store for next kernel call
                d = pd.p(idx):dot(tmp)			 -- x-th term of denominator of alpha
            end
            
            unknownWideReduction(idx,d,pd.scanAlphaDenominator)
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
            unknownWideReduction(idx,b,pd.scanBetaNumerator)
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
            unknownWideReduction(idx,b,pd.scanBetaNumerator) 
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
                pd.Adelta(idx) = fmap.applyJTJ(idx, pd.parameters, pd.delta, pd.CtC)
            end
        end

        terra kernels.recomputeResiduals(pd : PlanData)
            var d = 0.0f
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                var Ax = pd.Adelta(idx)
                var b = pd.b(idx)
                var newR = b - Ax
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
                util.atomicAdd(pd.scratch, cost)
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
            terra kernels.PCGComputeCtC(pd : PlanData)
                var idx : Index
                if idx:initFromCUDAParams() then                         
                    if not fmap.exclude(idx,pd.parameters) then 
                        var CtC = fmap.computeCtC(idx, pd.parameters, pd.preconditioner)
                        pd.CtC(idx) = CtC
                    end        
                end 
            end


            terra kernels.PCGFinalizeDiagonal(pd : PlanData)
                var idx : Index
                var d = [opt_float](0.0)
                if idx:initFromCUDAParams() then
                    if not fmap.exclude(idx,pd.parameters) then 
                        var unclampedCtC = pd.CtC(idx)
                        var invS_iiSq = 1.0 / pd.preconditioner(idx)
                        var minVal = square(pd.parameters.min_lm_diagonal) * invS_iiSq
                        var maxVal = square(pd.parameters.max_lm_diagonal) * invS_iiSq
                        var CtC = clamp(unclampedCtC, minVal, maxVal)
                        pd.CtC(idx) = CtC
                        if [CERES_style_preconditioning] then
                            var pre = [opt_float](1.0) / (CtC+pd.parameters.trust_region_radius*unclampedCtC) 
                            pd.preconditioner(idx) = pre
                            var residuum = pd.r(idx)
                            var p = pre*residuum    -- apply pre-conditioner M^-1
                            pd.p(idx) = p
                            d = residuum:dot(p)
                        end
                    end        
                end    
                unknownWideReduction(idx,d,pd.scanAlphaNumerator)
            end

            terra kernels.DebugDumpCtC(pd : PlanData)

                var idx : Index
                if idx:initFromCUDAParams() then                         
                    var CtC : unknownElement = pd.CtC(idx)
                    printf("\nC'C: %d: %f %f %f\n", idx, CtC(0), CtC(1), CtC(2))
                    var JtJ_ii : unknownElement = pd.CtC(idx) * pd.parameters.trust_region_radius
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
                    util.atomicAdd(pd.scratch, cost)
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
            var d = [opt_float](0.0f)
            var tIdx = 0 
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
               d = d + fmap.applyJTJ(tIdx, pd.parameters, pd.p, pd.Ap_X, pd.CtC)
            end 
            d = util.warpReduce(d)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.scanAlphaDenominator, d)
            end
        end

        terra kernels.computeAdelta_Graph(pd : PlanData)
            var tIdx = 0 
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
                fmap.applyJTJ(tIdx, pd.parameters, pd.delta, pd.Adelta, pd.CtC)
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
                util.atomicAdd(pd.scratch, cost)
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
        end
        if problemSpec:UsesLambda() then
            terra kernels.PCGComputeCtC_Graph(pd : PlanData)
                var tIdx = 0
                if util.getValidGraphElement(pd,[graphname],&tIdx) then
                    fmap.computeCtC(tIdx, pd.parameters, pd.CtC, pd.preconditioner)
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
                    util.atomicAdd(pd.scratch, cost)
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
                                                                        "PCGStep2",
                                                                        "PCGStep2_1stHalf",
                                                                        "PCGStep2_2ndHalf",
                                                                        "PCGStep3",
                                                                        "PCGLinearUpdate",
                                                                        "revertUpdate",
                                                                        "savePreviousUnknowns",
                                                                        "computeCost",
                                                                        "computeQ",
                                                                        "precompute",
                                                                        "copyResidualsToB",
                                                                        "computeAdelta",
                                                                        "computeAdelta_Graph",
                                                                        "recomputeResiduals",
                                                                        "PCGInit1_Graph",
                                                                        "PCGComputeCtC_Graph",
                                                                        "PCGStep1_Graph",
                                                                        "computeCost_Graph",
                                                                        "computeModelCost",
                                                                        "computeModelCost_Graph",
                                                                        "saveJToCRS",
                                                                        "DebugDumpCtC",
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
        C.cudaMemset(pd.scratch, 0, sizeof(opt_float))
        gpu.computeModelCost(pd)
        gpu.computeModelCost_Graph(pd)
        var f : opt_float
        C.cudaMemcpy(&f, pd.scratch, sizeof(opt_float), C.cudaMemcpyDeviceToHost)
        return f
    end

    local sqrtf = util.cpuMath.sqrt

    local terra computeQ(pd : &PlanData) : opt_float
        C.cudaMemset(pd.scratch, 0, sizeof(opt_float))
        gpu.computeQ(pd)

        var f : opt_float
        C.cudaMemcpy(&f, pd.scratch, sizeof(opt_float), C.cudaMemcpyDeviceToHost)

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
            logSolver(" model_cost_change=%f \n",model_cost_change)
            return model_cost_change
        end

        terra initLambda(pd : &PlanData)
            pd.parameters.trust_region_radius = 1e4
            --pd.parameters.trust_region_radius = 1e6
            --pd.parameters.trust_region_radius = 
            --pd.parameters.trust_region_radius = 0.005
            --pd.parameters.trust_region_radius = 585.37623720501074000000
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

    local terra GetToHost(ptr : &opaque, N : int) : &int
        var r = [&int](C.malloc(sizeof(int)*N))
        C.cudaMemcpy(r,ptr,N*sizeof(int),C.cudaMemcpyDeviceToHost)
        return r
    end
    local cusparseInner,cusparseOuter
    if use_dump_j then
        terra cusparseOuter(pd : &PlanData)
            var [parametersSym] = &pd.parameters
            --logSolver("saving J...\n")
            gpu.saveJToCRS(pd)
            if isGraph then
                gpu.saveJToCRS_Graph(pd)
            end
            --logSolver("... done\n")
            if pd.JTJ_csrRowPtrA == nil then
                --allocate row
                --C.printf("alloc JTJ\n")
                var numrows = nUnknowns + 1
                cd(C.cudaMalloc([&&opaque](&pd.JTJ_csrRowPtrA),sizeof(int)*numrows))
                var endJTJalloc : util.TimerEvent
                pd.timer:startEvent("J^TJ alloc",nil,&endJTJalloc)
                cd(CUsp.cusparseXcsrgemmNnz(pd.handle, CUsp.CUSPARSE_OPERATION_TRANSPOSE, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      nUnknowns,nUnknowns,nResidualsExp,
                                      pd.desc,nnzExp,pd.J_csrRowPtrA,pd.J_csrColIndA,
                                      pd.desc,nnzExp,pd.J_csrRowPtrA,pd.J_csrColIndA,
                                      pd.desc,pd.JTJ_csrRowPtrA, &pd.JTJ_nnz))
                pd.timer:endEvent(nil,endJTJalloc)
                
                cd(C.cudaMalloc([&&opaque](&pd.JTJ_csrColIndA), sizeof(int)*pd.JTJ_nnz))
                cd(C.cudaMalloc([&&opaque](&pd.JTJ_csrValA), sizeof(float)*pd.JTJ_nnz))
                cd(C.cudaThreadSynchronize())
            end
            
            var endJTJmm : util.TimerEvent
            pd.timer:startEvent("JTJ multiply",nil,&endJTJmm)
    
            cd(CUsp.cusparseScsrgemm(pd.handle, CUsp.CUSPARSE_OPERATION_TRANSPOSE, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
           nUnknowns,nUnknowns,nResidualsExp,
           pd.desc,nnzExp,pd.J_csrValA,pd.J_csrRowPtrA,pd.J_csrColIndA,
           pd.desc,nnzExp,pd.J_csrValA,pd.J_csrRowPtrA,pd.J_csrColIndA,
           pd.desc, pd.JTJ_csrValA, pd.JTJ_csrRowPtrA,pd.JTJ_csrColIndA ))
           pd.timer:endEvent(nil,endJTJmm)
           
            var endJtranspose : util.TimerEvent
            pd.timer:startEvent("J_transpose",nil,&endJtranspose)
            cd(CUsp.cusparseScsr2csc(pd.handle,nResidualsExp, nUnknowns,nnzExp,
                                 pd.J_csrValA,pd.J_csrRowPtrA,pd.J_csrColIndA,
                                 pd.JT_csrValA,pd.JT_csrColIndA,pd.JT_csrRowPtrA,
                                 CUsp.CUSPARSE_ACTION_NUMERIC,CUsp.CUSPARSE_INDEX_BASE_ZERO))
            pd.timer:endEvent(nil,endJtranspose)
        end
        terra cusparseInner(pd : &PlanData)
            var [parametersSym] = &pd.parameters
            
            if false then
                C.printf("begin debug dump\n")
                var J_csrColIndA = GetToHost(pd.J_csrColIndA,nnzExp)
                var J_csrRowPtrA = GetToHost(pd.J_csrRowPtrA,nResidualsExp + 1)
                for i = 0,nResidualsExp do
                    var b,e = J_csrRowPtrA[i],J_csrRowPtrA[i+1]
                    if b >= e or b < 0 or b >= nnzExp or e < 0 or e > nnzExp then
                        C.printf("ERROR: %d %d %d (total = %d)\n",i,b,e,nResidualsExp)
                    end
                    --C.printf("residual %d -> {%d,%d}\n",i,b,e)
                    for j = b,e do
                        if J_csrColIndA[j] < 0 or J_csrColIndA[j] >= nnzExp then
                            C.printf("ERROR: j %d (total = %d)\n",j,J_csrColIndA[j])
                        end
                        if j ~= b and J_csrColIndA[j-1] >= J_csrColIndA[j] then
                            C.printf("ERROR: sort j[%d] = %d, j[%d] = %d\n",j-1,J_csrColIndA[j-1],j,J_csrColIndA[j])
                        end
                        --C.printf("colindex: %d\n",J_csrColIndA[j])
                    end
                end
                C.printf("end debug dump\n")
            end
            
            var consts = array(0.f,1.f,2.f)
            cd(C.cudaMemset(pd.Ap_X._contiguousallocation, -1, sizeof(float)*nUnknowns))
            
            if use_fused_jtj then
                var endJTJp : util.TimerEvent
                pd.timer:startEvent("J^TJp",nil,&endJTJp)
                cd(CUsp.cusparseScsrmv(
                            pd.handle, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
                            nUnknowns, nUnknowns,pd.JTJ_nnz,
                            &consts[2], pd.desc,
                            pd.JTJ_csrValA, 
                            pd.JTJ_csrRowPtrA, pd.JTJ_csrColIndA,
                            [&float](pd.p._contiguousallocation),
                            &consts[0], [&float](pd.Ap_X._contiguousallocation)
                        ))
                pd.timer:endEvent(nil,endJTJp)
            else
                var endJp : util.TimerEvent
                pd.timer:startEvent("Jp",nil,&endJp)
                cd(CUsp.cusparseScsrmv(
                            pd.handle, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
                            nResidualsExp, nUnknowns,nnzExp,
                            &consts[1], pd.desc,
                            pd.J_csrValA, 
                            pd.J_csrRowPtrA, pd.J_csrColIndA,
                            [&float](pd.p._contiguousallocation),
                            &consts[0], pd.Jp
                        ))
                pd.timer:endEvent(nil,endJp)
                var endJT : util.TimerEvent
                pd.timer:startEvent("J^T",nil,&endJT)
                cd(CUsp.cusparseScsrmv(
                            pd.handle, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
                            nUnknowns, nResidualsExp, nnzExp,
                            &consts[2], pd.desc,
                            pd.JT_csrValA, 
                            pd.JT_csrRowPtrA, pd.JT_csrColIndA,
                            pd.Jp,
                            &consts[0],[&float](pd.Ap_X._contiguousallocation) 
                        ))
                pd.timer:endEvent(nil,endJT)
            end
        end
    else
        terra cusparseInner(pd : &PlanData) end
        terra cusparseOuter(pd : &PlanData) end
    end

	local terra init(data_ : &opaque, params_ : &&opaque, solverparams : &&opaque)
	   var pd = [&PlanData](data_)
	   pd.timer:init()
	   pd.timer:startEvent("overall",nil,&pd.endSolver)
       [util.initParameters(`pd.parameters,problemSpec,params_,true)]
       var [parametersSym] = &pd.parameters
        escape if use_dump_j then emit quote
            if pd.J_csrValA == nil then
                cd(CUsp.cusparseCreateMatDescr( &pd.desc ))
                cd(CUsp.cusparseSetMatType( pd.desc,CUsp.CUSPARSE_MATRIX_TYPE_GENERAL ))
                cd(CUsp.cusparseSetMatIndexBase( pd.desc,CUsp.CUSPARSE_INDEX_BASE_ZERO ))
                cd(CUsp.cusparseCreate( &pd.handle ))
                
                logSolver("nnz = %s\n",[tostring(nnzExp)])
                logSolver("nResiduals = %s\n",[tostring(nResidualsExp)])
                logSolver("nnz = %d, nResiduals = %d\n",int(nnzExp),int(nResidualsExp))
                
                -- J alloc
                C.cudaMalloc([&&opaque](&(pd.J_csrValA)), sizeof(opt_float)*nnzExp)
                C.cudaMalloc([&&opaque](&(pd.J_csrColIndA)), sizeof(int)*nnzExp)
                C.cudaMemset(pd.J_csrColIndA,-1,sizeof(int)*nnzExp)
                C.cudaMalloc([&&opaque](&(pd.J_csrRowPtrA)), sizeof(int)*(nResidualsExp+1))
                
                -- J^T alloc
                C.cudaMalloc([&&opaque](&pd.JT_csrValA), nnzExp*sizeof(float))
                C.cudaMalloc([&&opaque](&pd.JT_csrColIndA), nnzExp*sizeof(int))
                C.cudaMalloc([&&opaque](&pd.JT_csrRowPtrA), (nUnknowns + 1) *sizeof(int))
                
                -- Jp alloc
                cd(C.cudaMalloc([&&opaque](&pd.Jp), nResidualsExp*sizeof(float)))
                
                -- write J_csrRowPtrA end
                var nnz = nnzExp
                C.printf("setting rowptr[%d] = %d\n",nResidualsExp,nnz)
                C.cudaMemcpy(&pd.J_csrRowPtrA[nResidualsExp],&nnz,sizeof(int),C.cudaMemcpyHostToDevice)
            end
        end end end
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
        var q_tolerance = 1e-4
        var Q0 : opt_float
        var Q1 : opt_float
		var pd = [&PlanData](data_)
		[util.initParameters(`pd.parameters,problemSpec, params_,false)]
		if pd.nIter < pd.nIterations then
			C.cudaMemset(pd.scanAlphaNumerator, 0, sizeof(opt_float))	--scan in PCGInit1 requires reset
			C.cudaMemset(pd.scanAlphaDenominator, 0, sizeof(opt_float))	--scan in PCGInit1 requires reset
			C.cudaMemset(pd.scanBetaNumerator, 0, sizeof(opt_float))	--scan in PCGInit1 requires reset

			gpu.PCGInit1(pd)
			if isGraph then
				gpu.PCGInit1_Graph(pd)	
				gpu.PCGInit1_Finish(pd)	
			end

            var pre : double[3]
            var s : double[3]
            C.cudaMemcpy(&pre, pd.preconditioner.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
            
            s[0] = sqrtf(pre[0])
            s[1] = sqrtf(pre[1])
            s[2] = sqrtf(pre[2])
            logSolver("\t S %g %g %g\n", s[0], s[1], s[2])
            var v : double[3]


            escape 
                if problemSpec:UsesLambda() then
                    emit quote
                        C.cudaMemset(pd.scanAlphaNumerator, 0, sizeof(opt_float))
                        --logSolver(" trust_region_radius=%f ",pd.parameters.trust_region_radius)
                        gpu.PCGComputeCtC(pd)
                        gpu.PCGComputeCtC_Graph(pd)
                        gpu.PCGFinalizeDiagonal(pd)
                        --gpu.DebugDumpCtC(pd)
                        gpu.copyResidualsToB(pd)
                        --gpu.recomputeResiduals(pd)
                        Q0 = computeQ(pd)
                        logSolver("\nQ0=%g\n", Q0)
                        var v : double[3]
                var pre : double[3]

                C.cudaMemcpy(&v, pd.preconditioner.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t preconditioner %g %g %g\n", v[0], v[1], v[2])
                logSolver("\t CERES preconditioner %g %g %g\n", v[0]/(s[0]*s[0]), v[1]/(s[1]*s[1]), v[2]/(s[2]*s[2]))


                C.cudaMemcpy(&v, pd.r.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t r %g %g %g\n", v[0], v[1], v[2])
                logSolver("\t CERES r %g %g %g\n", -1*s[0]*v[0], -1*s[1]*v[1], -1*s[2]*v[2])

                var rad = pd.parameters.trust_region_radius
                C.cudaMemcpy(&v, pd.CtC.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t CtC %g %g %g\n", v[0], v[1], v[2])

                var diagJtJ : double[3]
                var diagA : double[3]
                var altPre : double[3]
                for i = 0,3 do
                    diagJtJ[i] = v[i] * rad
                    diagA[i] = (diagJtJ[i] + v[i])*s[i]*s[i]
                    altPre[i] = s[i]*s[i] / diagA[i]
                end
                
                logSolver("\t diag(JtJ) %g %g %g\n", diagJtJ[0], diagJtJ[1], diagJtJ[2])
                logSolver("\t diag(JStJS) %g %g %g\n", s[0]*s[0]*diagJtJ[0], s[1]*s[1]*diagJtJ[1], s[2]*s[2]*diagJtJ[2])

                logSolver("\t DtD %g %g %g\n", s[0]*s[0]*v[0], s[1]*s[1]*v[1], s[2]*s[2]*v[2])
                logSolver("\t K %g %g %g\n", diagA[0], diagA[1], diagA[2])
                logSolver("\t altPre %g %g %g\n", altPre[0], altPre[1], altPre[2])


                C.cudaMemcpy(&v, pd.b.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t b %g %g %g\n", v[0], v[1], v[2])
                logSolver("\t CERES b %g %g %g\n", -1*s[0]*v[0], -1*s[1]*v[1], -1*s[2]*v[2])

                C.cudaMemcpy(&v, pd.r.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t r %g %g %g\n", v[0], v[1], v[2])
                logSolver("\t CERES r %g %g %g\n", -1*s[0]*v[0], -1*s[1]*v[1], -1*s[2]*v[2])

                C.cudaMemcpy(&v, pd.delta.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t delta %g %g %g\n", v[0], v[1], v[2])
                logSolver("\t CERES delta %g %g %g\n", -v[0]/s[0], -v[1]/s[1], -v[2]/s[2])

                C.cudaMemcpy(&v, pd.p.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t p %g %g %g\n", v[0], v[1], v[2])
                logSolver("\t CERES p %g %g %g\n", -v[0]/s[0], -v[1]/s[1], -v[2]/s[2])
                    end
                end
            end
            cusparseOuter(pd)
            for lIter = 0, pd.lIterations do				
                var v : double[3]

                C.cudaMemcpy(&v, pd.r.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\nlinIter: %d\n\t r %g %g %g\n", lIter, v[0], v[1], v[2])

                C.cudaMemcpy(&v, pd.z.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t z %g %g %g\n", v[0], v[1], v[2])

                C.cudaMemcpy(&v, pd.p.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t p %g %g %g\n", v[0], v[1], v[2])

                C.cudaMemcpy(&v, pd.Ap_X.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\t Ap_X %g %g %g\n", v[0], v[1], v[2])


                C.cudaMemcpy(&v, pd.delta.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\tdelta %g %g %g\n", v[0], v[1], v[2])

                C.cudaMemcpy(&v, pd.scanAlphaNumerator, sizeof(double), C.cudaMemcpyDeviceToHost)
                logSolver("\tscanAlphaNumerator %g\n", v[0])
                C.cudaMemcpy(&v, pd.scanAlphaDenominator, sizeof(double), C.cudaMemcpyDeviceToHost)
                logSolver("\tscanAlphaDenominator %g\n", v[0])

                var numerator : double
                C.cudaMemcpy(&numerator, pd.scanAlphaNumerator, sizeof(double), C.cudaMemcpyDeviceToHost)
                logSolver("\talpha %g\n", numerator/v[0])


                C.cudaMemcpy(&v, pd.scanBetaNumerator, sizeof(double), C.cudaMemcpyDeviceToHost)
                logSolver("\tscanBetaNumerator %g\n", v[0])



                C.cudaMemset(pd.scanAlphaDenominator, 0, sizeof(opt_float))

				gpu.PCGStep1(pd)

				if isGraph then
					gpu.PCGStep1_Graph(pd)	
				end
				

				-- only does anything if use_dump_j is true
				cusparseInner(pd)
				
				C.cudaMemset(pd.scanBetaNumerator, 0, sizeof(opt_float))
				
				if [problemSpec:UsesLambda()] and ((lIter + 1) % residual_reset_period) == 0 then
                    gpu.PCGStep2_1stHalf(pd)
                    gpu.computeAdelta(pd)
                    -- TODO: merge these?
                    gpu.computeAdelta_Graph(pd)
                    gpu.recomputeResiduals(pd)
                    gpu.PCGStep2_2ndHalf(pd)
                else
                    gpu.PCGStep2(pd)
                end
                gpu.PCGStep3(pd)


				-- save new rDotz for next iteration
				C.cudaMemcpy(pd.scanAlphaNumerator, pd.scanBetaNumerator, sizeof(opt_float), C.cudaMemcpyDeviceToDevice)	
				
				if [problemSpec:UsesLambda()] then
	                Q1 = computeQ(pd)
	                var zeta = [opt_float](lIter+1)*(Q1 - Q0) / Q1 
                    logSolver("Q1=%g\n", Q1 )
                    logSolver("zeta=%g\n", zeta)
	                if zeta < q_tolerance then
	                    break
	                end
	                Q0 = Q1
				end
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
			
            C.cudaMemcpy(&v, pd.prevX.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
            logSolver("\tX %g %g %g\n", v[0], v[1], v[2])

            --[[ TODO: Remove
            if newCost > 1000000 then
                logSolver("Shitty cost!")  
                var v : double[3]


                C.cudaMemcpy(&v, pd.prevX.funcParams.data, sizeof(double)*3, C.cudaMemcpyDeviceToHost)
                logSolver("\tX %.20f %.20f %.20f\n", v[0], v[1], v[2])
                logSolver(" trust_region_radius=%.20f ",pd.parameters.trust_region_radius)

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
                
                return 0
            end
            --]]
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
                            pd.parameters.radius_decrease_factor = 2.0 * pd.parameters.radius_decrease_factor
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

    local terra cost(data_ : &opaque) : double
        var pd = [&PlanData](data_)
        return [double](pd.prevCost)
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
        pd.CtC:initGPU()
		pd.preconditioner:initGPU()
		pd.g:initGPU()
        pd.prevX:initGPU()
		
		[util.initPrecomputedImages(`pd.parameters,problemSpec)]	
		C.cudaMalloc([&&opaque](&(pd.scanAlphaNumerator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.scanBetaNumerator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.scanAlphaDenominator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.modelCostChange)), sizeof(opt_float))
		
		C.cudaMalloc([&&opaque](&(pd.scratch)), sizeof(opt_float))
		pd.J_csrValA = nil
		pd.JTJ_csrRowPtrA = nil
		return &pd.plan
	end
	return makePlan
end
