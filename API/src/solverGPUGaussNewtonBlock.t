
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

__syncthreads = cudalib.nvvm_barrier0

local vload = macro(function(x) return `terralib.attrload(x, {isvolatile = true}) end)
local vstore = macro(function(x,v) return `terralib.attrstore(x,v, {isvolatile = true}) end)

local FLOAT_EPSILON = `0.000001f
local MINF = -math.huge

--TODO this stuff needs to come from the cost function (patch size and stencil overlap)
opt.BLOCK_SIZE = 16
local BLOCK_SIZE 				=  opt.BLOCK_SIZE
local SHARED_MEM_SIZE_BLOCK	   	= ((BLOCK_SIZE+2)*(BLOCK_SIZE+2))
local SHARED_MEM_SIZE_VARIABLES = ((BLOCK_SIZE)*(BLOCK_SIZE))
local SHARED_MEM_SIZE_RESIDUUMS = ((SHARED_MEM_SIZE_VARIABLES)+4*(SHARED_MEM_SIZE_VARIABLES))

local function constanttable(tbl)
	return terralib.constant(terralib.new(int[#tbl],tbl))
end

local offsetX = constanttable{math.floor(0.0*BLOCK_SIZE), math.floor((1.0/2.0)*BLOCK_SIZE), math.floor((1.0/4.0)*BLOCK_SIZE), math.floor((3.0/4.0)*BLOCK_SIZE), math.floor((1.0/8.0)*BLOCK_SIZE), math.floor((5.0/8.0)*BLOCK_SIZE), math.floor((3.0/8.0)*BLOCK_SIZE), math.floor((7.0/8.0)*BLOCK_SIZE)} -- Halton sequence base 2

local offsetY = constanttable{math.floor(0.0*BLOCK_SIZE), math.floor((1.0/3.0)*BLOCK_SIZE), math.floor((2.0/3.0)*BLOCK_SIZE), math.floor((1.0/9.0)*BLOCK_SIZE), math.floor((4.0/9.0)*BLOCK_SIZE), math.floor((7.0/9.0)*BLOCK_SIZE), math.floor((2.0/9.0)*BLOCK_SIZE), math.floor((5.0/9.0)*BLOCK_SIZE)}	-- Halton sequence base 3

	
local terra min(a : float, b : float) : float
	if a < b then return a
	else return b end
end

local terra max(a : float, b : float) : float
	if a > b then return a	else return b end
end

local terra isInsideImage(i : int, j : int, width : uint, height : uint) : bool
	return i >= 0 and i < height and j >= 0 and j < width
end

local terra getLinearThreadId(tId_i : int, tId_j : int) : uint
	return tId_i*BLOCK_SIZE+tId_j;
end


local vload = macro(function(x) return `terralib.attrload(x, {isvolatile = true}) end)
local vstore = macro(function(x,v) return `terralib.attrstore(x,v, {isvolatile = true}) end)

local terra warpReduce(sdata : &float, threadIdx : int, threadsPerBlock : uint) --See Optimizing Parallel Reduction in CUDA by Mark Harris
	if threadIdx < 32 then
	    escape
	        for i = 5,0,-1 do
	            emit quote
	                if threadIdx + [2^i] < threadsPerBlock then vstore(sdata+threadIdx, vload(sdata+threadIdx) + vload(sdata + threadIdx + [2^i])) end
	            end
	        end
	    end
	end
end

local terra blockReduce(sdata : &float, threadIdx : int, threadsPerBlock : uint)
	--TODO: either we meta-program this ourself, or get #pragma unroll to work in Terra
	var stride = threadsPerBlock / 2
	while stride > 32 do
		if threadIdx < stride then
		    var t = vload(sdata + threadIdx) + vload(sdata + threadIdx + stride)
		    vstore(sdata + threadIdx,t)
		end
		__syncthreads()   
	    stride = stride / 2
	end
	warpReduce(sdata, threadIdx, threadsPerBlock)
end


return function(problemSpec, vars)

	local struct PlanData(S.Object) {
		plan : opt.Plan
		parameters : problemSpec:ParameterType(false)	--get non-blocked version
		scratchF : &float
		
		delta : problemSpec:UnknownType()	--current linear update to be computed -> num vars

		timer : Timer
	}
	
	--local patchBucket = cudalib.sharedmemory(float, math:ceil(SHARED_MEM_SIZE_VARIABLES / util.warpSize))
	local patchBucket = cudalib.sharedmemory(float,SHARED_MEM_SIZE_VARIABLES)
	
	local CopyToShared = terralib.memoize(function(Image,ImageBlock)
			
		local stencil = problemSpec:MaxStencil()
		local offset = stencil*problemSpec:BlockStride() + stencil
		local blockStride = problemSpec:BlockStride()
		
		return terra(blockCornerX : int64, blockCornerY : int64, image : Image, imageBlock : ImageBlock)
			
			var numBlockThreads : int = blockDim.x * blockDim.y			
			var numVariables : int = blockStride*blockStride
			
			var baseIdx : int = threadIdx.x + threadIdx.y*blockDim.x
			for i = 0,numVariables,numBlockThreads do
				var linearIdx : int = baseIdx + i
				
				if linearIdx < numVariables then				
					var localX = linearIdx % blockStride - stencil
					var localY = linearIdx / blockStride - stencil
					
					var globalX = localX + blockCornerX
					var globalY = localY + blockCornerY
					
					imageBlock(localX, localY) = image:get(globalX, globalY)	--bounded check
				end
			end			
		end
	end)
	
	local kernels = {}
	kernels.PCGStepBlock = function(data)
		local terra PCGStepBlockGPU(pd : &data.PlanData, ox : int, oy : int, nBlockIterations : uint)
			
			var W = pd.parameters.X:W()
			var H = pd.parameters.X:H()
	
			var tId_i : int = threadIdx.x -- local col idx
			var tId_j : int = threadIdx.y -- local row idx
	
	
			var gId_i : int = blockIdx.x * blockDim.x + threadIdx.x - ox -- global col idx
			var gId_j : int = blockIdx.y * blockDim.y + threadIdx.y - oy -- global row idx
			
			var blockCornerX : int = blockIdx.x * blockDim.x - ox
			var blockCornerY : int = blockIdx.y * blockDim.y - oy
			var blockParams : problemSpec:ParameterType(true)
			var P : problemSpec:UnknownType(true)
			-- load everything into shared memory
			escape				
				for i,p in ipairs(problemSpec.parameters) do
					if p.kind ~= "image" then
						emit quote
							blockParams.[p.name] = pd.parameters.[p.name]
						end
					else 
						local blockedType = problemSpec:BlockedTypeForImage(p)
						local stencil = problemSpec:MaxStencil()
						local offset = stencil*problemSpec:BlockStride() + stencil
						local shmem = cudalib.sharedmemory(p.type.metamethods.typ, SHARED_MEM_SIZE_VARIABLES)
						
						emit quote 
							blockParams.[p.name] = [blockedType] { data = [&uint8](shmem + offset) } 
							[CopyToShared(p.type,blockedType)](blockCornerX, blockCornerY, pd.parameters.[p.name], blockParams.[p.name])
						end
					end
				end
			end
			
			-- allocate Pre-conditioner (P)
			escape
				local blockedType = problemSpec:UnknownType(true)
				local shmem = cudalib.sharedmemory(problemSpec:UnknownType().metamethods.typ, SHARED_MEM_SIZE_VARIABLES)		--pre-conditioner
				emit quote
					P = [blockedType] { data = [&uint8](shmem) } 
				end
			end
			
			__syncthreads()

			
			--loadPatchToCache(X, pd.parameters.X, tId_i, tId_j, gId_i, gId_j, W, H)
			--TODO fix the shared memory here (replace pd.X with input.d_targetDepth)
			--loadPatchToCache(TargetDepth, pd.parameters.X, tId_i, tId_j, gId_i, gId_j, W, H)
	
			var Delta : problemSpec:UnknownType().metamethods.typ = 0.0f
			var R : problemSpec:UnknownType().metamethods.typ
			var Z : problemSpec:UnknownType().metamethods.typ
			var Pre : problemSpec:UnknownType().metamethods.typ
			var RDotZOld : problemSpec:UnknownType().metamethods.typ
			var AP : problemSpec:UnknownType().metamethods.typ

			__syncthreads()
	
			--//////////////////////////////////////////////////////////////////////////////////////////
			--// Initialize linear patch systems
			--//////////////////////////////////////////////////////////////////////////////////////////

			
			var d : problemSpec:UnknownType().metamethods.typ = 0.0f

			if isInsideImage(gId_i, gId_j, W, H) then
				--R = evalMinusJTFDevice(tId_i, tId_j, gId_i, gId_j, W, H, TargetDepth, X, parameters, &Pre) -- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0
				--TODO MAKE SURE THAT THIS IS CALLED THE RIGHT WAY (make sure to call blockParams
				--R = -data.problemSpec.functions.gradient.boundary(tId_i, tId_j, gId_i, gId_j, blockParams)
				R = 1.0f
				Pre = 1	--TODO fix this hack... the pre-conditioner needs to be the diagonal of JTJ
				var preRes : float = Pre*R																  -- apply preconditioner M^-1 
				if tId_i == 0 and tId_j == 0 and gId_i == 0 and gId_j == 0 then
					printf("%f ", preRes)
				end
				P(tId_i, tId_j) = preRes											   	  -- save for later
				d = R*preRes
			end
			
			patchBucket[getLinearThreadId(tId_i, tId_j)] = d;											   -- x-th term of nomimator for computing alpha and denominator for computing beta

			__syncthreads()
			blockReduce(patchBucket, getLinearThreadId(tId_i, tId_j), SHARED_MEM_SIZE_VARIABLES)
			__syncthreads()	-- TODO I'm not quire sure if we need this one

			if isInsideImage(gId_i, gId_j, W, H) then 
				RDotZOld = patchBucket[0]							   -- read result for later on
			end
	
			__syncthreads()
		
			
			
			--///////////////////////
			--Do patch PCG iterations
			--///////////////////////
			for patchIter = 0,nBlockIterations,1 do 
			
				var currentP : problemSpec:UnknownType().metamethods.typ = P(tId_i, tId_j)				
				var d : float = 0.0f
					
				if isInsideImage(gId_i, gId_j, W, H) then
					--AP = applyJTJDevice(tId_i, tId_j, gId_i, gId_j, W, H, TargetDepth, P, X, parameters);	-- A x p_k  => J^T x J x p_k 
					--d = currentP*AP;																		-- x-th term of denominator of alpha
					--AP = data.problemSpec.functions.applyJTJ.boundary(tId_i, tId_j, gId_i, gId_j, blockParams, P) -- A x p_k  => J^T x J x p_k 
					AP = 1.0f
					d = currentP*AP
				end
			
				patchBucket[getLinearThreadId(tId_i, tId_j)] = d
		
				__syncthreads()
				blockReduce(patchBucket, getLinearThreadId(tId_i, tId_j), SHARED_MEM_SIZE_VARIABLES)	
				__syncthreads()
	
				var dotProduct : float = patchBucket[0]
		
				var b : float = 0.0f
		
				if isInsideImage(gId_i, gId_j, W, H) then
					var alpha : float = 0.0f
					
					if dotProduct > FLOAT_EPSILON then	-- update step size alpha
						alpha = RDotZOld/dotProduct;
					end
					Delta = Delta+alpha*currentP;	-- do a decent step		
					R = R-alpha*AP;					-- update residuum						
					Z = Pre*R;						-- apply pre-conditioner M^-1
					b = Z*R;						-- compute x-th term of the nominator of beta
				end
			
				__syncthreads() -- Only write if every thread in the block has has read bucket[0]
		
				patchBucket[getLinearThreadId(tId_i, tId_j)] = b;
		
				__syncthreads()
				blockReduce(patchBucket, getLinearThreadId(tId_i, tId_j), SHARED_MEM_SIZE_VARIABLES)	-- sum over x-th terms to compute nominator of beta inside this block
				__syncthreads()
		
				if isInsideImage(gId_i, gId_j, W, H) then
					var rDotzNew : float = patchBucket[0];	-- get new nominator
					
					var beta : float = 0.0f;														 
					if RDotZOld > FLOAT_EPSILON then
						beta = rDotzNew/RDotZOld -- update step size beta
					end
					RDotZOld = rDotzNew -- save new rDotz for next iteration
					P(tId_i, tId_j) = Z+beta*currentP -- update decent direction
				end
				
				__syncthreads()
			end
		
			--	//////////////////////////////////////////////////////////////////////////////////////////
			--	// Save to global memory
			--	//////////////////////////////////////////////////////////////////////////////////////////
			
			
			if isInsideImage(gId_i, gId_j, W, H) then 
				--pd.delta[get1DIdx(gId_i, gId_j, W, H)] = Delta
				pd.delta(gId_i, gId_j) = Delta
			end
			
		end
		return { kernel = PCGStepBlockGPU, header = noHeader, footer = noFooter, params = {symbol(int), symbol(int), symbol(uint)}, mapMemberName = "X" }
	end
	
	kernels.PCGLinearUpdateBlock = function(data)
		local terra PCGLinearUpdateBlockGPU(pd : &data.PlanData)
			var w : int, h : int
			if positionForValidLane(pd, "X", &w, &h) then
				pd.parameters.X(w,h) = pd.parameters.X(w,h) + pd.delta(w,h)
			end
		end
		return { kernel = PCGLinearUpdateBlockGPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "X" }
	end
	
	local gpu = util.makeGPUFunctions(problemSpec, vars, PlanData, kernels)

	

	local terra impl(data_ : &opaque, images : &&opaque, edgeValues : &&opaque, params_ : &&opaque)
		var pd = [&PlanData](data_)
		pd.timer:init()

		var params = [&double](params_)

		--unpackstruct(pd.images) = [util.getImages(PlanData, images)]
		pd.parameters = [util.getParameters(problemSpec, images, edgeValues,params_)]

		var nIterations = 10	--non-linear iterations
		var lIterations = 10	--linear iterations
		var bIterations = 10	--block iterations
		
		for nIter = 0, nIterations do
            --var startCost = gpu.computeCost(pd, pd.images.unknown)
			--logSolver("iteration %d, cost=%f", nIter, startCost)
			
			var o : int = 0
			for lIter = 0, lIterations do
				gpu.PCGStepBlock(pd, offsetX[o], offsetY[o], bIterations)
				gpu.PCGLinearUpdateBlock(pd)
				o = (o+1)%8
			end	
			
		end

		pd.timer:evaluate()
		pd.timer:cleanup()
	end

	local terra makePlan() : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl

		pd.delta:initGPU()
		
		--var err = C.cudaMallocManaged([&&opaque](&(pd.scratchF)), sizeof(float), C.cudaMemAttachGlobal)
		--if err ~= 0 then C.printf("cudaMallocManaged error: %d", err) end
		return &pd.plan
	end
	return makePlan
end
