
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
	return i >= 0 and i < width and j >= 0 and j < height
end

local terra getLinearThreadId(tId_i : int, tId_j : int) : uint
	return tId_j*BLOCK_SIZE + tId_i;
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

	local unknownElement = problemSpec:UnknownType().metamethods.typ
	
	local struct PlanData(S.Object) {
		plan : opt.Plan
		parameters : problemSpec:ParameterType(false)	--get non-blocked version
		scratchF : &float
		
		--delta : problemSpec:UnknownType(false)	--current linear update to be computed -> num vars

		timer : Timer
		nIter : int
		currOffset : int
		nIterations : int
		lIterations : int
		bIterations : int
	}
	

	local CopyToShared = terralib.memoize(function(Image,ImageBlock)
			
		local stencil = problemSpec:MaxStencil()
		local offset = stencil*problemSpec:BlockStride() + stencil
		local blockStride = problemSpec:BlockStride()
		
		return terra(blockCornerX : int32, blockCornerY : int32, image : Image, imageBlock : ImageBlock)			
			
			var numBlockThreads : int = blockDim.x * blockDim.y			
			var numVariables : int = blockStride*blockStride
			
			var baseIdx : int = threadIdx.x + threadIdx.y*blockDim.x
			for i = 0,numVariables,numBlockThreads do
				var linearIdx : int = baseIdx + i
				
				if linearIdx < numVariables then				
					var localX : int = linearIdx % blockStride - stencil
					var localY : int = linearIdx / blockStride - stencil
					
					var globalX : int = localX + blockCornerX
					var globalY : int = localY + blockCornerY

					imageBlock(localX, localY) = image:get(globalX, globalY)	--bounded check							
				end
			end			
		end
	end)
	
	local InitShared = terralib.memoize(function(ImageBlock)
			
		local stencil = problemSpec:MaxStencil()
		local offset = stencil*problemSpec:BlockStride() + stencil
		local blockStride = problemSpec:BlockStride()
		
		return terra(imageBlock : ImageBlock, value : float)			
			
			var numBlockThreads : int = blockDim.x * blockDim.y			
			var numVariables : int = blockStride*blockStride
			
			var baseIdx : int = threadIdx.x + threadIdx.y*blockDim.x
			for i = 0,numVariables,numBlockThreads do
				var linearIdx : int = baseIdx + i
				
				if linearIdx < numVariables then				
					var localX : int = linearIdx % blockStride - stencil
					var localY : int = linearIdx / blockStride - stencil
					
					imageBlock(localX, localY) = value							
				end
			end			
		end
	end)
	
	local terra isBlockOnBoundary(gi : int, gj : int, width : uint, height : uint) : bool
		-- TODO meta program the bad based on the block size and stencil;
		-- TODO assert padX > stencil
		var padX : int = int(BLOCK_SIZE)
		var padY : int = int(BLOCK_SIZE)
		 
		if gi - padX < 0 or gi + padX >= width or 
		   gj - padY < 0 or gj + padY >= height then
			return true
		end
		return false
	end
	
	local kernels = {}
	
	kernels.computeCost = function(data)
		local terra computeCostGPU(pd : data.PlanData)
		
			var W = pd.parameters.X:W()
			var H = pd.parameters.X:H()
	
			var tId_i : int = threadIdx.x -- local col idx
			var tId_j : int = threadIdx.y -- local row idx
	
	
			var gId_i : int = blockIdx.x * blockDim.x + threadIdx.x -- global col idx
			var gId_j : int = blockIdx.y * blockDim.y + threadIdx.y -- global row idx
			
			var blockCornerX : int = blockIdx.x * blockDim.x
			var blockCornerY : int = blockIdx.y * blockDim.y
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
						local blockedType = problemSpec:BlockedTypeForImageEntry(p)
						local stencil = problemSpec:MaxStencil()
						local offset = stencil*problemSpec:BlockStride() + stencil
						local shared_mem_size = (problemSpec:BlockSize()+2.0*problemSpec:MaxStencil())*(problemSpec:BlockSize()+2.0*problemSpec:MaxStencil())
						local shmem = cudalib.sharedmemory(p.type.metamethods.typ, shared_mem_size)
						
						emit quote 
							blockParams.[p.name] = [blockedType] { data = [&uint8](shmem + offset) } 
							[CopyToShared(p.type,blockedType)](blockCornerX, blockCornerY, pd.parameters.[p.name], blockParams.[p.name])
						end
					end
				end
			end
			
			__syncthreads()
			
			
			var cost = 0.0f
			var w : int, h : int
			if positionForValidLane(pd, "X", &w, &h) and (not [problemSpec:EvalExclude(tId_i, tId_j, gId_i, gId_j,`blockParams)]) then
			--if positionForValidLane(pd, "X", &w, &h) then
				if true or isBlockOnBoundary(gId_i, gId_j, W, H) then
					cost = [float](data.problemSpec.functions.cost.boundary(tId_i, tId_j, gId_i, gId_j, blockParams))
				else
					--cost = [float](data.problemSpec.functions.cost.interior(tId_i, tId_j, gId_i, gId_j, blockParams))
				end
				--cost = [float](data.problemSpec.functions.cost_global.boundary(gId_i, gId_j, gId_i, gId_j, pd.parameters))
			end

			cost = util.warpReduce(cost)
			if (util.laneid() == 0) then
				util.atomicAdd(pd.scratchF, cost)
			end
			
			
		end
		local function header(pd)
			return quote @pd.scratchF = 0.0f end
		end
		local function footer(pd)
			return quote return @pd.scratchF end
		end
		
		return { kernel = computeCostGPU, header = header, footer = footer, mapMemberName = "X" }
	end

	kernels.PCGStepBlock = function(data)
		
		--local patchBucket = cudalib.sharedmemory(float, SHARED_MEM_SIZE_VARIABLES)
	
	
		local terra PCGStepBlockGPU(pd : data.PlanData, ox : int, oy : int, nBlockIterations : uint)
			
			var W = pd.parameters.X:W()
			var H = pd.parameters.X:H()
	
			var tId_i : int = threadIdx.x -- local col idx
			var tId_j : int = threadIdx.y -- local row idx	
	
			var gId_i : int = blockIdx.x*blockDim.x + threadIdx.x - ox -- global col idx
			var gId_j : int = blockIdx.y*blockDim.y + threadIdx.y - oy -- global row idx
			
			var blockCornerX : int = blockIdx.x*blockDim.x - ox
			var blockCornerY : int = blockIdx.y*blockDim.y - oy
			var blockParams : problemSpec:ParameterType(true)
			var P : problemSpec:UnknownType(true)
			
			
			var patchBucket : &float
			escape 
			--local patchBucket = cudalib.sharedmemory(float, math:ceil(SHARED_MEM_SIZE_VARIABLES / util.warpSize))
				local shmem = cudalib.sharedmemory(float,SHARED_MEM_SIZE_VARIABLES)
				emit quote
					patchBucket = shmem
				end
			end
			
			-- load everything into shared memory
			escape				
				for i,p in ipairs(problemSpec.parameters) do
					if p.kind ~= "image" then
						emit quote
							blockParams.[p.name] = pd.parameters.[p.name]
						end
					else 
						local blockedType = problemSpec:BlockedTypeForImageEntry(p)
						local stencil = problemSpec:MaxStencil()
						local offset = stencil*problemSpec:BlockStride() + stencil
						local shared_mem_size = (problemSpec:BlockSize()+2*problemSpec:MaxStencil())*(problemSpec:BlockSize()+2*problemSpec:MaxStencil())
						local shmem = cudalib.sharedmemory(p.type.metamethods.typ, shared_mem_size)
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
				local stencil = problemSpec:MaxStencil()
				local offset = stencil*problemSpec:BlockStride() + stencil
				local shared_mem_size = (problemSpec:BlockSize()+2*problemSpec:MaxStencil())*(problemSpec:BlockSize()+2*problemSpec:MaxStencil())
				local shmem = cudalib.sharedmemory(problemSpec:UnknownType().metamethods.typ, shared_mem_size)		--pre-conditioner
				emit quote
					P = [blockedType] { data = [&uint8](shmem + offset) } 
					[InitShared(blockedType)](P, 0.0f)
				end
			end
			
			var Delta : problemSpec:UnknownType().metamethods.typ = 0.0f
			var R : problemSpec:UnknownType().metamethods.typ
			var Z : problemSpec:UnknownType().metamethods.typ
			var Pre : problemSpec:UnknownType().metamethods.typ
			var RDotZOld : float
			var AP : problemSpec:UnknownType().metamethods.typ

			__syncthreads()	--after here, everything should be in shared memory
				
			var isValidPixel : bool = isInsideImage(gId_i, gId_j, W, H) and (not [problemSpec:EvalExclude(tId_i, tId_j, gId_i, gId_j,`blockParams)])
			--//////////////////////////////////////////////////////////////////////////////////////////
			--// Initialize linear patch systems
			--//////////////////////////////////////////////////////////////////////////////////////////
			
			var d : float = 0.0f
			
			if isValidPixel then
				-- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0
				if true or isBlockOnBoundary(gId_i, gId_j, W, H) then
					R, Pre = data.problemSpec.functions.evalJTF.boundary(tId_i, tId_j, gId_i, gId_j, blockParams)		
				else 
					--R, Pre = data.problemSpec.functions.evalJTF.interior(tId_i, tId_j, gId_i, gId_j, blockParams)		
				end
				R = -R

				var preRes : unknownElement = Pre*R		-- apply pre-conditioner M^-1 
				P(tId_i, tId_j) = preRes		-- save for later
				--d = R*preRes
				d = util.Dot(R, preRes)
			end
						
			
			patchBucket[getLinearThreadId(tId_i, tId_j)] = d;											   -- x-th term of nomimator for computing alpha and denominator for computing beta
			
			__syncthreads()
			blockReduce(patchBucket, getLinearThreadId(tId_i, tId_j), SHARED_MEM_SIZE_VARIABLES)
			__syncthreads()	-- TODO I'm not quire sure if we need this one

			if isValidPixel then 
				RDotZOld = patchBucket[0]							   -- read result for later on
			end
	
			__syncthreads()	
			
			
			--///////////////////////
			--Do patch PCG iterations
			--///////////////////////
			for patchIter = 0,nBlockIterations,1 do 
			
				var currentP : problemSpec:UnknownType().metamethods.typ = P(tId_i, tId_j)				
				var d : float = 0.0f
					
				if isValidPixel then
					-- A x p_k  => J^T x J x p_k 
					if true or isBlockOnBoundary(gId_i, gId_j, W, H) then
						AP = data.problemSpec.functions.applyJTJ.boundary(tId_i, tId_j, gId_i, gId_j, blockParams, P) 
					else 
						--AP = data.problemSpec.functions.applyJTJ.interior(tId_i, tId_j, gId_i, gId_j, blockParams, P) 
					end
					--d = currentP*AP
					d = util.Dot(currentP, AP)
					-- x-th term of denominator of alpha
				end
			
				patchBucket[getLinearThreadId(tId_i, tId_j)] = d
		
				__syncthreads()
				blockReduce(patchBucket, getLinearThreadId(tId_i, tId_j), SHARED_MEM_SIZE_VARIABLES)	
				__syncthreads()
	
				var dotProduct : float = patchBucket[0]
				
				var b : float = 0.0f		
				if isValidPixel then
					var alpha : float = 0.0f
					
					if dotProduct > FLOAT_EPSILON then	-- update step size alpha
						alpha = RDotZOld/dotProduct
					end
					Delta = Delta+alpha*currentP	-- do a decent step		
					R = R-alpha*AP					-- update residuum						
					Z = Pre*R						-- apply pre-conditioner M^-1
					--b = Z*R							-- compute x-th term of the nominator of beta
					b = util.Dot(Z,R)
				end
			
				__syncthreads() -- Only write if every thread in the block has has read bucket[0]
		
				patchBucket[getLinearThreadId(tId_i, tId_j)] = b;
		
				__syncthreads()
				blockReduce(patchBucket, getLinearThreadId(tId_i, tId_j), SHARED_MEM_SIZE_VARIABLES)	-- sum over x-th terms to compute nominator of beta inside this block
				__syncthreads()
				
		
				if isValidPixel then
					var rDotzNew : float = patchBucket[0]	-- get new nominator
								
					var beta : float = 0.0f														 
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
			
			
			--if isInsideImage(gId_i, gId_j, W, H) then 
			--	pd.delta(gId_i, gId_j) = Delta
			--end
			
			if isValidPixel then 
				pd.parameters.X(gId_i,gId_j) = pd.parameters.X(gId_i,gId_j) + Delta
			end
			
		end
		return { kernel = PCGStepBlockGPU, header = noHeader, footer = noFooter, mapMemberName = "X" }
	end
	
	-- kernels.PCGLinearUpdateBlock = function(data)
		-- local terra PCGLinearUpdateBlockGPU(pd : data.PlanData, ox : int, oy : int)
		
			-- var W = pd.parameters.X:W()
			-- var H = pd.parameters.X:H()
	
			-- var tId_i : int = threadIdx.x -- local col idx
			-- var tId_j : int = threadIdx.y -- local row idx
	
			-- var gId_i : int = blockIdx.x * blockDim.x + threadIdx.x - ox -- global col idx
			-- var gId_j : int = blockIdx.y * blockDim.y + threadIdx.y - oy -- global row idx
					
			-- if isInsideImage(gId_i, gId_j, W, H) then 
				-- pd.parameters.X(gId_i,gId_j) = pd.parameters.X(gId_i,gId_j) + pd.delta(gId_i,gId_j)
			-- end
		
		-- end
		-- return { kernel = PCGLinearUpdateBlockGPU, header = noHeader, footer = noFooter,  mapMemberName = "X" }
	-- end
	
	local gpu = util.makeGPUFunctions(problemSpec, vars, PlanData, kernels)


	local terra init(data_ : &opaque, images : &&opaque, edgeValues : &&opaque, params_ : &&opaque, solverparams : &&opaque)
		var pd = [&PlanData](data_)
		pd.timer:init()
		pd.parameters = [util.getParameters(problemSpec, images, edgeValues,params_)]
	    
		pd.nIter = 0
		pd.currOffset = 0
		
		pd.nIterations = @[&int](solverparams[0])
		pd.lIterations = @[&int](solverparams[1])
		pd.bIterations = @[&int](solverparams[2])
	end
	
	local terra step(data_ : &opaque, images : &&opaque, edgeValues : &&opaque, params_ : &&opaque,  solverparams : &&opaque)
		var pd = [&PlanData](data_)
		pd.parameters = [util.getParameters(problemSpec, images, edgeValues,params_)]
		if pd.nIter < pd.nIterations then
		
			var startCost = gpu.computeCost(pd)
			logSolver("iteration %d, cost=%f\n", pd.nIter, startCost)	
				
			for lIter = 0, pd.lIterations do
			    --var startCost = gpu.computeCost(pd)
				--logSolver("iteration %d, cost=%f\n", lIter, startCost)	
			
				var oX : int = offsetX[pd.currOffset]
				var oY : int = offsetY[pd.currOffset]				
			
				--C.printf("offset: (%d | %d)\n", oX, oY)
				gpu.PCGStepBlock(pd, oX, oY, pd.bIterations)
				--gpu.PCGLinearUpdateBlock(pd, oX, oY)
				pd.currOffset = (pd.currOffset+1)%8
				C.cudaDeviceSynchronize()				
			end
			pd.nIter = pd.nIter + 1
			return 1
		else
			var finalCost = gpu.computeCost(pd)
			logSolver("final cost=%f\n", finalCost)
		    pd.timer:evaluate()
		    pd.timer:cleanup()
		    return 0
		end
	end

	local terra makePlan() : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.init,pd.plan.step = init,step

		--pd.delta:initGPU()
		
		C.cudaMallocManaged([&&opaque](&(pd.scratchF)), sizeof(float), C.cudaMemAttachGlobal)
		--var err = C.cudaMallocManaged([&&opaque](&(pd.scratchF)), sizeof(float), C.cudaMemAttachGlobal)
		--if err ~= 0 then C.printf("cudaMallocManaged error: %d", err) end
		return &pd.plan
	end
	return makePlan
end
