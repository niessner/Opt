
local S = require("std")
local util = require("util")
local C = util.C
local Timer = util.Timer
local positionForValidLane = util.positionForValidLane
local deadThread = util.deadThread

local gpuMath = util.gpuMath

local function noHeader(pd)
	return quote end
end

local function noFooter(pd)
	return quote end
end


local FLOAT_EPSILON = `0.000001f
-- GAUSS NEWTON (non-block version)
return function(problemSpec, vars)
	local BLOCK_SIZE 				=  problemSpec:BlockSize()

	local unknownElement = problemSpec:UnknownType().metamethods.typ
	local unknownType = problemSpec:UnknownType()
	
	local struct PlanData(S.Object) {
		plan : opt.Plan
		parameters : problemSpec:ParameterType(false)	--get the non-blocked version
		scratchF : &float
		debugDumpImage : &float
		debugCostImage : &float
		debugJTJImage : &float
		debugJTFImage : &float
		debugPreImage : &float


		delta : problemSpec:UnknownType()	--current linear update to be computed -> num vars
		r : problemSpec:UnknownType()		--residuals -> num vars	--TODO this needs to be a 'residual type'
		z : problemSpec:UnknownType()		--preconditioned residuals -> num vars	--TODO this needs to be a 'residual type'
		p : problemSpec:UnknownType()		--decent direction -> num vars
		Ap_X : problemSpec:UnknownType()	--cache values for next kernel call after A = J^T x J x p -> num vars
		preconditioner : problemSpec:UnknownType() --pre-conditioner for linear system -> num vars
		--rDotzOld : problemSpec:UnknownType()	--Old nominator (denominator) of alpha (beta) -> num vars	
		rDotzOld : problemSpec:InternalImage(float, unknownType.metamethods.W, unknownType.metamethods.H, false)	--Old nominator (denominator) of alpha (beta) -> num vars	
		
		scanAlpha : &float					-- tmp variable for alpha scan
		scanBeta : &float					-- tmp variable for alpha scan
		
		timer : Timer
		nIter : int				--current non-linear iter counter
		nIterations : int		--non-linear iterations
		lIterations : int		--linear iterations
	}
	
	
	
	local terra isBlockOnBoundary(gi : int, gj : int, width : uint, height : uint) : bool
		-- TODO meta program the bad based on the block size and stencil;
		-- TODO assert padX > stencil
		
		
		var padX : int
		var padY : int
		escape
			local blockSize = problemSpec:BlockSize()
		
			emit quote
				padX = int(blockSize)
				padY = int(blockSize)
			end
		end
		
		if gi - padX < 0 or gi + padX >= width or 
		   gj - padY < 0 or gj + padY >= height then
			return true
		end
		
		return false
	end
	

	local kernels = {}
	kernels.PCGInit1 = function(data)
		local terra PCGInit1GPU(pd : data.PlanData)
			var d = 0.0f -- init for out of bounds lanes
			var w : int, h : int
			if [positionForValidLane(pd, "X", `&w, `&h, problemSpec)] then
				-- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0
								
				var residuum : unknownElement = 0.0f
				var pre : unknownElement = 0.0f	
				if (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then
								
					if true or isBlockOnBoundary(w, h, pd.parameters.X:W(), pd.parameters.X:H()) then
						residuum, pre = data.problemSpec.functions.evalJTF.boundary(w, h, w, h, pd.parameters)
					else 
						--residuum, pre = data.problemSpec.functions.evalJTF.interior(w, h, w, h, pd.parameters)
					end
					residuum = -residuum
					if not [deadThread(problemSpec)] then
						pd.r(w, h) = residuum
					end
				end
				
				if not [deadThread(problemSpec)] then
					pd.preconditioner(w, h) = pre
					var p = pre*residuum	-- apply pre-conditioner M^-1			   
					pd.p(w, h) = p
				
					d = util.Dot(residuum,p) 
				end
				
			end 
			d = util.warpReduce(d)	
			if (util.laneid() == 0) then
				util.atomicAdd(pd.scanAlpha, d)
			end
		end
		return { kernel = PCGInit1GPU, header = noHeader, footer = noFooter, mapMemberName = "X" }
	end
	
	kernels.PCGInit2 = function(data)
		local terra PCGInit2GPU(pd : data.PlanData)
			var w : int, h : int
			if [positionForValidLane(pd, "X", `&w, `&h, problemSpec)] and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) and (not [deadThread(problemSpec)]) then
				pd.rDotzOld(w,h) = pd.scanAlpha[0]
				pd.delta(w,h) = 0.0f	--TODO check if we need that
			end
		end
		return { kernel = PCGInit2GPU, header = noHeader, footer = noFooter, mapMemberName = "X" }
	end
	
	kernels.PCGStep1 = function(data)
		local terra PCGStep1GPU(pd : data.PlanData)
			var d = 0.0f
			var w : int, h : int
			if [positionForValidLane(pd, "X", `&w, `&h, problemSpec)] and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then
				var tmp : unknownElement = 0.0f
				 -- A x p_k  => J^T x J x p_k 
				if true or isBlockOnBoundary(w, h, pd.parameters.X:W(), pd.parameters.X:H()) then
					tmp = data.problemSpec.functions.applyJTJ.boundary(w, h, w, h, pd.parameters, pd.p)
				else 
					tmp = data.problemSpec.functions.applyJTJ.interior(w, h, w, h, pd.parameters, pd.p)
				end
				if not [deadThread(problemSpec)] then
					pd.Ap_X(w, h) = tmp								  -- store for next kernel call
					--d = pd.p(w, h)*tmp					              -- x-th term of denominator of alpha
					d = util.Dot(pd.p(w,h),tmp)
				end
			end
			if [deadThread(problemSpec)] then
				d = 0.0
			end
			d = util.warpReduce(d)
			if (util.laneid() == 0) then
				util.atomicAdd(pd.scanAlpha, d)
			end
		end
		return { kernel = PCGStep1GPU, header = noHeader, footer = noFooter, mapMemberName = "X" }
	end
	
	kernels.PCGStep2 = function(data)
		local terra PCGStep2GPU(pd : data.PlanData)
			var b = 0.0f 
			var w : int, h : int
			if [positionForValidLane(pd, "X", `&w, `&h, problemSpec)]  and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) and (not [deadThread(problemSpec)]) then
				-- sum over block results to compute denominator of alpha
				var dotProduct : float = pd.scanAlpha[0]
				var alpha = 0.0f
				
				-- update step size alpha
				if dotProduct > FLOAT_EPSILON then alpha = pd.rDotzOld(w, h)/dotProduct end 
			
				pd.delta(w, h) = pd.delta(w, h)+alpha*pd.p(w,h)		-- do a decent step
				
				var r = pd.r(w,h)-alpha*pd.Ap_X(w,h)				-- update residuum
				pd.r(w,h) = r										-- store for next kernel call
			
				var z = pd.preconditioner(w,h)*r					-- apply pre-conditioner M^-1
				pd.z(w,h) = z;										-- save for next kernel call
				
				--b = z*r;											-- compute x-th term of the nominator of beta
				b = util.Dot(z,r)
			end
			b = util.warpReduce(b)
			if (util.laneid() == 0) then
				util.atomicAdd(pd.scanBeta, b)
			end
		end
		return { kernel = PCGStep2GPU, header = noHeader, footer = noFooter, mapMemberName = "X" }
	end
	
	kernels.PCGStep3 = function(data)
		local terra PCGStep3GPU(pd : data.PlanData)			
			var w : int, h : int
			if [positionForValidLane(pd, "X", `&w, `&h, problemSpec)] and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) and (not [deadThread(problemSpec)]) then
				var rDotzNew : float =  pd.scanBeta[0]						-- get new nominator
				var rDotzOld : float = pd.rDotzOld(w,h)						-- get old denominator

				var beta : float = 0.0f			
						
				if rDotzOld > FLOAT_EPSILON then beta = rDotzNew/rDotzOld end	-- update step size beta
			
				pd.rDotzOld(w,h) = rDotzNew										-- save new rDotz for next iteration
				pd.p(w,h) = pd.z(w,h)+beta*pd.p(w,h)							-- update decent direction
			end
		end
		return { kernel = PCGStep3GPU, header = noHeader, footer = noFooter, mapMemberName = "X" }
	end
	
	kernels.PCGLinearUpdate = function(data)
		local terra PCGLinearUpdateGPU(pd : data.PlanData)
			var w : int, h : int
			if [positionForValidLane(pd, "X", `&w, `&h, problemSpec)] and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) and (not [deadThread(problemSpec)]) then
				pd.parameters.X(w,h) = pd.parameters.X(w,h) + pd.delta(w,h)
			end
		end
		return { kernel = PCGLinearUpdateGPU, header = noHeader, footer = noFooter, mapMemberName = "X" }
	end
	
	
	kernels.computeCost = function(data)
		local terra computeCostGPU(pd : data.PlanData)
			
			var cost : float = 0.0f
			var w : int, h : int
			if [positionForValidLane(pd, "X", `&w, `&h, problemSpec)]  and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then
				var params = pd.parameters				
				cost = cost + [float](data.problemSpec.functions.cost.boundary(w, h, w, h, params))
			end
			if [deadThread(problemSpec)] then
				cost = 0.0f
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

	if util.debugDumpInfo then
	kernels.dumpCostJTFAndPre = function(data)
		local terra dumpJTFAndPreGPU(pd : data.PlanData)
			var w : int, h : int
			if [positionForValidLane(pd, "X", `&w, `&h, problemSpec)] then
				-- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0
				
				var residuum : unknownElement = 0.0f
				var pre : unknownElement = 0.0f
				var cost : float = 0.0f

				cost = data.problemSpec.functions.cost.boundary(w, h, w, h, pd.parameters)
				residuum, pre = data.problemSpec.functions.evalJTF.boundary(w, h, w, h, pd.parameters)

				pd.debugCostImage[h*pd.parameters.X:W()+w] = cost
				[&unknownElement](pd.debugJTFImage)[h*pd.parameters.X:W()+w]  = residuum
				[&unknownElement](pd.debugPreImage)[h*pd.parameters.X:W()+w]  = pre
			end 
		end

		return { kernel = dumpJTFAndPreGPU, header = noHeader, footer = noFooter, mapMemberName = "X" }
	end

	kernels.dumpJTJ = function(data)
		local terra dumpJTJGPU(pd : data.PlanData)
			var d = 0.0f
			var w : int, h : int
			if [positionForValidLane(pd, "X", `&w, `&h, problemSpec)] then
				var tmp : unknownElement = 0.0f
				 -- A x p_k  => J^T x J x p_k 
				tmp = data.problemSpec.functions.applyJTJ.boundary(w, h, w, h, pd.parameters, pd.p)
				[&unknownElement](pd.debugJTJImage)[h*pd.parameters.X:W()+w] = tmp
			end
		end
		return { kernel = dumpJTJGPU, header = noHeader, footer = noFooter, mapMemberName = "X" }
	end
	end

	local gpu = util.makeGPUFunctions(problemSpec, vars, PlanData, kernels)

	---------------------------------------DEBUGGING FUNCTIONS------------------------------------------
	local terra initDebugImage(pd : &PlanData, imPtr : &&float, numChannels : int)
		var width, height = pd.parameters.X:W(), pd.parameters.X:H() 
		var numBytes : int = sizeof(float)*width*height*numChannels
		C.printf("Num bytes: %d\n", numBytes)
		var err = C.cudaMalloc([&&opaque](imPtr), numBytes)
		if err ~= 0 then C.printf("cudaMalloc error: %d", err) end
	end

	local terra initAllDebugImages(pd : &PlanData)
		C.printf("initAllDebugImages\n")
		initDebugImage(pd, &pd.debugDumpImage, 1)
		initDebugImage(pd, &pd.debugCostImage, 1)
		initDebugImage(pd, &pd.debugJTJImage, sizeof([unknownElement]) / 4)
		initDebugImage(pd, &pd.debugJTFImage, sizeof([unknownElement]) / 4)
		initDebugImage(pd, &pd.debugPreImage, sizeof([unknownElement]) / 4)
	end

	local terra debugImageWrite(pd : &PlanData, imPtr : &float, channelCount : int, filename : rawstring)
		var width : int = [int](pd.parameters.X:W())
		var height : int = [int](pd.parameters.X:H())
		var datatype : int = 0 -- floating point
		var fileHandle = C.fopen(filename, 'wb') -- b for binary
		C.fwrite(&width, sizeof(int), 1, fileHandle)
		C.fwrite(&height, sizeof(int), 1, fileHandle)
		C.fwrite(&channelCount, sizeof(int), 1, fileHandle)
		C.fwrite(&datatype, sizeof(int), 1, fileHandle)
  
		var size = sizeof(float) * [uint32](width*height*channelCount)
		var ptr = C.malloc(size)
		C.cudaMemcpy(ptr, imPtr, size, C.cudaMemcpyDeviceToHost)
		C.fwrite(ptr, sizeof(float), [uint32](width*height*channelCount), fileHandle)
	    C.fclose(fileHandle)
	    
	    C.free(ptr)

	end

	local terra debugImageWritePrefix(pd : &PlanData, imPtr : &float, channelCount : int, prefix : rawstring)
		var buffer : int8[128]
		--var suffix = "AD"
		var suffix = "optNoAD"
		C.sprintf(buffer, "%s_%s.imagedump", prefix, suffix)
		debugImageWrite(pd, imPtr, channelCount, buffer)
	end



	local terra dumpImage(pd : &PlanData, ptr: &float, name : rawstring, nIter : int, lIter : int)
		if ([util.debugDumpInfo]) then
			var buffer : int8[64]
			C.sprintf(buffer, "%s_%d_%d.imagedump", name, nIter, lIter)
			debugImageWrite(pd, ptr, 1, buffer)
		end
	end

	---------------------------------------END DEBUGGING FUNCTIONS------------------------------------------

	
	local terra init(data_ : &opaque, images : &&opaque, edgeValues : &&opaque, params_ : &&opaque, solverparams : &&opaque)
		var pd = [&PlanData](data_)
		pd.timer:init()
		pd.parameters = [util.getParameters(problemSpec, images, edgeValues,params_)]

	    pd.nIter = 0
		
		escape 
	    	if util.debugDumpInfo then
	    		emit quote
		    		C.printf("initDebugImages\n")
		    		initAllDebugImages(pd)
	    		end
	    	end
		end

		pd.nIterations = @[&int](solverparams[0])
		pd.lIterations = @[&int](solverparams[1])
	end
	local terra step(data_ : &opaque, images : &&opaque, edgeValues : &&opaque, params_ : &&opaque, solverparams : &&opaque)
		var pd = [&PlanData](data_)
		pd.parameters = [util.getParameters(problemSpec, images, edgeValues,params_)]

		escape 
	    	if util.debugDumpInfo then
	    		emit quote
	    			if pd.nIter == 0 then
		    			C.printf("dumpingCostJTFAndPre\n")
		    			gpu.dumpCostJTFAndPre(pd)
		    			C.printf("saving\n")
		    			debugImageWritePrefix(pd, pd.debugCostImage, 1, "cost")
		    			debugImageWritePrefix(pd, pd.debugJTFImage, sizeof([unknownElement]) / 4, "JTF")
		    			debugImageWritePrefix(pd, pd.debugPreImage, sizeof([unknownElement]) / 4, "Pre")
		    		end
	    		end
	    	end
		end
	
		if pd.nIter < pd.nIterations then
		    var startCost = gpu.computeCost(pd)
			logSolver("iteration %d, cost=%f\n", pd.nIter, startCost)
			
			pd.scanAlpha[0] = 0.0	--scan in PCGInit1 requires reset
			gpu.PCGInit1(pd)
			gpu.PCGInit2(pd)
			
			escape
				if util.debugDumpInfo then
		    		emit quote
		    			if pd.nIter == 0 then
			    			C.printf("dumpingJTJ\n")
			    			gpu.dumpJTJ(pd)
			    			C.printf("saving\n")
			    			debugImageWritePrefix(pd, pd.debugJTJImage, sizeof([unknownElement]) / 4, "JTJ")
			    		end
		    		end
		    	end
		    end

			for lIter = 0, pd.lIterations do	
				pd.scanAlpha[0] = 0.0	--scan in PCGStep1 requires reset
				gpu.PCGStep1(pd)
				pd.scanBeta[0] = 0.0	--scan in PCGStep2 requires reset
				gpu.PCGStep2(pd)
				gpu.PCGStep3(pd)
				
				--var currCost = gpu.computeCost(pd)
				--logSolver("after linear iteration %d, cost=%f\n", lIter, currCost)
			end
			
			gpu.PCGLinearUpdate(pd)
		    pd.nIter = pd.nIter + 1
		    return 1
		else
			escape
				if util.debugDumpInfo then
		    		emit quote
						debugImageWritePrefix(pd, [&float](pd.parameters.X.data), sizeof([unknownElement]) / 4, "result")
					end
				end
			end
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

		pd.delta:initGPU()
		pd.r:initGPU()
		pd.z:initGPU()
		pd.p:initGPU()
		pd.Ap_X:initGPU()
		pd.preconditioner:initGPU()
		pd.rDotzOld:initGPU()
		
		--TODO make this exclusively GPU
		C.cudaMallocManaged([&&opaque](&(pd.scanAlpha)), sizeof(float), C.cudaMemAttachGlobal)
		C.cudaMallocManaged([&&opaque](&(pd.scanBeta)), sizeof(float), C.cudaMemAttachGlobal)
		
		C.cudaMallocManaged([&&opaque](&(pd.scratchF)), sizeof(float), C.cudaMemAttachGlobal)
		return &pd.plan
	end
	return makePlan
end