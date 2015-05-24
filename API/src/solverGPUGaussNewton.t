
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



local FLOAT_EPSILON = `0.000001f
-- GAUSS NEWTON (non-block version)
return function(problemSpec, vars)

	local struct PlanData(S.Object) {
		plan : opt.Plan
		parameters : problemSpec:ParameterType(false)	--get the non-blocked version
		scratchF : &float
		debugDumpImage : &float -- imageWidth*imageHeight floats
		
		delta : problemSpec:UnknownType()	--current linear update to be computed -> num vars
		r : problemSpec:UnknownType()		--residuals -> num vars	--TODO this needs to be a 'residual type'
		z : problemSpec:UnknownType()		--preconditioned residuals -> num vars	--TODO this needs to be a 'residual type'
		p : problemSpec:UnknownType()		--decent direction -> num vars
		Ap_X : problemSpec:UnknownType()	--cache values for next kernel call after A = J^T x J x p -> num vars
		preconditioner : problemSpec:UnknownType() --pre-conditioner for linear system -> num vars
		rDotzOld : problemSpec:UnknownType()	--Old nominator (denominator) of alpha (beta) -> num vars	

		scanAlpha : &float					-- tmp variable for alpha scan
		scanBeta : &float					-- tmp variable for alpha scan
		
		timer : Timer
	}
	
	local kernels = {}
	
	kernels.PCGInit1 = function(data)
		local terra PCGInit1GPU(pd : &data.PlanData)
			var d = 0.0f -- init for out of bounds lanes
			var w : int, h : int
			if positionForValidLane(pd, "X", &w, &h) then
				var residuum = -data.problemSpec.functions.gradient.boundary(w, h, w, h, pd.parameters)	-- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 
				pd.r(w, h) = residuum

				-- TODO pd.precondition(w,h) needs to computed somehow (ideally in the gradient?
				-- TODO: don't let this be 0
				pd.preconditioner(w, h) = 1 --data.problemSpec.gradientPreconditioner(w, h)	-- TODO fix this hack... the pre-conditioner needs to be the diagonal of JTJ
				--pd.preconditioner(w,h) = data.problemSpec.gradientPreconditioner(w,h)
				var p = pd.preconditioner(w, h)*residuum				   -- apply pre-conditioner M^-1
				pd.p(w, h) = p
			
				d = residuum*p;										   -- x-th term of nominator for computing alpha and denominator for computing beta
			end 
			d = util.warpReduce(d)	
			if (util.laneid() == 0) then
				util.atomicAdd(pd.scanAlpha, d)
			end
		end
		return { kernel = PCGInit1GPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "X" }
	end
	
	kernels.PCGInit2 = function(data)
		local terra PCGInit2GPU(pd : &data.PlanData)
			var w : int, h : int
			if positionForValidLane(pd, "X", &w, &h) then
				pd.rDotzOld(w,h) = pd.scanAlpha[0]
				pd.delta(w,h) = 0.0f
			end
		end
		return { kernel = PCGInit2GPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "X" }
	end
	
	kernels.PCGStep1 = function(data)
		local terra PCGStep1GPU(pd : &data.PlanData)
			var d = 0.0f
			var w : int, h : int
			if positionForValidLane(pd, "X", &w, &h) then
				var tmp = data.problemSpec.functions.applyJTJ.boundary(w, h, w, h, pd.parameters, pd.p) -- A x p_k  => J^T x J x p_k 
				pd.Ap_X(w, h) = tmp								  -- store for next kernel call
				d = pd.p(w, h)*tmp					              -- x-th term of denominator of alpha
			end
			d = util.warpReduce(d)
			if (util.laneid() == 0) then
				util.atomicAdd(pd.scanAlpha, d)
			end
		end
		return { kernel = PCGStep1GPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "X" }
	end
	
	kernels.PCGStep2 = function(data)
		local terra PCGStep2GPU(pd : &data.PlanData)
			var b = 0.0f 
			var w : int, h : int
			if positionForValidLane(pd, "X", &w, &h) then
				-- sum over block results to compute denominator of alpha
				var dotProduct = pd.scanAlpha[0]
				var alpha = 0.0f
				
				-- update step size alpha
				if dotProduct > FLOAT_EPSILON then alpha = pd.rDotzOld(w, h)/dotProduct end 
			
				pd.delta(w, h) = pd.delta(w, h)+alpha*pd.p(w,h)		-- do a decent step
				
				var r = pd.r(w,h)-alpha*pd.Ap_X(w,h)				-- update residuum
				pd.r(w,h) = r										-- store for next kernel call
			
				var z = pd.preconditioner(w,h)*r						-- apply pre-conditioner M^-1
				pd.z(w,h) = z;										-- save for next kernel call
				
				b = z*r;											-- compute x-th term of the nominator of beta
			end
			b = util.warpReduce(b)
			if (util.laneid() == 0) then
				util.atomicAdd(pd.scanBeta, b)
			end
		end
		return { kernel = PCGStep2GPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "X" }
	end
	
	kernels.PCGStep3 = function(data)
		local terra PCGStep3GPU(pd : &data.PlanData)			
			var w : int, h : int
			if positionForValidLane(pd, "X", &w, &h) then
				var rDotzNew =  pd.scanBeta[0]									-- get new nominator
				var rDotzOld = pd.rDotzOld(w,h)									-- get old denominator

				var beta = 0.0f														 
				if rDotzOld > FLOAT_EPSILON then beta = rDotzNew/rDotzOld end	-- update step size beta
			
				pd.rDotzOld(w,h) = rDotzNew										-- save new rDotz for next iteration
				pd.p(w,h) = pd.z(w,h)+beta*pd.p(w,h)							-- update decent direction
			end
		end
		return { kernel = PCGStep3GPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "X" }
	end
	
	kernels.PCGLinearUpdate = function(data)
		local terra PCGLinearUpdateGPU(pd : &data.PlanData)
			var w : int, h : int
			if positionForValidLane(pd, "X", &w, &h) then
				pd.parameters.X(w,h) = pd.parameters.X(w,h) + pd.delta(w,h)
			end
		end
		return { kernel = PCGLinearUpdateGPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "X" }
	end

	kernels.computeCostImage = function(data)
		local terra computeCostImageGPU(pd : &data.PlanData)
			var cost = 0.0f
			var w : int, h : int
			if util.positionForValidLane(pd, "X", &w, &h) then
				var params = pd.parameters
				cost = [float](data.problemSpec.functions.cost.boundary(w, h, w, h, params))
				pd.debugDumpImage[h*pd.parameters.X:W() + w] = cost
			end
		end
		return { kernel = computeCostImageGPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "X" }
	end

	kernels.finitify = function(data)
		local terra finitifyGPU(pd : &data.PlanData)
			
			var w : int, h : int
			if util.positionForValidLane(pd, "X", &w, &h) then
				var result = ([&float](pd.parameters.Y.data))[h*pd.parameters.X:W() + w]
				if result < 0 then 
					result = -10000.0
				end
				([&float](pd.parameters.Y.data))[h*pd.parameters.X:W() + w] = result

				result = ([&float](pd.parameters.Z.data))[h*pd.parameters.X:W() + w]
				if result < 0 then 
					result = -10000.0
				end
				([&float](pd.parameters.Z.data))[h*pd.parameters.X:W() + w] = result
			end
		end
		return { kernel = finitifyGPU, header = noHeader, footer = noFooter, params = {}, mapMemberName = "X" }
	end



	local gpu = util.makeGPUFunctions(problemSpec, vars, PlanData, kernels)


	
	local terra initDebugDumpImage(pd : &PlanData)
		var width, height = pd.parameters.X:W(), pd.parameters.X:H() 
		var err = C.cudaMalloc([&&opaque](&(pd.debugDumpImage)), sizeof(float)*width*height)
		if err ~= 0 then C.printf("cudaMalloc error: %d", err) end
	end

	local terra debugDumpImageWrite(pd : &PlanData, filename : rawstring)
		var width : int = [int](pd.parameters.X:W())
		var height : int = [int](pd.parameters.X:H())
		var channelCount : int = 1
		var datatype : int = 0 -- floating point
		var fileHandle = C.fopen(filename, 'wb') -- b for binary
		C.fwrite(&width, sizeof(int), 1, fileHandle)
		C.fwrite(&height, sizeof(int), 1, fileHandle)
		C.fwrite(&channelCount, sizeof(int), 1, fileHandle)
		C.fwrite(&datatype, sizeof(int), 1, fileHandle)
  
		var size = sizeof(float) * [uint64](width*height)
		var ptr = C.malloc(size)
		C.cudaMemcpy(ptr, pd.debugDumpImage, size, C.cudaMemcpyDeviceToHost)
		C.fwrite(ptr, sizeof(float), [uint64](width*height), fileHandle)
	    C.fclose(fileHandle)
	    --C.printf("width, height: %d, %d\n", width, height)
	    for j = 0,height do
	    	for i = 0,width do
	    		--C.printf("(%d, %d): %f\n", i,j, ([&float](ptr))[j*width + i])
	    	end
	    end
	    
	    C.free(ptr)

	end

	local terra dumpCostImage(pd : &PlanData, name : rawstring, nIter : int)
		if ([util.debugDumpInfo]) then
			gpu.computeCostImage(pd)
			var buffer : int8[64]
			C.sprintf(buffer, "%s%d.imagedump", name, nIter)
			debugDumpImageWrite(pd, buffer)
		end
	end

	local terra dumpUnknownImage(pd : &PlanData, name : rawstring, nIter : int)
		if ([util.debugDumpInfo]) then
			var width, height = pd.parameters.X:W(), pd.parameters.X:H() 
			C.cudaMemcpy(pd.debugDumpImage, pd.parameters.X.data, sizeof(float)*width*height, C.cudaMemcpyDeviceToDevice)
			C.cudaDeviceSynchronize()
			var buffer : int8[64]
			C.sprintf(buffer, "%s%d.imagedump", name, nIter)
			debugDumpImageWrite(pd, buffer)
		end
	end

	local terra dumpImage(pd : &PlanData, ptr: &float, name : rawstring, nIter : int, lIter : int)
		if ([util.debugDumpInfo]) then
			var width, height = pd.parameters.X:W(), pd.parameters.X:H() 
			C.cudaMemcpy(pd.debugDumpImage, ptr, sizeof(float)*width*height, C.cudaMemcpyDeviceToDevice)
			C.cudaDeviceSynchronize()
			var buffer : int8[64]
			C.sprintf(buffer, "%s_%d_%d.imagedump", name, nIter, lIter)
			debugDumpImageWrite(pd, buffer)
		end
	end

	local terra dumpConstImages(pd : &PlanData)
		if ([util.debugDumpInfo]) then
			var width, height = pd.parameters.X:W(), pd.parameters.X:H() 
			C.cudaMemcpy(pd.debugDumpImage, pd.parameters.Y.data, sizeof(float)*width*height, C.cudaMemcpyDeviceToDevice)
			C.cudaDeviceSynchronize()
			debugDumpImageWrite(pd, "initialDepth.imagedump")
			C.cudaMemcpy(pd.debugDumpImage, pd.parameters.I.data, sizeof(float)*width*height, C.cudaMemcpyDeviceToDevice)
			C.cudaDeviceSynchronize()
			debugDumpImageWrite(pd, "targetIntensity.imagedump")
			C.cudaMemcpy(pd.debugDumpImage, pd.parameters.Z.data, sizeof(float)*width*height, C.cudaMemcpyDeviceToDevice)
			C.cudaDeviceSynchronize()
			debugDumpImageWrite(pd, "previousDepth.imagedump")
		end
	end

	local terra hackShapeFromShadingInit(pd : &PlanData)
		gpu.finitify(pd) -- hack for SFS
		var width, height = pd.parameters.X:W(), pd.parameters.X:H() 
		C.cudaMemcpy(pd.parameters.X.data, pd.parameters.Y.data, sizeof(float)*width*height, C.cudaMemcpyDeviceToDevice)
		C.cudaDeviceSynchronize()
	end

	local terra dumpAllInfo(pd : &PlanData, nIter : int)
		dumpUnknownImage(pd, "unknown", nIter)
		dumpCostImage(pd, "costImage", nIter)
	end

	local terra impl(data_ : &opaque, images : &&opaque, edgeValues : &&opaque, params_ : &&opaque)
		var pd = [&PlanData](data_)

		pd.timer:init()

		pd.parameters = [util.getParameters(problemSpec, images, edgeValues,params_)]

		var nIterations = 10	--non-linear iterations
		var lIterations = 10	--linear iterations
		
		
		initDebugDumpImage(pd)
		--hackShapeFromShadingInit(pd) -- hack for SFS
		dumpConstImages(pd)
		
		for nIter = 0, nIterations do
			dumpAllInfo(pd, nIter)
            var startCost = gpu.computeCost(pd, pd.parameters.X)
			logSolver("iteration %d, cost=%f\n", nIter, startCost)

			pd.scanAlpha[0] = 0.0	--scan in PCGInit1 requires reset
			gpu.PCGInit1(pd)
			dumpImage(pd, [&float](pd.r.data), "residuum", nIter, -1)
			dumpImage(pd, [&float](pd.p.data), "p", nIter, -1)
			--var a = pd.scanAlpha[0]
			--C.printf("Alpha %15.15f\n", a)
			--break
			gpu.PCGInit2(pd)
			
			for lIter = 0, lIterations do
				pd.scanAlpha[0] = 0.0	--scan in PCGStep1 requires reset
				gpu.PCGStep1(pd)
				dumpImage(pd, [&float](pd.Ap_X.data), "Ap_X", nIter, lIter)
				pd.scanBeta[0] = 0.0	--scan in PCGStep2 requires reset
				gpu.PCGStep2(pd)
				dumpImage(pd, [&float](pd.delta.data), "delta", nIter, lIter)
				dumpImage(pd, [&float](pd.r.data), "residuum", nIter, lIter)
				dumpImage(pd, [&float](pd.z.data), "z", nIter, lIter)
				gpu.PCGStep3(pd)
				dumpImage(pd, [&float](pd.rDotzOld.data), "rDotzOld", nIter, lIter)
				dumpImage(pd, [&float](pd.p.data), "p", nIter, lIter)
			end
			
			gpu.PCGLinearUpdate(pd)
		end
		dumpAllInfo(pd, nIterations)
		var startCost = gpu.computeCost(pd, pd.parameters.X)
		logSolver("Final, cost=%f", startCost)
		
		pd.timer:evaluate()
		pd.timer:cleanup()
	end

	local terra makePlan() : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl

		pd.delta:initGPU()
		pd.r:initGPU()
		pd.z:initGPU()
		pd.p:initGPU()
		pd.Ap_X:initGPU()
		pd.preconditioner:initGPU()
		pd.rDotzOld:initGPU()

		var err = C.cudaMallocManaged([&&opaque](&(pd.scratchF)), sizeof(float), C.cudaMemAttachGlobal)
		if err ~= 0 then C.printf("cudaMallocManaged error: %d", err) end


		--TODO make this exclusively GPU
		C.cudaMallocManaged([&&opaque](&(pd.scanAlpha)), sizeof(float), C.cudaMemAttachGlobal)
		C.cudaMallocManaged([&&opaque](&(pd.scanBeta)), sizeof(float), C.cudaMemAttachGlobal)
		return &pd.plan
	end
	return makePlan
end