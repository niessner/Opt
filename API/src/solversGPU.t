
local S = require("std")
local util = require("util")
local C = util.C

solversGPU = {}

solversGPU.gradientDescentGPU = function(Problem, tbl, vars)

	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		gradStore : vars.unknownType
		scratchF : &float
		scratchD : &double
		dims : int64[#vars.dims + 1]
	}
	
	local cuda = {}

	terra cuda.computeGradient(pd : PlanData, [vars.imagesAll])
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		--printf("%d, %d\n", w, h)

		if w < pd.gradW and h < pd.gradH then
			pd.gradStore(w,h) = tbl.gradient(w, h, vars.imagesAll)
		end
	end

	terra cuda.updatePosition(pd : PlanData, learningRate : double, maxDelta : &double, [vars.imagesAll])
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		--printf("%d, %d\n", w, h)
		
		if w < pd.gradW and h < pd.gradH then
			var addr = &vars.unknownImage(w,h)
			var delta = learningRate * pd.gradStore(w, h)
			@addr = @addr - delta

			delta = delta * delta
			var deltaD : double = delta
			var deltaI64 = @[&int64](&deltaD)
			--printf("delta=%f, deltaI=%d\n", delta, deltaI)
			terralib.asm(terralib.types.unit,"red.global.max.u64 [$0],$1;", "l,l", true, maxDelta, deltaI64)
		end
	end

	terra cuda.costSum(pd : PlanData, sum : &float, [vars.imagesAll])
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		if w < pd.gradW and h < pd.gradH then
			var v = [float](tbl.cost.fn(w, h, vars.imagesAll))
			terralib.asm(terralib.types.unit,"red.global.add.f32 [$0],$1;","l,f",true,sum,v)
		end
	end
	
	cuda = terralib.cudacompile(cuda, false)
	 
	local terra totalCost(pd : &PlanData, [vars.imagesAll])
		var launch = terralib.CUDAParams { (pd.gradW - 1) / 32 + 1, (pd.gradH - 1) / 32 + 1,1, 32,32,1, 0, nil }

		@pd.scratchF = 0.0f
		cuda.costSum(&launch, @pd, pd.scratchF, [vars.imagesAll])
		C.cudaDeviceSynchronize()

		return @pd.scratchF
	end

	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)
		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dims = pd.dims

		var [vars.imagesAll] = [util.getImages(vars,imageBindings,dims)]

		-- TODO: parameterize these
		var initialLearningRate = 0.01
		var maxIters = 5000
		var tolerance = 1e-10

		-- Fixed constants (these do not need to be parameterized)
		var learningLoss = 0.8
		var learningGain = 1.1
		var minLearningRate = 1e-25

		var learningRate = initialLearningRate

		for iter = 0,maxIters do

			var startCost = totalCost(pd, vars.imagesAll)
			log("iteration %d, cost=%f, learningRate=%f\n", iter, startCost, learningRate)
			
			--
			-- compute the gradient
			--
			var launch = terralib.CUDAParams { (pd.gradW - 1) / 32 + 1, (pd.gradH - 1) / 32 + 1,1, 32,32,1, 0, nil }

			--[[
			{ "gridDimX", uint },
							{ "gridDimY", uint },
							{ "gridDimZ", uint },
							{ "blockDimX", uint },
							{ "blockDimY", uint },
							{ "blockDimZ", uint },
							{ "sharedMemBytes", uint },
							{"hStream" , terra.types.pointer(opaque) } }
							--]]
			--C.printf("gridDimX: %d, gridDimY %d, blickDimX %d, blockdimY %d\n", launch.gridDimX, launch.gridDimY, launch.blockDimX, launch.blockDimY)
			cuda.computeGradient(&launch, @pd, [vars.imagesAll])
			C.cudaDeviceSynchronize()
			
			--
			-- move along the gradient by learningRate
			--
			@pd.scratchD = 0.0
			cuda.updatePosition(&launch, @pd, learningRate, pd.scratchD, [vars.imagesAll])
			C.cudaDeviceSynchronize()
			log("maxDelta %f\n", @pd.scratchD)
			var maxDelta = @pd.scratchD

			--
			-- update the learningRate
			--
			var endCost = totalCost(pd, vars.imagesAll)
			if endCost < startCost then
				learningRate = learningRate * learningGain

				if maxDelta < tolerance then
					break
				end
			else
				learningRate = learningRate * learningLoss

				if learningRate < minLearningRate then
					break
				end
			end
		end
	end

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dims[0] = 1
		for i = 0,[#vars.dims] do
			pd.dims[i+1] = actualDims[i]
		end

		pd.gradW = pd.dims[vars.gradWIndex]
		pd.gradH = pd.dims[vars.gradHIndex]

		pd.gradStore:initGPU(pd.gradW, pd.gradH)
		C.cudaMallocManaged([&&opaque](&(pd.scratchF)), sizeof(float), C.cudaMemAttachGlobal)
		C.cudaMallocManaged([&&opaque](&(pd.scratchD)), sizeof(double), C.cudaMemAttachGlobal)

		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

solversGPU.linearizedConjugateGradientGPU = function(Problem, tbl, vars)
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		dims : int64[#vars.dims + 1]
		
		b : vars.unknownType
		r : vars.unknownType
		p : vars.unknownType
		Ap : vars.unknownType
		zeroes : vars.unknownType
		
		scratchF : &float
	}
	
	local cuda = {}
	
	terra cuda.initialize(pd : PlanData, [vars.imagesAll])
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		if w < pd.gradW and h < pd.gradH then
			pd.r(w, h) = -tbl.gradient(w, h, vars.unknownImage, vars.dataImages)
			pd.b(w, h) = tbl.gradient(w, h, pd.zeroes, vars.dataImages)
			pd.p(w, h) = pd.r(w, h)
		end
	end
	
	terra cuda.computeAp(pd : PlanData, [vars.imagesAll])
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		if w < pd.gradW and h < pd.gradH then
			pd.Ap(w, h) = tbl.gradient(w, h, pd.p, vars.dataImages) - pd.b(w, h)
		end
	end

	terra cuda.updateResidualAndPosition(pd : PlanData, alpha : float, [vars.imagesAll])
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		if w < pd.gradW and h < pd.gradH then
			vars.unknownImage(w, h) = vars.unknownImage(w, h) + alpha * pd.p(w, h)
			pd.r(w, h) = pd.r(w, h) - alpha * pd.Ap(w, h)
		end
	end
	
	terra cuda.updateP(pd : PlanData, beta : float)
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		if w < pd.gradW and h < pd.gradH then
			pd.p(w, h) = pd.r(w, h) + beta * pd.p(w, h)
		end
	end
	
	terra cuda.costSum(pd : PlanData, sum : &float, [vars.imagesAll])
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		if w < pd.gradW and h < pd.gradH then
			var v = [float](tbl.cost.fn(w, h, vars.imagesAll))
			terralib.asm(terralib.types.unit,"red.global.add.f32 [$0],$1;","l,f", true, sum, v)
		end
	end
	
	terra cuda.imageInnerProduct(pd : PlanData, a : vars.unknownType, b : vars.unknownType, sum : &float)
		var w = blockDim.x * blockIdx.x + threadIdx.x;
		var h = blockDim.y * blockIdx.y + threadIdx.y;

		if w < pd.gradW and h < pd.gradH then
			var v = [float](a(w, h) * b(w, h))
			terralib.asm(terralib.types.unit,"red.global.add.f32 [$0],$1;","l,f", true, sum, v)
		end
	end
	
	cuda = terralib.cudacompile(cuda, false)
	
	local terra totalCost(pd : &PlanData, [vars.imagesAll])
		var launch = terralib.CUDAParams { (pd.gradW - 1) / 32 + 1, (pd.gradH - 1) / 32 + 1,1, 32,32,1, 0, nil }

		C.cudaDeviceSynchronize()
		@pd.scratchF = 0.0f
		C.cudaDeviceSynchronize()
		cuda.costSum(&launch, @pd, pd.scratchF, [vars.imagesAll])
		C.cudaDeviceSynchronize()
		
		return @pd.scratchF
	end
	
	-- TODO: ask zach how to do templates so this can live at a higher scope
	local terra imageInnerProduct(pd : &PlanData, a : vars.unknownType, b : vars.unknownType)
		var launch = terralib.CUDAParams { (pd.gradW - 1) / 32 + 1, (pd.gradH - 1) / 32 + 1,1, 32,32,1, 0, nil }
		
		C.cudaDeviceSynchronize()
		@pd.scratchF = 0.0f
		C.cudaDeviceSynchronize()
		cuda.imageInnerProduct(&launch, @pd, a, b, pd.scratchF)
		C.cudaDeviceSynchronize()
		
		return @pd.scratchF
	end

	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)
		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dims = pd.dims

		var [vars.imagesAll] = [util.getImages(vars, imageBindings, dims)]

		var launch = terralib.CUDAParams { (pd.gradW - 1) / 32 + 1, (pd.gradH - 1) / 32 + 1,1, 32,32,1, 0, nil }
		
		-- TODO: parameterize these
		var maxIters = 1000
		var tolerance = 1e-5

		C.cudaDeviceSynchronize()
		cuda.initialize(&launch, @pd, [vars.imagesAll])
		C.cudaDeviceSynchronize()
		
		var rTr = imageInnerProduct(pd, pd.r, pd.r)

		for iter = 0, maxIters do
	
			var iterStartCost = totalCost(pd, vars.imagesAll)
			
			C.cudaDeviceSynchronize()
			cuda.computeAp(&launch, @pd, [vars.imagesAll])
			
			var den = imageInnerProduct(pd, pd.p, pd.Ap)
			var alpha = rTr / den
			
			--log("den=%f, alpha=%f\n", den, alpha)
			
			cuda.updateResidualAndPosition(&launch, @pd, alpha, [vars.imagesAll])
			
			var rTrNew = imageInnerProduct(pd, pd.r, pd.r)
			
			log("iteration %d, cost=%f, rTr=%f\n", iter, iterStartCost, rTrNew)
			
			if(rTrNew < tolerance) then break end
			
			var beta = rTrNew / rTr
			cuda.updateP(&launch, @pd, beta)
			
			rTr = rTrNew
		end
		
		var finalCost = totalCost(pd, vars.imagesAll)
		log("final cost=%f\n", finalCost)
	end

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dims[0] = 1
		for i = 0,[#vars.dims] do
			pd.dims[i+1] = actualDims[i]
		end

		pd.gradW = pd.dims[vars.gradWIndex]
		pd.gradH = pd.dims[vars.gradHIndex]

		pd.b:initGPU(pd.gradW, pd.gradH)
		pd.r:initGPU(pd.gradW, pd.gradH)
		pd.p:initGPU(pd.gradW, pd.gradH)
		pd.Ap:initGPU(pd.gradW, pd.gradH)
		pd.zeroes:initGPU(pd.gradW, pd.gradH)
		
		C.cudaMallocManaged([&&opaque](&(pd.scratchF)), sizeof(float), C.cudaMemAttachGlobal)
		
		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

return solversGPU