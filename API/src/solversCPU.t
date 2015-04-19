
local S = require("std")
local util = require("util")
local C = util.C

solversCPU = {}

solversCPU.gradientDescentCPU = function(Problem, tbl, vars)
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		gradient : vars.unknownType
		dimensions : int64[#vars.dimensions + 1]
		images : vars.planImagesType
	}

	local cpu = util.makeCPUFunctions(tbl, vars, PlanData)

	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)

		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dimensions = pd.dimensions

		unpackstruct(pd.images) = [util.getImages(vars, PlanData, imageBindings, dimensions)]

		-- TODO: parameterize these
		var initialLearningRate = 0.01
		var maxIters = 10
		var tolerance = 1e-10

		-- Fixed constants (these do not need to be parameterized)
		var learningLoss = 0.8
		var learningGain = 1.1
		var minLearningRate = 1e-25

		var learningRate = initialLearningRate

		for iter = 0, maxIters do
			var startCost = cpu.computeCost(pd)
			logSolver("iteration %d, cost=%f, learningRate=%f\n", iter, startCost, learningRate)
			--C.getchar()

			cpu.computeGradient(pd, pd.gradient, pd.images.unknown)
			
			--
			-- move along the gradient by learningRate
			--
			var maxDelta = 0.0
			for h = 0,pd.gradH do
				for w = 0,pd.gradW do
					var delta = -learningRate * pd.gradient(w, h)
					pd.images.unknown(w, h) = pd.images.unknown(w, h) + delta
					maxDelta = util.max(C.fabsf(delta), maxDelta)
				end
			end

			--
			-- update the learningRate
			--
			var endCost = cpu.computeCost(pd)
			if endCost < startCost then
				learningRate = learningRate * learningGain

				if maxDelta < tolerance then
					logSolver("terminating, maxDelta=%f\n", maxDelta)
					break
				end
			else
				learningRate = learningRate * learningLoss

				if learningRate < minLearningRate then
					logSolver("terminating, learningRate=%f\n", learningRate)
					break
				end
			end
		end
	end

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dimensions[0] = 1
		for i = 0,[#vars.dimensions] do
			pd.dimensions[i+1] = actualDims[i]
		end

		pd.gradW = pd.dimensions[vars.gradWIndex]
		pd.gradH = pd.dimensions[vars.gradHIndex]

		pd.gradient:initCPU(pd.gradW, pd.gradH)

		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

solversCPU.conjugateGradientCPU = function(Problem, tbl, vars)
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		costW : int
		costH : int
		dimensions : int64[#vars.dimensions + 1]
		images : vars.planImagesType
				
		currentValues : vars.unknownType
		currentResiduals : vars.unknownType
		
		gradient : vars.unknownType
		prevGradient : vars.unknownType
		
		searchDirection : vars.unknownType
	}
	
	local cpu = util.makeCPUFunctions(tbl, vars, PlanData)
	
	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)

		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dimensions = pd.dimensions

		unpackstruct(pd.images) = [util.getImages(vars, PlanData, imageBindings, dimensions)]
		
		var maxIters = 1000
		
		var prevBestAlpha = 0.0

		for iter = 0, maxIters do

			var iterStartCost = cpu.computeCost(pd)
			logSolver("iteration %d, cost=%f\n", iter, iterStartCost)

			cpu.computeGradient(pd, pd.gradient, pd.images.unknown)
			
			--
			-- compute the search direction
			--
			var beta = 0.0
			if iter == 0 then
				for h = 0, pd.gradH do
					for w = 0, pd.gradW do
						pd.searchDirection(w, h) = -pd.gradient(w, h)
					end
				end
			else
				var num = 0.0
				var den = 0.0
				
				--
				-- Polak-Ribiere conjugacy
				-- 
				for h = 0, pd.gradH do
					for w = 0, pd.gradW do
						var g = pd.gradient(w, h)
						var p = pd.prevGradient(w, h)
						num = num + (-g * (-g + p))
						den = den + p * p
					end
				end
				beta = util.max(num / den, 0.0)
				
				var epsilon = 1e-5
				if den > -epsilon and den < epsilon then
					beta = 0.0
				end
				
				for h = 0, pd.gradH do
					for w = 0, pd.gradW do
						pd.searchDirection(w, h) = -pd.gradient(w, h) + beta * pd.searchDirection(w, h)
					end
				end
			end
			
			cpu.copyImage(pd.prevGradient, pd.gradient)
			
			--
			-- line search
			--
			cpu.copyImage(pd.currentValues, pd.images.unknown)
			cpu.computeResiduals(pd, pd.currentValues, pd.currentResiduals)
			
			var bestAlpha = cpu.lineSearchQuadraticFallback(pd, pd.currentValues, pd.currentResiduals, pd.searchDirection, pd.images.unknown, prevBestAlpha)
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.images.unknown(w, h) = pd.currentValues(w, h) + bestAlpha * pd.searchDirection(w, h)
				end
			end
			
			prevBestAlpha = bestAlpha
			
			logSolver("alpha=%12.12f, beta=%12.12f\n\n", bestAlpha, beta)
			if bestAlpha == 0.0 and beta == 0.0 then
				break
			end
		end
	end

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dimensions[0] = 1
		for i = 0,[#vars.dimensions] do
			pd.dimensions[i+1] = actualDims[i]
		end

		pd.gradW = pd.dimensions[vars.gradWIndex]
		pd.gradH = pd.dimensions[vars.gradHIndex]
		
		pd.costW = pd.dimensions[vars.costWIndex]
		pd.costH = pd.dimensions[vars.costHIndex]

		pd.currentValues:initCPU(pd.gradW, pd.gradH)
		
		pd.currentResiduals:initCPU(pd.costW, pd.costH)
		
		pd.gradient:initCPU(pd.gradW, pd.gradH)
		pd.prevGradient:initCPU(pd.gradW, pd.gradH)
		
		pd.searchDirection:initCPU(pd.gradW, pd.gradH)

		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

solversCPU.linearizedConjugateGradientCPU = function(Problem, tbl, vars)
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		dimensions : int64[#vars.dimensions + 1]
		images : vars.planImagesType
		
		b : vars.unknownType
		r : vars.unknownType
		p : vars.unknownType
		zeroes : vars.unknownType
		Ap : vars.unknownType
	}
	
	local cpu = util.makeCPUFunctions(tbl, vars, PlanData)

	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)
		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dimensions = pd.dimensions

		unpackstruct(pd.images) = [util.getImages(vars, PlanData, imageBindings, dimensions)]
		
		-- TODO: parameterize these
		var maxIters = 1000
		var tolerance = 1e-5

		cpu.computeGradient(pd, pd.r, pd.images.unknown)
		cpu.scaleImage(pd.r, -1.0f)
		
		cpu.copyImage(pd.images.unknown, pd.zeroes)
		cpu.computeGradient(pd, pd.b, pd.images.unknown)
		
		cpu.copyImage(pd.p, pd.r)
		
		--for h = 0, pd.gradH do
		--	for w = 0, pd.gradW do
		--		pd.r(w, h) = -tbl.gradient.boundary(w, h, pd.images.unknown, vars.dataImages)
		--		pd.b(w, h) = tbl.gradient.boundary(w, h, pd.zeroes, vars.dataImages)
		--		pd.p(w, h) = pd.r(w, h)
		--	end
		--end
		
		var rTr = cpu.imageInnerProduct(pd.r, pd.r)

		for iter = 0,maxIters do

			var iterStartCost = cpu.computeCost(pd)
			
			cpu.computeGradient(pd, pd.Ap, pd.p)
			cpu.addImage(pd.Ap, pd.b, -1.0f)
			
			--[[for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.Ap(w, h) = tbl.gradient.boundary(w, h, pd.p, vars.dataImages) - pd.b(w, h)
				end
			end]]
			
			var den = cpu.imageInnerProduct(pd.p, pd.Ap)
			var alpha = rTr / den
			
			cpu.addImage(pd.images.unknown, pd.p, alpha)
			cpu.addImage(pd.r, pd.Ap, -alpha)
			
			--for h = 0, pd.gradH do
			--	for w = 0, pd.gradW do
			--		pd.images.unknown(w, h) = pd.images.unknown(w, h) + alpha * pd.p(w, h)
			--		pd.r(w, h) = pd.r(w, h) - alpha * pd.Ap(w, h)
			--	end
			--end
			
			var rTrNew = cpu.imageInnerProduct(pd.r, pd.r)
			
			logSolver("iteration %d, cost=%f, rTr=%f\n", iter, iterStartCost, rTrNew)
			
			if(rTrNew < tolerance) then break end
			
			var beta = rTrNew / rTr
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.p(w, h) = pd.r(w, h) + beta * pd.p(w, h)
				end
			end
			
			rTr = rTrNew
		end
		
		var finalCost = cpu.computeCost(pd)
		logSolver("final cost=%f\n", finalCost)
	end

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dimensions[0] = 1
		for i = 0,[#vars.dimensions] do
			pd.dimensions[i+1] = actualDims[i]
		end

		pd.gradW = pd.dimensions[vars.gradWIndex]
		pd.gradH = pd.dimensions[vars.gradHIndex]

		pd.b:initCPU(pd.gradW, pd.gradH)
		pd.r:initCPU(pd.gradW, pd.gradH)
		pd.p:initCPU(pd.gradW, pd.gradH)
		pd.Ap:initCPU(pd.gradW, pd.gradH)
		pd.zeroes:initCPU(pd.gradW, pd.gradH)
		
		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

solversCPU.lbfgsCPU = function(Problem, tbl, vars)

	local maxIters = 1000
	
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		costW : int
		costH : int
		dimensions : int64[#vars.dimensions + 1]
		images : vars.planImagesType
		
		gradient : vars.unknownType
		prevGradient : vars.unknownType
				
		p : vars.unknownType
		sList : vars.unknownType[maxIters]
		yList : vars.unknownType[maxIters]
		syProduct : float[maxIters]
		yyProduct : float[maxIters]
		alphaList : float[maxIters]
		
		-- variables used for line search
		currentValues : vars.unknownType
		currentResiduals : vars.unknownType
	}
	
	local cpu = util.makeCPUFunctions(tbl, vars, PlanData)
	
	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)

		-- two-loop recursion: http://papers.nips.cc/paper/5333-large-scale-l-bfgs-using-mapreduce.pdf
		
		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dimensions = pd.dimensions

		unpackstruct(pd.images) = [util.getImages(vars, PlanData, imageBindings, dimensions)]

		var m = 2
		var k = 0
		
		var prevBestAlpha = 0.0
		
		cpu.computeGradient(pd, pd.gradient, pd.images.unknown)

		for iter = 0, maxIters - 1 do

			var iterStartCost = cpu.computeCost(pd)
			logSolver("iteration %d, cost=%f\n", iter, iterStartCost)
			
			--
			-- compute the search direction p
			--
			cpu.setImage(pd.p, pd.gradient, -1.0f)
			
			if k >= 1 then
				for i = k - 1, k - m - 1, -1 do
					if i < 0 then break end
					pd.alphaList[i] = cpu.imageInnerProduct(pd.sList[i], pd.p) / pd.syProduct[i]
					cpu.addImage(pd.p, pd.yList[i], -pd.alphaList[i])
				end
				var scale = pd.syProduct[k - 1] / pd.yyProduct[k - 1]
				cpu.scaleImage(pd.p, scale)
				for i = k - m, k do
					if i >= 0 then
						var beta = cpu.imageInnerProduct(pd.yList[i], pd.p) / pd.syProduct[i]
						cpu.addImage(pd.p, pd.sList[i], pd.alphaList[i] - beta)
					end
				end
			end
			
			--
			-- line search
			--
			cpu.copyImage(pd.currentValues, pd.images.unknown)
			cpu.computeResiduals(pd, pd.currentValues, pd.currentResiduals)
			
			var bestAlpha = cpu.lineSearchQuadraticFallback(pd, pd.currentValues, pd.currentResiduals, pd.p, pd.images.unknown, prevBestAlpha)
			
			-- compute new x and s
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					var delta = bestAlpha * pd.p(w, h)
					pd.images.unknown(w, h) = pd.currentValues(w, h) + delta
					pd.sList[k](w, h) = delta
				end
			end
			
			cpu.copyImage(pd.prevGradient, pd.gradient)
			
			cpu.computeGradient(pd, pd.gradient, pd.images.unknown)
			
			-- compute new y
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.yList[k](w, h) = pd.gradient(w, h) - pd.prevGradient(w, h)
				end
			end
			
			pd.syProduct[k] = cpu.imageInnerProduct(pd.sList[k], pd.yList[k])
			pd.yyProduct[k] = cpu.imageInnerProduct(pd.yList[k], pd.yList[k])
			
			prevBestAlpha = bestAlpha
			
			k = k + 1
			
			logSolver("alpha=%12.12f\n\n", bestAlpha)
			if bestAlpha == 0.0 then
				break
			end
		end
	end
	
	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dimensions[0] = 1
		for i = 0,[#vars.dimensions] do
			pd.dimensions[i+1] = actualDims[i]
		end

		pd.gradW = pd.dimensions[vars.gradWIndex]
		pd.gradH = pd.dimensions[vars.gradHIndex]
		
		pd.costW = pd.dimensions[vars.costWIndex]
		pd.costH = pd.dimensions[vars.costHIndex]

		pd.gradient:initCPU(pd.gradW, pd.gradH)
		pd.prevGradient:initCPU(pd.gradW, pd.gradH)
		
		pd.currentValues:initCPU(pd.gradW, pd.gradH)
		pd.currentResiduals:initCPU(pd.costW, pd.costH)
		
		pd.p:initCPU(pd.gradW, pd.gradH)
		
		for i = 0, maxIters - 1 do
			pd.sList[i]:initCPU(pd.gradW, pd.gradH)
			pd.yList[i]:initCPU(pd.gradW, pd.gradH)
		end

		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

-- vector-free L-BFGS using two-loop recursion: http://papers.nips.cc/paper/5333-large-scale-l-bfgs-using-mapreduce.pdf
solversCPU.vlbfgsCPU = function(Problem, tbl, vars)

	local maxIters = 1000
	local m = 2
	local b = 2 * m + 1
	
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		costW : int
		costH : int
		dimensions : int64[#vars.dimensions + 1]
		images : vars.planImagesType
		
		gradient : vars.unknownType
		prevGradient : vars.unknownType

		p : vars.unknownType
		sList : vars.unknownType[m]
		yList : vars.unknownType[m]
		alphaList : float[maxIters]
		
		dotProductMatrix : vars.unknownType
		dotProductMatrixStorage : vars.unknownType
		coefficients : double[b]
		
		-- variables used for line search
		currentValues : vars.unknownType
		currentResiduals : vars.unknownType
	}
	
	local terra imageFromIndex(pd : &PlanData, index : int)
		if index < m then
			return pd.sList[index]
		elseif index < 2 * m then
			return pd.yList[index - m]
		else
			return pd.gradient
		end
	end
	
	local terra nextCoefficientIndex(index : int)
		if index == m - 1 or index == 2 * m - 1 or index == 2 * m then
			return -1
		end
		return index + 1
	end
	
	local cpu = util.makeCPUFunctions(tbl, vars, PlanData)
	
	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)
		
		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dimensions = pd.dimensions

		unpackstruct(pd.images) = [util.getImages(vars, PlanData, imageBindings, dimensions)]

		var k = 0
		
		-- using an initial guess of alpha means that it will invoke quadratic optimization on the first iteration,
		-- which is only sometimes a good idea.
		var prevBestAlpha = 1.0
		
		cpu.computeGradient(pd, pd.gradient, pd.images.unknown)

		for iter = 0, maxIters - 1 do

			var iterStartCost = cpu.computeCost(pd)
			logSolver("iteration %d, cost=%f\n", iter, iterStartCost)
			
			--
			-- compute the search direction p
			--
			if k == 0 then
				cpu.setImage(pd.p, pd.gradient, -1.0f)
			else
				-- compute the top half of the dot product matrix
				cpu.copyImage(pd.dotProductMatrixStorage, pd.dotProductMatrix)
				for i = 0, b do
					for j = i, b do
						var prevI = nextCoefficientIndex(i)
						var prevJ = nextCoefficientIndex(j)
						if prevI == -1 or prevJ == -1 then
							pd.dotProductMatrix(i, j) = cpu.imageInnerProduct(imageFromIndex(pd, i), imageFromIndex(pd, j))
						else
							pd.dotProductMatrix(i, j) = pd.dotProductMatrixStorage(prevI, prevJ)
						end
					end
				end
				
				-- compute the bottom half of the dot product matrix
				for i = 0, b do
					for j = 0, i do
						pd.dotProductMatrix(i, j) = pd.dotProductMatrix(j, i)
					end
				end
			
				for i = 0, 2 * m do pd.coefficients[i] = 0.0 end
				pd.coefficients[2 * m] = -1.0
				
				for i = k - 1, k - m - 1, -1 do
					if i < 0 then break end
					var j = i - (k - m)
					
					var num = 0.0
					for q = 0, b do
						num = num + pd.coefficients[q] * pd.dotProductMatrix(q, j)
					end
					var den = pd.dotProductMatrix(j, j + m)
					pd.alphaList[i] = num / den
					pd.coefficients[j + m] = pd.coefficients[j + m] - pd.alphaList[i]
				end
				
				var scale = pd.dotProductMatrix(m - 1, 2 * m - 1) / pd.dotProductMatrix(2 * m - 1, 2 * m - 1)
				for i = 0, b do
					pd.coefficients[i] = pd.coefficients[i] * scale
				end
				
				for i = k - m, k do
					if i >= 0 then
						var j = i - (k - m)
						var num = 0.0
						for q = 0, b do
							num = num + pd.coefficients[q] * pd.dotProductMatrix(q, m + j)
						end
						var den = pd.dotProductMatrix(j, j + m)
						var beta = num / den
						pd.coefficients[j] = pd.coefficients[j] + (pd.alphaList[i] - beta)
					end
				end
				
				--
				-- reconstruct p from basis vectors
				--
				cpu.scaleImage(pd.p, 0.0f)
				for i = 0, b do
					var image = imageFromIndex(pd, i)
					var coefficient = pd.coefficients[i]
					cpu.addImage(pd.p, image, coefficient)
				end
			end
			
			--
			-- line search
			--
			cpu.copyImage(pd.currentValues, pd.images.unknown)
			cpu.computeResiduals(pd, pd.currentValues, pd.currentResiduals)
			
			var bestAlpha = cpu.lineSearchQuadraticFallback(pd, pd.currentValues, pd.currentResiduals, pd.p, pd.images.unknown, prevBestAlpha)
			
			-- cycle the oldest s and y
			var yListStore = pd.yList[0]
			var sListStore = pd.sList[0]
			for i = 0, m - 1 do
				pd.yList[i] = pd.yList[i + 1]
				pd.sList[i] = pd.sList[i + 1]
			end
			pd.yList[m - 1] = yListStore
			pd.sList[m - 1] = sListStore
			
			-- compute new x and s
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					var delta = bestAlpha * pd.p(w, h)
					pd.images.unknown(w, h) = pd.currentValues(w, h) + delta
					pd.sList[m - 1](w, h) = delta
				end
			end
			
			cpu.copyImage(pd.prevGradient, pd.gradient)
			
			cpu.computeGradient(pd, pd.gradient, pd.images.unknown)
			
			-- compute new y
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.yList[m - 1](w, h) = pd.gradient(w, h) - pd.prevGradient(w, h)
				end
			end
			
			prevBestAlpha = bestAlpha
			
			k = k + 1
			
			logSolver("alpha=%12.12f\n\n", bestAlpha)
			if bestAlpha == 0.0 then
				break
			end
		end
	end
	
	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dimensions[0] = 1
		for i = 0,[#vars.dimensions] do
			pd.dimensions[i + 1] = actualDims[i]
		end

		pd.gradW = pd.dimensions[vars.gradWIndex]
		pd.gradH = pd.dimensions[vars.gradHIndex]
		
		pd.costW = pd.dimensions[vars.costWIndex]
		pd.costH = pd.dimensions[vars.costHIndex]

		pd.gradient:initCPU(pd.gradW, pd.gradH)
		pd.prevGradient:initCPU(pd.gradW, pd.gradH)
		
		pd.currentValues:initCPU(pd.gradW, pd.gradH)
		pd.currentResiduals:initCPU(pd.costW, pd.costH)
		
		pd.p:initCPU(pd.gradW, pd.gradH)
		
		for i = 0, m do
			pd.sList[i]:initCPU(pd.gradW, pd.gradH)
			pd.yList[i]:initCPU(pd.gradW, pd.gradH)
		end
		
		pd.dotProductMatrix:initCPU(b, b)
		pd.dotProductMatrixStorage:initCPU(b, b)

		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

return solversCPU