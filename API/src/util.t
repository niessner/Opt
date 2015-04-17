
local S = require("std")

local util = {}

util.C = terralib.includecstring [[
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
]]

local C = util.C

util.max = terra(x : double, y : double)
	return terralib.select(x > y, x, y)
end

util.getImages = function(vars,imageBindings,actualDims)
	local results = terralib.newlist()
	for i,argumentType in ipairs(vars.argumentTypes) do
		local Windex,Hindex = vars.dimIndex[argumentType.metamethods.W],vars.dimIndex[argumentType.metamethods.H]
		assert(Windex and Hindex)
		results:insert(`argumentType 
		 { W = actualDims[Windex], 
		   H = actualDims[Hindex], 
		   impl = 
		   @imageBindings[i - 1]}) 
	end
	return results
end

util.makeImageInnerProduct = function(imageType)
	local terra imageInnerProduct(a : imageType, b : imageType)
		var sum = 0.0
		for h = 0, a.H do
			for w = 0, a.W do
				sum = sum + a(w, h) * b(w, h)
			end
		end
		return sum
	end
	return imageInnerProduct
end

util.makeSetImage = function(imageType)
	local terra setImage(targetImage : imageType, sourceImage : imageType, scale : float)
		for h = 0, targetImage.H do
			for w = 0, targetImage.W do
				targetImage(w, h) = sourceImage(w, h) * scale
			end
		end
	end
	return setImage
end

util.makeCopyImage = function(imageType)
	local terra copyImage(targetImage : imageType, sourceImage : imageType)
		for h = 0, targetImage.H do
			for w = 0, targetImage.W do
				targetImage(w, h) = sourceImage(w, h)
			end
		end
	end
	return copyImage
end

util.makeScaleImage = function(imageType)
	local terra scaleImage(targetImage : imageType, scale : float)
		for h = 0, targetImage.H do
			for w = 0, targetImage.W do
				targetImage(w, h) = targetImage(w, h) * scale
			end
		end
	end
	return scaleImage
end

util.makeAddImage = function(imageType)
	local terra addImage(targetImage : imageType, addedImage : imageType, scale : float)
		for h = 0, targetImage.H do
			for w = 0, targetImage.W do
				targetImage(w, h) = targetImage(w, h) + addedImage(w, h) * scale
			end
		end
	end
	return addImage
end

util.makeComputeCost = function(tbl, images)
	local terra computeCost([images])
		var result = 0.0
		for h = 0, [ images[1] ].H do
			for w = 0, [ images[1] ].W do
				var v = tbl.cost.boundary(w, h, images)
				result = result + v
			end
		end
		return result
	end
	return computeCost
end

util.makeComputeGradient = function(tbl, imageType, images)
	local terra computeGradient(gradientOut : imageType, [images])
		for h = 0, gradientOut.H do
			for w = 0, gradientOut.W do
				gradientOut(w, h) = tbl.gradient.boundary(w, h, images)
			end
		end
	end
	return computeGradient
end

util.makeDeltaCost = function(tbl, imageType, dataImages)
	local terra deltaCost(baseResiduals : imageType, currentValues : imageType, [dataImages])
		var result : double = 0.0
		for h = 0, currentValues.H do
			for w = 0, currentValues.W do
				var residual = tbl.cost.boundary(w, h, currentValues, dataImages)
				var delta = residual - baseResiduals(w, h)
				result = result + [double](delta)
			end
		end
		return result
	end
	return deltaCost
end

util.makeSearchCost = function(tbl, imageType, cpu, dataImages)
	local terra searchCost(baseValues : imageType, baseResiduals : imageType, searchDirection : imageType, alpha : float, valueStore : imageType, [dataImages])
		for h = 0, baseValues.H do
			for w = 0, baseValues.W do
				valueStore(w, h) = baseValues(w, h) + alpha * searchDirection(w, h)
			end
		end
		return cpu.deltaCost(baseResiduals, valueStore, dataImages)
	end
	return searchCost
end

util.makeSearchCostParallel = function(tbl, imageType, cpu, dataImages)
	local terra searchCostParallel(baseValues : imageType, baseResiduals : imageType, searchDirection : imageType, count : int, alphas : &float, costs : &float, valueStore : imageType, [dataImages])
		for i = 0, count do
			for h = 0, baseValues.H do
				for w = 0, baseValues.W do
					valueStore(w, h) = baseValues(w, h) + alphas[i] * searchDirection(w, h)
				end
			end
			costs[i] = cpu.deltaCost(baseResiduals, valueStore, dataImages)
		end
	end
	return searchCostParallel
end

util.makeComputeResiduals = function(tbl, imageType, dataImages)
	local terra computeResiduals(values : imageType, residuals : imageType, [dataImages])
		for h = 0, values.H do
			for w = 0, values.W do
				residuals(w, h) = tbl.cost.boundary(w, h, values, dataImages)
			end
		end
	end
	return computeResiduals
end

util.makeLineSearchBruteForce = function(tbl, imageType, cpu, dataImages)
	local terra lineSearchBruteForce(baseValues : imageType, baseResiduals : imageType, searchDirection : imageType, valueStore : imageType, [dataImages])

		-- Constants
		var lineSearchMaxIters = 1000
		var lineSearchBruteForceStart = 1e-5
		var lineSearchBruteForceMultiplier = 1.1
				
		var alpha = lineSearchBruteForceStart
		var bestAlpha = 0.0
		
		var terminalCost = 10.0

		var bestCost = 0.0
		
		for lineSearchIndex = 0, lineSearchMaxIters do
			alpha = alpha * lineSearchBruteForceMultiplier
			
			var searchCost = cpu.computeSearchCost(baseValues, baseResiduals, searchDirection, alpha, valueStore, dataImages)
			
			if searchCost < bestCost then
				bestAlpha = alpha
				bestCost = searchCost
			elseif searchCost > terminalCost then
				break
			end
		end
		
		return bestAlpha
	end
	return lineSearchBruteForce
end

util.makeLineSearchQuadraticMinimum = function(tbl, imageType, cpu, dataImages)
	local terra lineSearchQuadraticMinimum(baseValues : imageType, baseResiduals : imageType, searchDirection : imageType, valueStore : imageType, alphaGuess : float, [dataImages])

		var alphas : float[4] = array(alphaGuess * 0.5f, alphaGuess * 1.0f, alphaGuess * 1.5f, 0.0f)
		var costs : float[4]
		
		cpu.computeSearchCostParallel(baseValues, baseResiduals, searchDirection, 3, alphas, costs, valueStore, dataImages)
		
		var a1 = alphas[0] var a2 = alphas[1] var a3 = alphas[2]
		var c1 = costs[0] var c2 = costs[1] var c3 = costs[2]
		var a = ((c2-c1)*(a1-a3) + (c3-c1)*(a2-a1))/((a1-a3)*(a2*a2-a1*a1) + (a2-a1)*(a3*a3-a1*a1))
		var b = ((c2 - c1) - a * (a2*a2 - a1*a1)) / (a2 - a1)
		alphas[3] = -b / (2.0 * a)
		costs[3] = cpu.computeSearchCost(baseValues, baseResiduals, searchDirection, alphas[3], valueStore, dataImages)
		
		var bestCost = 0.0
		var bestAlpha = 0.0
		for i = 0, 4 do
			if costs[i] < bestCost then
				bestAlpha = alphas[i]
				bestCost = costs[i]
			elseif i == 3 then
				C.printf("quadratic minimization failed, bestAlpha=%f\n", bestAlpha)
				--cpu.dumpLineSearch(baseValues, baseResiduals, searchDirection, valueStore, dataImages)
			end
		end
		
		return bestAlpha
	end
	return lineSearchQuadraticMinimum
end

util.makeDumpLineSearch = function(tbl, imageType, cpu, dataImages)
	local terra dumpLineSearch(baseValues : imageType, baseResiduals : imageType, searchDirection : imageType, valueStore : imageType, [dataImages])

		-- Constants
		var lineSearchMaxIters = 1000
		var lineSearchBruteForceStart = 1e-5
		var lineSearchBruteForceMultiplier = 1.1
				
		var alpha = lineSearchBruteForceStart
		
		var file = C.fopen("C:/code/debug.txt", "wb")

		for lineSearchIndex = 0, lineSearchMaxIters do
			alpha = alpha * lineSearchBruteForceMultiplier
			
			var searchCost = cpu.computeSearchCost(baseValues, baseResiduals, searchDirection, alpha, valueStore, dataImages)
			
			C.fprintf(file, "%15.15f\t%15.15f\n", alpha, searchCost)
			
			if searchCost >= 10.0 then break end
		end
		
		C.fclose(file)
		log("debug alpha outputted")
		C.getchar()
	end
	return dumpLineSearch
end

util.makeCPUFunctions = function(tbl, imageType, dataImages, allImages)
	local cpu = {}
	cpu.computeCost = util.makeComputeCost(tbl, allImages)
	cpu.computeGradient = util.makeComputeGradient(tbl, imageType, allImages)
	cpu.copyImage = util.makeCopyImage(imageType)
	cpu.setImage = util.makeSetImage(imageType)
	cpu.addImage = util.makeAddImage(imageType)
	cpu.scaleImage = util.makeScaleImage(imageType)
	cpu.deltaCost = util.makeDeltaCost(tbl, imageType, dataImages)
	cpu.computeSearchCost = util.makeSearchCost(tbl, imageType, cpu, dataImages)
	cpu.computeSearchCostParallel = util.makeSearchCostParallel(tbl, imageType, cpu, dataImages)
	cpu.computeResiduals = util.makeComputeResiduals(tbl, imageType, dataImages)
	cpu.imageInnerProduct = util.makeImageInnerProduct(imageType)
	cpu.dumpLineSearch = util.makeDumpLineSearch(tbl, imageType, cpu, dataImages)
	cpu.lineSearchBruteForce = util.makeLineSearchBruteForce(tbl, imageType, cpu, dataImages)
	cpu.lineSearchQuadraticMinimum = util.makeLineSearchQuadraticMinimum(tbl, imageType, cpu, dataImages)
	return cpu
end

return util
