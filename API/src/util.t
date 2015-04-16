
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

util.makeComputeCost = function(tbl, images)
	local terra computeCost([images])
		var result = 0.0
		for h = 0, [ images[1] ].H do
			for w = 0, [ images[1] ].W do
				var v = tbl.cost.fn(w, h, images)
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
				gradientOut(w, h) = tbl.gradient(w, h, images)
			end
		end
	end
	return computeGradient
end

util.makeDeltaCost = function(tbl, imageType, dataImages)
	local terra deltaCost(baseResiduals : imageType, currentValues : imageType, [dataImages])
		var result = 0.0
		for h = 0, currentValues.H do
			for w = 0, currentValues.W do
				var residual = tbl.cost.fn(w, h, currentValues, dataImages)
				var delta = residual - baseResiduals(w, h)
				result = result + delta
			end
		end
		return result
	end
	return deltaCost
end

util.makeSearchCost = function(tbl, imageType, dataImages)
	local deltaCost = util.makeDeltaCost(tbl, imageType, dataImages)
	local terra searchCost(baseValues : imageType, baseResiduals : imageType, searchDirection : imageType, alpha : float, valueStore : imageType, [dataImages])
		for h = 0, baseValues.H do
			for w = 0, baseValues.W do
				valueStore(w, h) = baseValues(w, h) + alpha * searchDirection(w, h)
			end
		end
		return deltaCost(baseResiduals, valueStore, dataImages)
	end
	return searchCost
end

util.makeComputeResiduals = function(tbl, imageType, dataImages)
	local terra computeResiduals(values : imageType, residuals : imageType, [dataImages])
		for h = 0, values.H do
			for w = 0, values.W do
				residuals(w, h) = tbl.cost.fn(w, h, values, dataImages)
			end
		end
	end
	return computeResiduals
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

util.makeLineSearchBruteForce = function(tbl, imageType, dataImages)

	local computeSearchCost = util.makeSearchCost(tbl, imageType, dataImages)
	local computeResiduals = util.makeComputeResiduals(tbl, imageType, dataImages)

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
			
			var searchCost = computeSearchCost(baseValues, baseResiduals, searchDirection, alpha, valueStore, dataImages)
			
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

util.makeLineSearchQuadraticMinimum = function(tbl, imageType, dataImages)

	local computeSearchCost = util.makeSearchCost(tbl, imageType, dataImages)
	local computeResiduals = util.makeComputeResiduals(tbl, imageType, dataImages)

	local terra lineSearchQuadraticMinimum(baseValues : imageType, baseResiduals : imageType, searchDirection : imageType, valueStore : imageType, alphaGuess : float, [dataImages])

		var alphas = array(alphaGuess * 0.25, alphaGuess * 0.5, alphaGuess * 0.75, 0.0)
		var costs : float[4]
		var bestCost = 0.0
		var bestAlpha = 0.0
		
		for alphaIndex = 0, 4 do
			var alpha = 0.0
			if alphaIndex <= 2 then alpha = alphas[alphaIndex]
			else
				var a1 = alphas[0] var a2 = alphas[1] var a3 = alphas[2]
				var c1 = costs[0] var c2 = costs[1] var c3 = costs[2]
				var a = ((c2-c1)*(a1-a3) + (c3-c1)*(a2-a1))/((a1-a3)*(a2*a2-a1*a1) + (a2-a1)*(a3*a3-a1*a1))
				var b = ((c2 - c1) - a * (a2*a2 - a1*a1)) / (a2 - a1)
				var c = c1 - a * a1 * a1 - b * a1
				-- 2ax + b = 0, x = -b / 2a
				alpha = -b / (2.0 * a)
			end
			
			var searchCost = computeSearchCost(baseValues, baseResiduals, searchDirection, alpha, valueStore, dataImages)
			
			if searchCost < bestCost then
				bestAlpha = alpha
				bestCost = searchCost
			elseif alphaIndex == 3 then
				C.printf("quadratic minimization failed\n")
			end
			
			costs[alphaIndex] = searchCost
		end
		
		return bestAlpha
	end
	return lineSearchQuadraticMinimum
end


				
				
util.makeDumpLineSearchValues = function(tbl, imageType, dataImages)

	local computeSearchCost = util.makeSearchCost(tbl, imageType, dataImages)
	local computeResiduals = util.makeComputeResiduals(tbl, imageType, dataImages)

	local terra dumpLineSearchValues(baseValues : imageType, baseResiduals : imageType, searchDirection : imageType, valueStore : imageType, [dataImages])

		-- Constants
		var lineSearchMaxIters = 1000
		var lineSearchBruteForceStart = 1e-5
		var lineSearchBruteForceMultiplier = 1.1
				
		var alpha = lineSearchBruteForceStart
		
		var file = C.fopen("C:/code/debug.txt", "wb")

		for lineSearchIndex = 0, lineSearchMaxIters do
			alpha = alpha * lineSearchBruteForceMultiplier
			
			var searchCost = computeSearchCost(baseValues, baseResiduals, searchDirection, alpha, valueStore, dataImages)
			
			C.fprintf(file, "%15.15f\t%15.15f\n", alpha * 1000.0, searchCost)
			
			if searchCost >= 100.0 then break end
		end
		
		C.fclose(file)
		log("debug alpha outputted")
		C.getchar()
	end
	return dumpLineSearchValues
end

util.makeCPUFunctions = function(tbl, imageType, dataImages, allImages)
	local cpu = {}
	cpu.computeCost = util.makeComputeCost(tbl, allImages)
	cpu.computeGradient = util.makeComputeGradient(tbl, imageType, allImages)
	cpu.computeSearchCost = util.makeSearchCost(tbl, imageType, dataImages)
	cpu.computeResiduals = util.makeComputeResiduals(tbl, imageType, dataImages)
	cpu.lineSearchBruteForce = util.makeLineSearchBruteForce(tbl, imageType, dataImages)
	cpu.lineSearchQuadraticMinimum = util.makeLineSearchQuadraticMinimum(tbl, imageType, dataImages)
	cpu.dumpLineSearchValues = util.makeDumpLineSearchValues(tbl, imageType, dataImages)
	return cpu
end

return util
