
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
return util
