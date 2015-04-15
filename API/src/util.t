
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

return util
