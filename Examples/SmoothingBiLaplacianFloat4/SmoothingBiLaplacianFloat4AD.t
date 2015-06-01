local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()
local X = S:Image("X", opt.float4,W,H,0)
local A = S:Image("A", opt.float4,W,H,1)


local w_fitSqrt = S:Param("w_fitSqrt", float, 0)
local w_regSqrt = S:Param("w_regSqrt", float, 1)

local terms = terralib.newlist()

local laplacianCost = (4*X(0,0) - (X(-1,0) + X(0,1) + X(1,0) + X(0,-1)))
local laplacianCostF = ad.select(opt.InBounds(0,0,1,1),laplacianCost,0)


for i = 0,3 do	
	terms:insert(w_regSqrt*laplacianCostF(i))
	
	local fittingCost = X(0,0,i) - A(0,0,i)
	terms:insert(w_fitSqrt*fittingCost)
end

local cost = ad.sumsquared(unpack(terms))
return S:Cost(cost)
