local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()
local X = S:Image("X", opt.float4,W,H,0)
local A = S:Image("A", opt.float4,W,H,1)
local G = S:Adjacency("G", {W,H}, {W,H}, 0)

local w_fitSqrt = S:Param("w_fitSqrt", float, 0)
local w_regSqrt = S:Param("w_regSqrt", float, 1)

local terms = terralib.newlist()

for i = 0,3 do

	for adj in G:neighbors(0,0) do
		local laplacianCost = X(0,0,i) - X(adj.x, adj.y, i)
		terms:insert(w_regSqrt*laplacianCost)
	end

	local fittingCost = X(0,0,i) - A(0,0,i)
	terms:insert(w_fitSqrt*fittingCost)
end

local cost = ad.sumsquared(unpack(terms))
return S:Cost(cost)
