local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()
local X = S:Image("X",W,H,0)
local A = S:Image("A",W,H,1)

--local zero = S:Param("zero",float,0)

local w_fit = 0.1 -- keep us rational for now TODO PARAMETER
local w_reg = 1.0
local laplacianCost0 = X(0,0) - X(1,0)
local laplacianCost1 = X(0,0) - X(-1,0)
local laplacianCost2 = X(0,0) - X(0,1)
local laplacianCost3 = X(0,0) - X(0,-1)

local laplacianCost0F = ad.select(opt.InBounds(1,0,0,0),laplacianCost0,0)
local laplacianCost1F = ad.select(opt.InBounds(-1,0,0,0),laplacianCost1,0)
local laplacianCost2F = ad.select(opt.InBounds(0,1,0,0),laplacianCost2,0)
local laplacianCost3F = ad.select(opt.InBounds(0,-1,0,0),laplacianCost3,0)

local fittingCost = X(0,0) - A(0,0)



local cost = ad.sumsquared(
	math.sqrt(w_reg)*laplacianCost0F,
	math.sqrt(w_reg)*laplacianCost1F,
	math.sqrt(w_reg)*laplacianCost2F,
	math.sqrt(w_reg)*laplacianCost3F,
	math.sqrt(w_fit)*fittingCost)


	
return S:Cost(cost)
