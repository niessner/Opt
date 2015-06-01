local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()
local X = S:Image("X",W,H,0)
local A = S:Image("A",W,H,1)
S:UsePreconditioner(false)

local w_fit = S:Param("w_fit", float, 0)
local w_reg = S:Param("w_reg", float, 1)

local laplacianCost0 = X(0,0) - X(1,0)
local laplacianCost1 = X(0,0) - X(-1,0)
local laplacianCost2 = X(0,0) - X(0,1)
local laplacianCost3 = X(0,0) - X(0,-1)

local laplacianCost0F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(1,0,0,0),laplacianCost0,0),0)
local laplacianCost1F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(-1,0,0,0),laplacianCost1,0),0)
local laplacianCost2F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(0,1,0,0),laplacianCost2,0),0)
local laplacianCost3F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(0,-1,0,0),laplacianCost3,0),0)

local fittingCost = X(0,0) - A(0,0)

local cost = ad.sumsquared(
	ad.sqrt(w_reg)*laplacianCost0F,
	ad.sqrt(w_reg)*laplacianCost1F,
	ad.sqrt(w_reg)*laplacianCost2F,
	ad.sqrt(w_reg)*laplacianCost3F,
	ad.sqrt(w_fit)*fittingCost)
	
return S:Cost(cost)
