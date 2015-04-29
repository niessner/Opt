local W,H = opt.Dim("W",0), opt.Dim("H",1)
local X = ad.Image("X",W,H,0)
local A = ad.Image("A",W,H,1)
local B = ad.Image("B",W,H,2)
local P = opt.InBounds(1,1)

local laplacianCost = (4*X(0,0) - (X(-1,0) + X(0,1) + X(1,0) + X(0,-1)))
local laplacianCostF = ad.select(P,laplacianCost,0)
local reconstructionCost = (3.0*B(0,0)+0.14)*(X(0,0) - A(0,0))

local cost = ad.sumsquared(laplacianCostF,reconstructionCost)

return ad.Cost(cost)