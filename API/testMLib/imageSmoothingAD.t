local W,H = opt.Dim("W",0), opt.Dim("H",1)
local X = ad.Image("X",W,H,0)
local A = ad.Image("A",W,H,1)
local P = opt.InBounds(1,1)


local w = 0.1 -- keep us rational for now
local laplacianCost = (4*A(0,0)*X(0,0) - (X(-1,0) + X(0,1) + X(1,0) + X(0,-1)))
local laplacianCostF = ad.select(P,laplacianCost,0)
local reconstructionCost = X(0,0) - A(0,0)

local cost = ad.sumsquared(laplacianCostF,math.sqrt(w)*reconstructionCost)

return ad.Cost(cost)
