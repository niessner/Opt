local W,H = opt.Dim("W"), opt.Dim("H")
local X = ad.Image("X",W,H)
local A = ad.Image("A",W,H)
local P = opt.InBounds(5,5)


local w = 0.1 -- keep us rational for now
local laplacianCost = (4*X(0,0) - (X(-1,0) + X(0,1) + X(1,0) + X(0,-1)))
local laplacianCostF = ad.select(P,laplacianCost,0)
local reconstructionCost = X(0,0) - A(0,0)

local cost = laplacianCostF*laplacianCostF + w*reconstructionCost*reconstructionCost

return ad.Cost({W,H},{X,A},cost)
