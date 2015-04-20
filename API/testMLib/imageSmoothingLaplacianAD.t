local W,H = opt.Dim("W"), opt.Dim("H")
local X = ad.Image("X",W,H)
local A = ad.Image("A",W,H)
local P = opt.InBounds(1,1)


local w = 1.0 -- keep us rational for now

local leftCost  = X(0,0) - X(-1, 0)
local rightCost = X(0,0) - X( 1, 0)
local upCost    = X(0,0) - X( 0,1)
local downCost  = X(0,0) - X( 0, -1)


local leftCostF = ad.select(P,leftCost,0)
local rightCostF = ad.select(P,rightCost,0)
local upCostF = ad.select(P,upCost,0)
local downCostF = ad.select(P,downCost,0)
local reconstructionCost = X(0,0) - A(0,0)

local cost = leftCostF*leftCostF + rightCostF*rightCostF + upCostF*upCostF    + downCostF*downCostF  + w*reconstructionCost*reconstructionCost 

return ad.Cost({W,H},{X,A},cost)
