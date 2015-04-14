local w = 1/ad.toexp(10) -- keep us rational for now
local W = opt.Dim("W")
local H = opt.Dim("H")

local X = ad.Image("X",W,H)
local A = ad.Image("A",W,H)

local N = X:inbounds(-1,0) + X:inbounds(0,1) + X:inbounds(1,0) + X:inbounds(0,-1)
local laplacianCost = (4*X(0,0) - (X(-1,0) + X(0,1) + X(1,0) + X(0,-1)))/N
local reconstructionCost = X(0,0) - A(0,0)

local cost = laplacianCost*laplacianCost --+ w*reconstructionCost*reconstructionCost

return ad.Cost({W,H},{X,A},cost)
