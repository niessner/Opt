local W,H = opt.Dim("W",0), opt.Dim("H",1)
local X = ad.Image("X",W,H,0)
local A = ad.Image("A",W,H,1)
local imageAdj = ad.Adjacency( {W,H}, {W,H}, 0)
local weights = ad.EdgeValues(float, imageAdj,0)

local w = 0.1 -- keep us rational for now
local laplacianCost = (4*X(0,0) - (X(-1,0) + X(0,1) + X(1,0) + X(0,-1)))
local sum = ad.sum(imageAdj,function(a) 
	return X(a) * w(a)	
end)
local sumWeights = ad.sum(imageAdj,function(x) return w(a) end)

local laplacianCostF = sumWeights*sum
local reconstructionCost = X(0,0) - A(0,0)

local cost = ad.sumsquared(laplacianCostF,math.sqrt(w)*reconstructionCost)

return ad.Cost(cost)