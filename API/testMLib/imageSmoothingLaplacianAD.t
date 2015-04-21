function square (a)
   return a*a
end

local W,H = opt.Dim("W"), opt.Dim("H")
local X = ad.Image("X",W,H)
local A = ad.Image("A",W,H)
local P = opt.InBounds(1,1)

local w = 1.0 -- keep us rational for now

local reconstructionCost = X(0,0) - A(0,0)
local neighbors = {X(-1, 0), X( 1, 0), X( 0,1), X( 0,-1)}

local cost = w*square(reconstructionCost)
for i, neighbor in ipairs(neighbors) do
    local edgeCost = ad.select(P,X(0,0) - neighbor,0)
    cost = cost + square(edgeCost)
end

return ad.Cost({W,H},{X,A},cost)
