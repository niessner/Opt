local C = terralib.includecstring[[
#include <stdio.h>
#include <math.h>
]]

local w = .1 -- keep us rational for now
local W,H = opt.Dim("W"), opt.Dim("H")

local X = ad.Image("X",W,H)
local A = ad.Image("A",W,H)

local P = opt.InBounds(1,1)
local laplacianCost = (4*X(0,0) - (X(-1,0) + X(0,1) + X(1,0) + X(0,-1)))
local laplacianCostF = ad.select(P,laplacianCost,0)
local reconstructionCost = X(0,0) - A(0,0)

local cost = laplacianCostF*laplacianCostF + w*reconstructionCost*reconstructionCost

return ad.Cost({W,H},{X,A},cost)

--[=[
local mc,mg = ct.cost.fn,ct.gradient
ct.cost.fn = terra(i : uint64, j : uint64, xImage : opt.Image(float,W,H), aImage : opt.Image(float,W,H))
    var a = mc(i,j,xImage,aImage)
    var b = orig.cost.fn(i,j,xImage,aImage)
    if C.fabs(a-b) > 0.1f then
        C.printf("cost (%d,%d) %f %f\n",i,j,a,b)
    end
    return a
end

ct.gradient = terra(i : uint64, j : uint64, xImage : opt.Image(float,W,H), aImage : opt.Image(float,W,H))
    var a = mg(i,j,xImage,aImage)
    var b = orig.gradient(i,j,xImage,aImage)
    if C.fabs(a-b) > 0.1f then
        C.printf("grad (%d,%d) %f %f\n",i,j,a,b)
    end
    return a
end]]

return ct
]=]