local orig = require("imageSmoothing")
local W,H = orig.cost.dimensions[1],orig.cost.dimensions[2] --opt.Dim("W",0), opt.Dim("H",1)
local X = ad.Image("X",W,H,0)
local A = ad.Image("A",W,H,1)
local P = opt.InBounds(1,1)

local w_fit = 0.1 -- keep us rational for now
local w_reg = 1.0
local laplacianCost = (4*X(0,0) - (X(-1,0) + X(0,1) + X(1,0) + X(0,-1)))
local laplacianCostF = ad.select(P,laplacianCost,0)
local reconstructionCost = X(0,0) - A(0,0)

local cost = ad.sumsquared(math.sqrt(w_reg)*laplacianCostF,math.sqrt(w_fit)*reconstructionCost)

local tbl = ad.Cost(cost)

local applyJTJ = tbl.applyJTJ

local X,A = orig.cost.boundary:gettype().parameters[3],orig.cost.boundary:gettype().parameters[4]
assert(X)
assert(A)

local C = terralib.includecstring [[
#include <stdio.h>
]]

tbl.applyJTJ = terra(i : uint64, j : uint64, xImage : X, aImage : A, pImage : X)
    var v  = orig.applyJTJ.boundary(i,j,xImage,aImage,pImage)
    var v2 = applyJTJ.boundary(i,j,xImage,aImage,pImage)
    C.printf("%d %d %f %f\n",i,j,v,v2)
    return v
end

return tbl