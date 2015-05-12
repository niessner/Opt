local origFile = terralib.loadfile("E:/Work/DSL/Optimization/API/testMLib/imageSmoothing.t")
local orig = origFile()
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

local vprintf = terralib.externfunction("vprintf", {&int8,&int8} -> int)

tbl.applyJTJ.boundary = terra(i : uint64, j : uint64, xImage : X, aImage : A, pImage : X)
    var v : float  = orig.applyJTJ.boundary(i,j,xImage,aImage,pImage)
    var v2 : float = applyJTJ.boundary(i,j,xImage,aImage,pImage)
	--if i == 10 and j == 10 then
	var a : float = v - v2
	if a < 0.0f then a = -a end
	--if a > 0.001f then 
	--if i == 0 and j == 0 then
	var b  = 3
	if i >= b and i < xImage:W()-b and j >= b and j < xImage:H()-b and a > 0.1f then
		printf("%d %d %f %f\n",int(i),int(j),v,v2)
	end
  return v2
end

return tbl