local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()
local X = S:Unknown("X", float4,{W,H},0)
local T = S:Image("T", float4,{W,H},1)
local M = S:Image("M", float, {W,H},2)
S:UsePreconditioner(false)


local terms = terralib.newlist()


S:Exclude(ad.not_(ad.eq(M(0,0),0)))

-- mask
local m = M(0,0)
local m0 = M( 1,0)
local m1 = M(-1,0)
local m2 = M( 0,1)
local m3 = M(0,-1)

--image 0
local p = X(0,0)
local q0 = X( 1,0)
local q1 = X(-1,0)
local q2 = X( 0,1)
local q3 = X(0,-1)

-- image 1
local t = T(0,0)
local t0 = T( 1,0)
local t1 = T(-1,0)
local t2 = T( 0,1)
local t3 = T(0,-1)

local laplacianCost0 = ((p - q0) - (t - t0))
local laplacianCost1 = ((p - q1) - (t - t1))
local laplacianCost2 = ((p - q2) - (t - t2))
local laplacianCost3 = ((p - q3) - (t - t3))

local function f4(x)
    return ad.Vector(x, x, x, x)
end

laplacianCost0 = ad.select(opt.InBounds(0,0),ad.select(opt.InBounds(1,0),laplacianCost0,f4(0.0)),f4(0.0))
laplacianCost1 = ad.select(opt.InBounds(0,0),ad.select(opt.InBounds(-1,0),laplacianCost1,f4(0.0)),f4(0.0))
laplacianCost2 = ad.select(opt.InBounds(0,0),ad.select(opt.InBounds(0,1),laplacianCost2,f4(0.0)),f4(0.0))
laplacianCost3 = ad.select(opt.InBounds(0,0),ad.select(opt.InBounds(0,-1),laplacianCost3,f4(0.0)),f4(0.0))
	
terms:insert(laplacianCost0)
terms:insert(laplacianCost1)
terms:insert(laplacianCost2)
terms:insert(laplacianCost3)

--[[
-- Hack to get example to work with dumpJ
    local zero = 0.0
    local zeroIm = S:ComputedImage("zero",{W,H},zero)
    local hack = zeroIm(0,0)*(p(0)+p(1)+p(2)+p(3))
    terms:insert(hack)
    --]]

return S:Cost(unpack(terms))
