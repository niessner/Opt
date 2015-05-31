local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()
local X = S:Image("X", opt.float4,W,H,0)
local T = S:Image("T", opt.float4,W,H,1)
local M = S:Image("M", float, W,H,2)
S:UsePreconditioner(false)


local w_fitSqrt = S:Param("w_fitSqrt", float, 0)
local w_regSqrt = S:Param("w_regSqrt", float, 1)

local terms = terralib.newlist()

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

--[[
local laplacianCost0 = ad.select(ad.or_(ad.eq(m,0), ad.eq(m0,0)), (p - q0) - (t - t0), ad.Vector(0.0, 0.0, 0.0, 0.0))
local laplacianCost1 = ad.select(ad.or_(ad.eq(m,0), ad.eq(m1,0)), (p - q1) - (t - t1), ad.Vector(0.0, 0.0, 0.0, 0.0))
local laplacianCost2 = ad.select(ad.or_(ad.eq(m,0), ad.eq(m2,0)), (p - q2) - (t - t2), ad.Vector(0.0, 0.0, 0.0, 0.0))
local laplacianCost3 = ad.select(ad.or_(ad.eq(m,0), ad.eq(m3,0)), (p - q3) - (t - t3), ad.Vector(0.0, 0.0, 0.0, 0.0))


for i = 0,3 do	
	--local laplacianCost0F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(1,0,0,0),laplacianCost0(i),0),0)
	--local laplacianCost1F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(-1,0,0,0),laplacianCost1(i),0),0)
	--local laplacianCost2F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(0,1,0,0),laplacianCost2(i),0),0)
	--local laplacianCost3F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(0,-1,0,0),laplacianCost3(i),0),0)
	
	local laplacianCost0F =  laplacianCost0(i)
	local laplacianCost1F =  laplacianCost1(i)
	local laplacianCost2F =  laplacianCost2(i)
	local laplacianCost3F =  laplacianCost3(i)
	
	terms:insert(laplacianCost0F)
	terms:insert(laplacianCost1F)
	terms:insert(laplacianCost2F)
	terms:insert(laplacianCost3F)
end
local cost = ad.sumsquared(unpack(terms))
--]]

local l0 = (p - q0) - (t - t0)
local l1 = (p - q1) - (t - t1)
local l2 = (p - q2) - (t - t2)
local l3 = (p - q3) - (t - t3)

--local laplacianCost = (4*p - (q0 + q1 + q2 + q3) - (4*t - (t0 + t1 + t2 + t3)))

local laplacianCost = l0 + l1 + l2 + l3
laplacianCost = ad.select(ad.eq(m,0),laplacianCost,0)
for i = 0,3 do	
	terms:insert(laplacianCost(i))
end

local cost = ad.sumsquared(unpack(terms))
return S:Cost(cost)
