local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()
local X = S:Image("X", opt.float4,W,H,0)
local T = S:Image("T", opt.float4,W,H,1)
local M = S:Image("M", float, W,H,2)
S:UsePreconditioner(false)


local w_fitSqrt = S:Param("w_fitSqrt", float, 0)
local w_regSqrt = S:Param("w_regSqrt", float, 1)

local terms = terralib.newlist()


ad.excludeUnknown(X(0,0), ad.eq(M(0,0),0))

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

local fitting = ad.select(ad.not_(ad.eq(m, 0)), ad.Vector(0.0, 0.0, 0.0, 0.0)

for i = 0,3 do	

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
return S:Cost(cost)
