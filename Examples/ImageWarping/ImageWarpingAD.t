local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()

local X = S:Image("X", opt.float3,W,H,0)						--uv, a <- unknown
local Urshape = S:Image("Urshape", opt.float2,W,H,1)			--urshape
local Constraints = S:Image("Constraints", opt.float2,W,H,2)	--constraints
local Mask = S:Image("Mask", float, W,H,3)						--validity mask for constraints

local w_fitSqrt = S:Param("w_fitSqrt", float, 0)
local w_regSqrt = S:Param("w_regSqrt", float, 1)


local eval_dR(CosAlpha : float, SinAlpha : float) : Matrix
	local R : Matrix
	R(0,0) = -SinAlpha
	R(0,1) = -CosAlpha
	R(1,0) = CosAlpha
	R(1,1) = -SinAlpha;
	return R
end

local evalR(angle : float) : Matrix
	return evalR(ad.cos(angle), ad.sin(angle))
end


local terms = terralib.newlist()

local m = Mask(0,0)	-- float2
local x = Vector(X(0,0,0), X(0,0,1))	-- uv-unknown : float2

--fitting
local constraintUV = Constraints(0,0)	-- float2
local e_fit = Vector(0.0f, 0.0f)
--if constraintUV(0) ~= 0.0f and constraintUV(1) ~= 0.0f and m == 0 then
ad.select(m == 0, e_fit = constraintUV - x, e_fit = Vector(0.0f, 0.0f)

terms:insert(w_fitSqrt*e_fit)

--regularization
local a = X(0,0,2)			-- rotation : float
local R = evalR(a)			-- rotation : float2x2
local xHat = Urshape(0,0)	-- uv-urshape : float2


local ARAPCost0 = (x - X(1,0,i))	-	R*(xHat - UrShape(1,0))
local ARAPCost1 = (x - X(-1,0,i))	-	R*(xHat - UrShape(-1,0))
local ARAPCost2 = (x - X(0,1,i))	-	R*(xHat - UrShape(0,1))
local ARAPCost3 = (x - X(0,-1,i))	-	R*(XHat - UrShape(0,-1))

ARAPCost0 = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(1,0,0,0),	ARAPCost0, 0))
ARAPCost1 = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(-1,0,0,0),	ARAPCost1, 0))
ARAPCost2 = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(0,1,0,0),	ARAPCost2, 0))	
ARAPCost3 = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(0,-1,0,0),	ARAPCost3, 0))

for i = 0,1 do
	terms:insert(w_regSqrt*ARAPCost0(i))
	terms:insert(w_regSqrt*ARAPCost1(i))
	terms:insert(w_regSqrt*ARAPCost2(i))
	terms:insert(w_regSqrt*ARAPCost3(i))
end

--[[
for i = 0,3 do
	local laplacianCost0 = X(0,0,i) - X(1,0,i)
	local laplacianCost1 = X(0,0,i) - X(-1,0,i)
	local laplacianCost2 = X(0,0,i) - X(0,1,i)
	local laplacianCost3 = X(0,0,i) - X(0,-1,i)
	
	local laplacianCost0F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(1,0,0,0),laplacianCost0,0),0)
	local laplacianCost1F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(-1,0,0,0),laplacianCost1,0),0)
	local laplacianCost2F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(0,1,0,0),laplacianCost2,0),0)
	local laplacianCost3F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(0,-1,0,0),laplacianCost3,0),0)
	
	local fittingCost = X(0,0,i) - A(0,0,i)
	
	terms:insert(w_regSqrt*laplacianCost0F)
	terms:insert(w_regSqrt*laplacianCost1F)
	terms:insert(w_regSqrt*laplacianCost2F)
	terms:insert(w_regSqrt*laplacianCost3F)
	terms:insert(w_fitSqrt*fittingCost)
end
--]]

local cost = ad.sumsquared(unpack(terms))
return S:Cost(cost)
