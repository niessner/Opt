local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()

local X = S:Image("X", opt.float3,W,H,0)						--uv, a <- unknown
local Urshape = S:Image("Urshape", opt.float2,W,H,1)			--urshape
local Constraints = S:Image("Constraints", opt.float2,W,H,2)	--constraints
local Mask = S:Image("Mask", float, W,H,3)						--validity mask for constraints

local w_fitSqrt = S:Param("w_fitSqrt", float, 0)
local w_regSqrt = S:Param("w_regSqrt", float, 1)

function eval_dR (CosAlpha, SinAlpha)
	return Vector(-SinAlpha, -CosAlpha, CosAlpha, -SinAlpha)
end

function evalR (angle)
	return evalR(ad.cos(angle), ad.sin(angle))
end

function mul(m, v)
	return Vector(m(0)*v(0)+m(1)*v(1), m(2)*v(0)+m(3)*v(1))
end

local terms = terralib.newlist()

local m = Mask(0,0)	-- float
local x = Vector(X(0,0,0), X(0,0,1))	-- uv-unknown : float2

--fitting
local constraintUV = Constraints(0,0)	-- float2
--if constraintUV(0) ~= 0.0f and constraintUV(1) ~= 0.0f and m == 0 then
local e_fit = ad.select(ad.equal(m,0), constraintUV - x, Vector(0.0, 0.0))

terms:insert(w_fitSqrt*e_fit)

--regularization
local a = X(0,0,2)			-- rotation : float
local R = evalR(a)			-- rotation : float2x2
local xHat = Urshape(0,0)	-- uv-urshape : float2


local ARAPCost0 = (x - X(1 ,0,i))	-	mul(R,(xHat - UrShape(1,0)))
local ARAPCost1 = (x - X(-1,0,i))	-	mul(R,(xHat - UrShape(-1,0)))
local ARAPCost2 = (x - X(0 ,1,i))	-	mul(R,(xHat - UrShape(0,1)))
local ARAPCost3 = (x - X(0,-1,i))	-	mul(R,(XHat - UrShape(0,-1)))

ARAPCost0 = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds( 1,0,0,0),	ARAPCost0, 0))
ARAPCost1 = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(-1,0,0,0),	ARAPCost1, 0))
ARAPCost2 = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds( 0,1,0,0),	ARAPCost2, 0))	
ARAPCost3 = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(0,-1,0,0),	ARAPCost3, 0))

for i = 0,1 do
	terms:insert(w_regSqrt*ARAPCost0(i))
	terms:insert(w_regSqrt*ARAPCost1(i))
	terms:insert(w_regSqrt*ARAPCost2(i))
	terms:insert(w_regSqrt*ARAPCost3(i))
end

local cost = ad.sumsquared(unpack(terms))
return S:Cost(cost)
