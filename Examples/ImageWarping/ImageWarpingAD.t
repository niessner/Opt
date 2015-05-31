

local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()
S:UsePreconditioner(true)

local X = 			S:Image("X", opt.float3,W,H,0)				--uv, a <- unknown
local UrShape = 	S:Image("UrShape", opt.float2,W,H,1)		--urshape
local Constraints = S:Image("Constraints", opt.float2,W,H,2)	--constraints
local Mask = 		S:Image("Mask", float, W,H,3)				--validity mask for constraints

local w_fitSqrt = S:Param("w_fitSqrt", float, 0)
local w_regSqrt = S:Param("w_regSqrt", float, 1)

function evalRot(CosAlpha, SinAlpha)
	return ad.Vector(CosAlpha, -SinAlpha, SinAlpha, CosAlpha)
end

function evalR (angle)
	return evalRot(ad.cos(angle), ad.sin(angle))
end

function mul(matrix, v)
	return ad.Vector(matrix(0)*v(0)+matrix(1)*v(1), matrix(2)*v(0)+matrix(3)*v(1))
end

local terms = terralib.newlist()

local m = Mask(0,0)	-- float
local x = ad.Vector(X(0,0,0), X(0,0,1))	-- uv-unknown : float2

--fitting
local constraintUV = Constraints(0,0)	-- float2

local e_fit = ad.select(ad.eq(m,0.0), x - constraintUV, ad.Vector(0.0, 0.0))
e_fit = ad.select(ad.greatereq(constraintUV(0), 0.0), e_fit, ad.Vector(0.0, 0.0))
e_fit = ad.select(ad.greatereq(constraintUV(1), 0.0), e_fit, ad.Vector(0.0, 0.0))

terms:insert(w_fitSqrt*e_fit(0))
terms:insert(w_fitSqrt*e_fit(1))


--regularization
local a = X(0,0,2)			-- rotation : float
local R = evalR(a)			-- rotation : float2x2
local xHat = UrShape(0,0)	-- uv-urshape : float2

local n0 = ad.Vector(X( 1,0,0), X( 1,0,1))
local n1 = ad.Vector(X(-1,0,0), X(-1,0,1))
local n2 = ad.Vector(X( 0,1,0), X( 0,1,1))
local n3 = ad.Vector(X(0,-1,0), X(0,-1,1))

local ARAPCost0 = (x - n0)	-	mul(R, (xHat - UrShape( 1,0)))
local ARAPCost1 = (x - n1)	-	mul(R, (xHat - UrShape(-1,0)))
local ARAPCost2 = (x - n2)	-	mul(R, (xHat - UrShape( 0,1)))
local ARAPCost3 = (x - n3)	-	mul(R, (xHat - UrShape(0,-1)))


local ARAPCost0F = ad.select(opt.InBounds(0,0,0,0),	ad.select(opt.InBounds( 1,0,0,0), ARAPCost0, ad.Vector(0.0, 0.0)), ad.Vector(0.0, 0.0))
local ARAPCost1F = ad.select(opt.InBounds(0,0,0,0),	ad.select(opt.InBounds(-1,0,0,0), ARAPCost1, ad.Vector(0.0, 0.0)), ad.Vector(0.0, 0.0))
local ARAPCost2F = ad.select(opt.InBounds(0,0,0,0),	ad.select(opt.InBounds( 0,1,0,0), ARAPCost2, ad.Vector(0.0, 0.0)), ad.Vector(0.0, 0.0))	
local ARAPCost3F = ad.select(opt.InBounds(0,0,0,0),	ad.select(opt.InBounds(0,-1,0,0), ARAPCost3, ad.Vector(0.0, 0.0)), ad.Vector(0.0, 0.0))

local m0 = Mask( 1,0)
local m1 = Mask(-1,0)
local m2 = Mask( 0,1)
local m3 = Mask(0,-1)

ARAPCost0F = ad.select(ad.eq(m0, 0.0), ARAPCost0F, ad.Vector(0.0, 0.0))
ARAPCost1F = ad.select(ad.eq(m1, 0.0), ARAPCost1F, ad.Vector(0.0, 0.0))
ARAPCost2F = ad.select(ad.eq(m2, 0.0), ARAPCost2F, ad.Vector(0.0, 0.0))
ARAPCost3F = ad.select(ad.eq(m3, 0.0), ARAPCost3F, ad.Vector(0.0, 0.0))

for i = 0,1 do
	--terms:insert(w_regSqrt*ARAPCost0F(i))
	--terms:insert(w_regSqrt*ARAPCost1F(i))
	--terms:insert(w_regSqrt*ARAPCost2F(i))
	--terms:insert(w_regSqrt*ARAPCost3F(i))
end

local cost = ad.sumsquared(unpack(terms))
return S:Cost(cost)


