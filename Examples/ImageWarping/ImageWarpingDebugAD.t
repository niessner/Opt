

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

function evalDRot(CosAlpha, SinAlpha)
	return ad.Vector(-SinAlpha, -CosAlpha, CosAlpha, -SinAlpha)
end

function evalDR (angle)
	return evalDRot(ad.cos(angle), ad.sin(angle))
end

function mul(matrix, v)
	return ad.Vector(matrix(0)*v(0)+matrix(1)*v(1), matrix(2)*v(0)+matrix(3)*v(1))
end

local terms = terralib.newlist()

local m = Mask(0,0)	-- float
local x = ad.Vector(X(0,0,0), X(0,0,1))	-- uv-unknown : float2

--fitting
local constraintUV = Constraints(0,0)	-- float2

--local e_fit = ad.select(ad.eq(m,0.0), x - constraintUV, ad.Vector(0.0, 0.0))
--e_fit = ad.select(ad.greatereq(constraintUV(0), 0.0), e_fit, ad.Vector(0.0, 0.0))
--e_fit = ad.select(ad.greatereq(constraintUV(1), 0.0), e_fit, ad.Vector(0.0, 0.0))
local e_fit = x - constraintUV

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


--local ARAPCost0F = ad.select(opt.InBounds(0,0,0,0),	ad.select(opt.InBounds( 1,0,0,0), ARAPCost0, ad.Vector(0.0, 0.0)), ad.Vector(0.0, 0.0))
--local ARAPCost1F = ad.select(opt.InBounds(0,0,0,0),	ad.select(opt.InBounds(-1,0,0,0), ARAPCost1, ad.Vector(0.0, 0.0)), ad.Vector(0.0, 0.0))
--local ARAPCost2F = ad.select(opt.InBounds(0,0,0,0),	ad.select(opt.InBounds( 0,1,0,0), ARAPCost2, ad.Vector(0.0, 0.0)), ad.Vector(0.0, 0.0))	
--local ARAPCost3F = ad.select(opt.InBounds(0,0,0,0),	ad.select(opt.InBounds(0,-1,0,0), ARAPCost3, ad.Vector(0.0, 0.0)), ad.Vector(0.0, 0.0))

local ARAPCost0F = ARAPCost0
local ARAPCost1F = ARAPCost1
local ARAPCost2F = ARAPCost2
local ARAPCost3F = ARAPCost3

local m0 = Mask( 1,0)
local m1 = Mask(-1,0)
local m2 = Mask( 0,1)
local m3 = Mask(0,-1)

--ARAPCost0F = ad.select(ad.eq(m0, 0.0), ARAPCost0F, ad.Vector(0.0, 0.0))
--ARAPCost1F = ad.select(ad.eq(m1, 0.0), ARAPCost1F, ad.Vector(0.0, 0.0))
--ARAPCost2F = ad.select(ad.eq(m2, 0.0), ARAPCost2F, ad.Vector(0.0, 0.0))
--ARAPCost3F = ad.select(ad.eq(m3, 0.0), ARAPCost3F, ad.Vector(0.0, 0.0))

for i = 0,1 do
	terms:insert(w_regSqrt*ARAPCost0F(i))
	terms:insert(w_regSqrt*ARAPCost1F(i))
	terms:insert(w_regSqrt*ARAPCost2F(i))
	terms:insert(w_regSqrt*ARAPCost3F(i))
end



local JTJTerms = terralib.newlist()

function getP(pImage, i, j) 
	return ad.Vector(pImage(i,j,0), pImage(i,j,1))
end

function dot(a,b)
	return a(0)*b(0) + a(1)*b(1)
end

do
	local b = ad.Vector(0.0, 0.0)
	local bA = 0.0
	local i = 0
	local j = 0
	local pImage = self:Image("P",opt.float3,W,H,-1)
	
	-- fit/pos
	local constraintUV = Constraints(0,0)	
	local validConstraint = (constraintUV(0) >= 0 and constraintUV(1) >= 0) and Mask(0,0) == 0.0
	if validConstraint then
	 	b = b + (2.0*w_fitSqrt*w_fitSqrt)*getP(pImage,i,j)
	end

	-- pos/reg
	local e_reg = ad.Vector(0.0, 0.0)
	local p00 = getP(pImage, i, j)

	--local valid0 = inBounds(gi+0, gj-1, X) and Mask(i+0, j-1) == 0.0
	--local valid1 = inBounds(gi+0, gj+1, X) and Mask(i+0, j+1) == 0.0
	--local valid2 = inBounds(gi-1, gj+0, X) and Mask(i-1, j+0) == 0.0
	--local valid3 = inBounds(gi+1, gj+0, X) and Mask(i+1, j+0) == 0.0
				
	--if valid0 then
		e_reg = e_reg + 2 * (p00 - getP(pImage,i+0, j-1))
	--end
	--if valid1 then
		e_reg = e_reg + 2 * (p00 - getP(pImage,i+0, j+1))
	--end
	--if valid2 then
		e_reg = e_reg + 2 * (p00 - getP(pImage,i-1, j+0))
	--end
	--if valid3 then
		e_reg = e_reg + 2 * (p00 - getP(pImage,i+1, j+0))
	--end
	
	
	b = b + (2.0*w_regSqrt*w_regSqrt)*e_reg

	-- angle/reg
	local e_reg_angle = 0.0
	local dR = evalDR(X(i,j)(2))
	local angleP = pImage(i,j)(2)
	local pHat = UrShape(i,j)
		
	--if valid0 then
	do
		local qHat = UrShape(i+0, j-1)
		local D = mul(dR,(pHat - qHat))
		e_reg_angle = e_reg_angle + dot(D,D)*angleP 
	end
	--if valid1 then
	do
		local qHat = UrShape(i+0, j+1)
		local D = mul(dR,(pHat - qHat))
		e_reg_angle = e_reg_angle + dot(D,D)*angleP 
	end
	--if valid2 then
	do
		local qHat = UrShape(i-1, j+0)
		var D = mul(dR,(pHat - qHat))
		e_reg_angle = e_reg_angle + dot(D,D)*angleP 
	end
	--if valid3 then
	do
		var qHat = UrShape(i+1, j+0)
		var D = mul(dR,(pHat - qHat))
		e_reg_angle = e_reg_angle + dot(D,D)*angleP 
	end
			
	bA = bA + (2.0*w_regSqrt*w_regSqrt)*e_reg_angle

	return make_float3(b(0), b(1), bA)

end







local cost = ad.sumsquared(unpack(terms))
return S:Cost(cost)


