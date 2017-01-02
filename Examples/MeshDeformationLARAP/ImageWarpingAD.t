package.path = package.path .. ';../shared/?.t;'
require("opt_precision")

if OPT_DOUBLE_PRECISION then
    OPT_FLOAT3 = double3
else
    OPT_FLOAT3 = float3
end

local W,H,D = opt.Dim("W",0), opt.Dim("H",1), opt.Dim("D",2)
local S = ad.ProblemSpec()
S:UsePreconditioner(true)

local Offset = S:Unknown("Offset",OPT_FLOAT3,{W,H,D},0)
local Angle = S:Unknown("Angle",OPT_FLOAT3,{W,H,D},1)
local UrShape = 	S:Image("UrShape", OPT_FLOAT3,{W,H,D},2)		--urshape
local Constraints = S:Image("Constraints", OPT_FLOAT3,{W,H,D},3)	--constraints

local w_fitSqrt = S:Param("w_fitSqrt", float, 4)
local w_regSqrt = S:Param("w_regSqrt", float, 5)

function evalRot(CosAlpha, CosBeta, CosGamma, SinAlpha, SinBeta, SinGamma)
	return ad.Vector(
		CosGamma*CosBeta, 
		-SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha, 
		SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha,
		SinGamma*CosBeta,
		CosGamma*CosAlpha + SinGamma*SinBeta*SinAlpha,
		-CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha,
		-SinBeta,
		CosBeta*SinAlpha,
		CosBeta*CosAlpha)
end
	
function evalR(alpha, beta, gamma)
	return evalRot(ad.cos(alpha), ad.cos(beta), ad.cos(gamma), ad.sin(alpha), ad.sin(beta), ad.sin(gamma))
end
	
function mul(matrix, v)
	return ad.Vector(
			matrix(0)*v(0)+matrix(1)*v(1)+matrix(2)*v(2),
			matrix(3)*v(0)+matrix(4)*v(1)+matrix(5)*v(2),
			matrix(6)*v(0)+matrix(7)*v(1)+matrix(8)*v(2))
end

local terms = terralib.newlist()

local x = Offset(0,0,0)

--fitting
local constraint = Constraints(0,0,0)	-- float3

local e_fit = x - constraint
e_fit = ad.select(ad.greatereq(constraint(0), -999999.9), e_fit, ad.Vector(0.0, 0.0, 0.0))
terms:insert(w_fitSqrt*e_fit)

--regularization
local a = Angle(0,0,0)				-- rotation : float3
local R = evalR(a(0), a(1), a(2))	-- rotation : float3x3
local xHat = UrShape(0,0,0)			-- uv-urshape : float3

local offsets = { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}}
for iii ,o in ipairs(offsets) do
		
		local i,j,k = unpack(o)
	    local n = Offset(i,j,k)

		--local ARAPCost = (x - xHat)
		--local ARAPCostF = ad.select(opt.InBounds(i,j,999999), ARAPCost, ad.Vector(0.0, 0.0, 0.0))
		--terms:insert(w_regSqrt*ARAPCostF)

		local ARAPCost = (x - n) - mul(R, (xHat - UrShape(i,j,k)))
		local ARAPCostF = ad.select(opt.InBounds(0,0,0),	ad.select(opt.InBounds(i,j,k), ARAPCost, ad.Vector(0.0, 0.0, 0.0)), ad.Vector(0.0, 0.0, 0.0))
		terms:insert(w_regSqrt*ARAPCostF)
end

return S:Cost(unpack(terms))
