local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()
S:UsePreconditioner(true)

local Offset = S:Unknown("Offset",opt.float2,{W,H},0)
local Angle = S:Unknown("Angle",float,{W,H},1)
local UrShape = 	S:Image("UrShape", opt.float2,{W,H},2)		--urshape
local Constraints = S:Image("Constraints", opt.float2,{W,H},3)	--constraints
local Mask = 		S:Image("Mask", float, {W,H},4)				--validity mask for constraints

local w_fitSqrt = S:Param("w_fitSqrt", float, 5)
local w_regSqrt = S:Param("w_regSqrt", float, 6)

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
local x = Offset(0,0)

--fitting
local constraintUV = Constraints(0,0)	-- float2

local e_fit = ad.select(ad.eq(m,0.0), x - constraintUV, ad.Vector(0.0, 0.0))
e_fit = ad.select(ad.greatereq(constraintUV(0), 0.0), e_fit, ad.Vector(0.0, 0.0))
e_fit = ad.select(ad.greatereq(constraintUV(1), 0.0), e_fit, ad.Vector(0.0, 0.0))
e_fit = w_fitSqrt*e_fit

terms:insert(e_fit)

--regularization
local a = Angle(0,0) -- rotation : float
local R = evalR(a)			-- rotation : float2x2
local xHat = UrShape(0,0)	-- uv-urshape : float2

local offsets = { {1,0}, {-1,0}, {0,1}, {0, -1} }
for ii ,o in ipairs(offsets) do
    local i,j = unpack(o)
    local n = Offset(i,j)
    local ARAPCost = (x - n) - mul(R, (xHat - UrShape( i,j)))
    local ARAPCostF = ad.select(opt.InBounds(0,0),	ad.select(opt.InBounds( i,j), ARAPCost, ad.Vector(0.0, 0.0)), ad.Vector(0.0, 0.0))
    local m = Mask(i,j)
    ARAPCostF = w_regSqrt*ad.select(ad.eq(m, 0.0), ARAPCostF, ad.Vector(0.0, 0.0))
    terms:insert(ARAPCostF)
end

return S:Cost(unpack(terms))


