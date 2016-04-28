local W,H = Dim("W",0), Dim("H",1)
UsePreconditioner(true)

local Offset = Unknown("Offset",float2,{W,H},0)
local Angle = Unknown("Angle",float,{W,H},1)
local UrShape = 	Image("UrShape", float2,{W,H},2)		--urshape
local Constraints = Image("Constraints", float2,{W,H},3)	--constraints
local Mask = 		Image("Mask", float, {W,H},4)				--validity mask for constraints

local w_fitSqrt = Param("w_fitSqrt", float, 5)
local w_regSqrt = Param("w_regSqrt", float, 6)

function Rotate2D(angle, v)
	local CosAlpha, SinAlpha = cos(angle), sin(angle)
    local matrix = Vector(CosAlpha, -SinAlpha, SinAlpha, CosAlpha)
	return Vector(matrix(0)*v(0)+matrix(1)*v(1), matrix(2)*v(0)+matrix(3)*v(1))
end

--fitting
local e_fit = w_fitSqrt*(Offset(0,0)- Constraints(0,0))
local valid = All(greatereq(Constraints(0,0),0))
Energy(Select(valid, e_fit , 0.0))

--regularization
for i,j in Stencil { {1,0}, {-1,0}, {0,1}, {0, -1} } do
    local ARAPCost = (Offset(0,0) - Offset(i,j))
                  - Rotate2D(Angle(0,0), (UrShape(0,0) - UrShape( i,j)))
    local valid = eq(Mask(i,j),0)
    Energy(Select(valid,w_regSqrt*ARAPCost,0))
end
