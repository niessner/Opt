local IO = terralib.includec("stdio.h")
local adP = ad.ProblemSpec()
local P = adP.P
local N = opt.Dim("N",0)


local w_fitSqrt = adP:Param("w_fitSqrt", float, 0)
local w_regSqrt = adP:Param("w_regSqrt", float, 1)
local w_rotSqrt = adP:Param("w_rotSqrt", float, 2)
local X = 			adP:Image("X", opt.float12,{N},3)			--vertex.xyz, rotation_matrix <- unknown
local UrShape = 	adP:Image("UrShape", opt.float3,{N},4)		--urshape: vertex.xyz
local Constraints = adP:Image("Constraints", opt.float3,{N},5)	--constraints
local G = adP:Graph("G", 6, "v0", {N}, 7, "v1", {N}, 9)
P:UsePreconditioner(true)	--really needed here

function dot3(v0, v1) 
	return v0(0)*v1(0)+v0(1)*v1(1)+v0(2)*v1(2)
end
	
function mul(matrix, v)
	return ad.Vector(
			matrix(0)*v(0)+matrix(1)*v(1)+matrix(2)*v(2),
			matrix(3)*v(0)+matrix(4)*v(1)+matrix(5)*v(2),
			matrix(6)*v(0)+matrix(7)*v(1)+matrix(8)*v(2))
end

local terms = terralib.newlist()
	
--fitting
local x_fit = ad.Vector(X(0,0), X(0,1), X(0,2))	--vertex-unknown : float3
local x = X(0)
local constraint = Constraints(0)						--target : float3
local e_fit = x_fit - constraint
e_fit = ad.select(ad.greatereq(constraint(0), -999999.9), e_fit, ad.Vector(0.0, 0.0, 0.0))
terms:insert(w_fitSqrt*e_fit)

--rot
local R = ad.Vector(x(3), x(4), x(5), x(6), x(7), x(8), x(9), x(10), x(11))	--matrix : float9
local c0 = ad.Vector(R(0), R(3), R(6))
local c1 = ad.Vector(R(1), R(4), R(7))
local c2 = ad.Vector(R(2), R(5), R(8))
terms:insert(w_rotSqrt*dot3(c0,c1))
terms:insert(w_rotSqrt*dot3(c0,c2))
terms:insert(w_rotSqrt*dot3(c1,c2))
terms:insert(w_rotSqrt*(dot3(c0,c0)-1))
terms:insert(w_rotSqrt*(dot3(c1,c1)-1))
terms:insert(w_rotSqrt*(dot3(c2,c2)-1))

--reg
local _x0 = X(G.v0)	--float12
local _x1 = X(G.v1)	--float12
local x0 = ad.Vector(_x0(0), _x0(1), _x0(2))	--vertex-unknown : float3
local x1 = ad.Vector(_x1(0), _x1(1), _x1(2))	--vertex-unknown : float3
local R0 = ad.Vector(_x0(3), _x0(4), _x0(5), _x0(6), _x0(7), _x0(8), _x0(9), _x0(10), _x0(11))	--matrix : float9

local x0Hat = UrShape(G.v0)	--uv-urshape : float3
local x1Hat = UrShape(G.v1)	--uv-urshape : float3
	

local regCost = (x1 - x0) - mul(R0, (x1Hat - x0Hat))

terms:insert(w_regSqrt*regCost)

return adP:Cost(unpack(terms))


