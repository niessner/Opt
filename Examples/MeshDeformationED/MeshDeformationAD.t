local N = Dim("N",0)

local w_fitSqrt = Param("w_fitSqrt", float, 0)
local w_regSqrt = Param("w_regSqrt", float, 1)
local w_rotSqrt = Param("w_rotSqrt", float, 2)
local X = 			Unknown("X", opt_float12,{N},3)			--vertex.xyz, rotation_matrix <- unknown
local UrShape = 	Image("UrShape", opt_float3,{N},4)		--urshape: vertex.xyz
local Constraints = Image("Constraints", opt_float3,{N},5)	--constraints
local G = Graph("G", 6, "v0", {N}, 7, "v1", {N}, 9)
UsePreconditioner(true)	--really needed here

local Offset = Slice(X,0,3) -- select part of unknown for position

--fitting
local e_fit = Offset(0) - Constraints(0)
local valid = greatereq(Constraints(0)(0), -999999.9)
Energy(Select(valid, w_fitSqrt*e_fit, 0))

--rot
local RotMatrix = Slice(X,3,12) -- extract rotation matrix
local R = RotMatrix(0)
local c0 = Vector(R(0), R(3), R(6))
local c1 = Vector(R(1), R(4), R(7))
local c2 = Vector(R(2), R(5), R(8))
Energy(w_rotSqrt*Dot3(c0,c1))
Energy(w_rotSqrt*Dot3(c0,c2))
Energy(w_rotSqrt*Dot3(c1,c2))
Energy(w_rotSqrt*(Dot3(c0,c0)-1))
Energy(w_rotSqrt*(Dot3(c1,c1)-1))
Energy(w_rotSqrt*(Dot3(c2,c2)-1))

local regCost = (Offset(G.v1) - Offset(G.v0)) - 
                Matrix3x3Mul(RotMatrix(G.v0), (UrShape(G.v1) - UrShape(G.v0)))
Energy(w_regSqrt*regCost)