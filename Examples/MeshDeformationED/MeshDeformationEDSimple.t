require("helper")
local N = Dim("N",0)

local X = 			Array1D("X", opt.float12,N,0)
local UrShape = 	Array1D("UrShape", opt.float3,N,1)
local Constraints = Array1D("Constraints", opt.float3,N,2)
local G = Graph("G", 0, 
                 "v0", N, 0, 
                 "v1", N, 1)
UsePreconditioner(true)

local w_fitSqrt = Param("w_fitSqrt", float, 0)
local w_regSqrt = Param("w_regSqrt", float, 1)
local w_rotSqrt = Param("w_rotSqrt", float, 2)

local Offset = Slice(X,0,3)

--fitting
local x_fit = ad.Vector(X(0,0,0), X(0,0,1), X(0,0,2))	--vertex-unknown : float3
local x = X(0,0)
local e_fit = Offset(0,0) - Constraints(0,0)
local valid = greatereq(Constraints(0,0,0), -999999.9)
Energy(Select(valid, w_fitSqrt*e_fit, 0))

--rot
local RotMatrix = Slice(X,3,12)
local R = RotMatrix(0,0)
local c0 = ad.Vector(R(0), R(3), R(6))
local c1 = ad.Vector(R(1), R(4), R(7))
local c2 = ad.Vector(R(2), R(5), R(8))
Energy(w_rotSqrt*Dot(c0,c1))
Energy(w_rotSqrt*Dot(c0,c2))
Energy(w_rotSqrt*Dot(c1,c2))
Energy(w_rotSqrt*(Dot(c0,c0)-1))
Energy(w_rotSqrt*(Dot(c1,c1)-1))
Energy(w_rotSqrt*(Dot(c2,c2)-1))

local regCost = (Offset(G.v1) - Offset(G.v0)) - 
                Matrix3x3Mul(RotMatrix(G.v0), (UrShape(G.v1) - UrShape(G.v0)))

Energy(w_regSqrt*regCost)

return Result()


