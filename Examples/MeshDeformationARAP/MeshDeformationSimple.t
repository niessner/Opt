require("helper")

local N = opt.Dim("N",0)
local X = 			Array1D("X", opt.float6,N,0)			--vertex.xyz, rotation.xyz <- unknown
local UrShape = 	Array1D("UrShape", opt.float3,N,1)		--urshape: vertex.xyz
local Constraints = Array1D("Constraints", opt.float3,N,2)	--constraints
local G = Graph("G", 0, 
                "v0", N, 0, 
                "v1", N, 1)

UsePreconditioner(true)
local w_fitSqrt = Param("w_fitSqrt", float, 0)
local w_regSqrt = Param("w_regSqrt", float, 1)

local Offset = Slice(X,0,3)
local Angle = Slice(X,3,6)

--fitting
local e_fit = Offset(0,0) - Constraints(0,0)
local valid = greatereq(Constraints(0,0,0), -999999.9)
Energy(Select(valid,w_fitSqrt*e_fit,0))

--regularization
local ARAPCost = (Offset(G.v0) - Offset(G.v1)) 
                  - Rotate(Angle(G.v0),UrShape(G.v0) - UrShape(G.v1))
Energy(w_regSqrt*ARAPCost)

return Result()