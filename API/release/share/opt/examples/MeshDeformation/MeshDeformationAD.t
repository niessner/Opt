local N = Dim("N",0)
local w_fitSqrt = Param("w_fitSqrt", float, 0)
local w_regSqrt = Param("w_regSqrt", float, 1)
local Offset = 			Unknown("Offset", opt.float3,{N},2)			--vertex.xyz, rotation.xyz <- unknown
local Angle = 			Unknown("Angle", opt.float3,{N},3)			--vertex.xyz, rotation.xyz <- unknown
local UrShape = 		Image("UrShape", opt.float3, {N},4)			--urshape: vertex.xyz
local Constraints =		Image("Constraints", opt.float3,{N},5)		--constraints
local G = Graph("G", 6, "v0", {N}, 7, "v1", {N}, 9)

UsePreconditioner(true)

--fitting
local e_fit = Offset(0) - Constraints(0)
Energy(Select(greatereq(Constraints(0)(0), -999999.9),w_fitSqrt*e_fit,0))

--regularization
local ARAPCost = (Offset(G.v0) - Offset(G.v1)) 
               - Rotate3D(Angle(G.v0),UrShape(G.v0) - UrShape(G.v1))

Energy(w_regSqrt*ARAPCost)