local W,H = Dim("W",0), Dim("H",1)
local Offset = Unknown("Offset",opt_float2,{W,H},0)
local Angle = Unknown("Angle",opt_float,{W,H},1)			
local UrShape = Array("UrShape", opt_float2,{W,H},2) --original mesh position
local Constraints = Array("Constraints", opt_float2,{W,H},3) -- user constraints
local Mask = Array("Mask", opt_float, {W,H},4) -- validity mask for mesh
local w_fitSqrt = Param("w_fitSqrt", float, 5)
local w_regSqrt = Param("w_regSqrt", float, 6)
local G = Graph("G", 7, "v0", {W,H}, 8, "v1", {W,H}, 9)

UsePreconditioner(true)
--Exclude(Not(eq(Mask(0,0),0)))


--regularization
local ARAPCost = (Offset(G.v0) - Offset(G.v1)) - Rotate2D(Angle(G.v0),UrShape(G.v0) - UrShape(G.v1))
Energy(w_regSqrt*ARAPCost)

--fitting
local e_fit = (Offset(0,0)- Constraints(0,0))
local valid = All(greatereq(Constraints(0,0),0))
Energy(w_fitSqrt*Select(valid, e_fit , 0.0))