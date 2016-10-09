
local W,H = Dim("W",0), Dim("H",1)

local w_fitSqrt = Param("w_fitSqrt", float, 0)
local w_regSqrt = Param("w_regSqrt", float, 1)
local X = Unknown("X", opt.float4,{W,H},2)
local A = Image("A", opt.float4,{W,H},3)
local G = Graph("G", 4, "v0", {W, H}, 5, "v1", {W, H}, 7)

Energy(w_fitSqrt*(X(0,0) - A(0,0))) 
Energy(w_regSqrt*(X(G.v0) - X(G.v1)),
        w_regSqrt*(X(G.v1) - X(G.v0)))