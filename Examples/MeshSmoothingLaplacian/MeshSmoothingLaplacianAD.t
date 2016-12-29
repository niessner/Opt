local adP = ad.ProblemSpec()
local P = adP.P
local N = opt.Dim("N",0)

local w_fitSqrt = adP:Param("w_fit", float, 0)
local w_regSqrt = adP:Param("w_reg", float, 1)
local X = adP:Unknown("X", opt.float3,{N},2)
local A = adP:Image("A", opt.float3,{N},3)
local G = adP:Graph("G", 4, "v0", {N}, 5, "v1", {N}, 7)
P:Stencil(2)
P:UsePreconditioner(true)

local terms = terralib.newlist()
terms:insert(w_fitSqrt*(X(0) - A(0)))
terms:insert(w_regSqrt*(X(G.v1) - X(G.v0)))

return adP:Cost(unpack(terms))


