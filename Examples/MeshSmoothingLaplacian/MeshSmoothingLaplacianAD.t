local IO = terralib.includec("stdio.h")
local adP = ad.ProblemSpec()
local P = adP.P
local W,H = opt.Dim("W",0), opt.Dim("H",1)

local X = adP:Image("X", opt.float3,W,H,0)
local A = adP:Image("A", opt.float3,W,H,1)
local G = adP:Graph("G", 0, "v0", W, H, 0, "v1", W, H, 1)
P:Stencil(2)
P:UsePreconditioner(false)

local C = terralib.includecstring [[
#include <math.h>
]]

local w_fitSqrt = adP:Param("w_fit", float, 0)
local w_regSqrt = adP:Param("w_reg", float, 1)


local cost = ad.sumsquared(w_fitSqrt*(X(0,0) - A(0,0)), w_regSqrt*(X(G.v1) - X(G.v0)))
return adP:Cost(cost)

