local IO = terralib.includec("stdio.h")
local adP = ad.ProblemSpec()
local P = adP.P
local W,H = opt.Dim("W",0), opt.Dim("H",1)

local w_fitSqrt = adP:Param("w_fit", float, 0)
local w_regSqrt = adP:Param("w_reg", float, 1)
local X = adP:Image("X", opt.float3,W,H,2)
local A = adP:Image("A", opt.float3,W,H,3)
--local G = adP:Graph("G", 0, "v0", W, H, 0, "v1", W, H, 1)
local G = adP:Graph("G", 4, "v0", W, H, 5,6, --current vertex
                            "v1", W, H, 7,8, --neighboring vertex
                            "v2", W, H, 9,10, --prev neighboring vertex
                            "v3", W, H, 11,12) --next neighboring vertex
P:Stencil(2)
P:UsePreconditioner(true)

local C = terralib.includecstring [[
#include <math.h>
]]



function dot3(v0, v1) 
	return v0(0)*v1(0)+v0(1)*v1(1)+v0(2)*v1(2)
end

function cot(v0, v1) 
	local adotb = dot3(v0, v1)
	local disc = dot3(v0, v0)*dot3(v1, v1) - adotb*adotb
	--disc = ad.abs(disc)
	disc = ad.select(ad.greater(disc, 0.0), disc,  0.0001)
	return dot3(v0, v1) / ad.sqrt(disc)
end

function angle(v0, v1)
	return ad.acos(dot3(v0,v1))
end

function normalize(v)
	local denom = ad.sqrt(dot3(v, v))
	return v / denom
end

function length(v0, v1) 
	local diff = v0 - v1
	return ad.sqrt(dot3(diff, diff))
end

local a = X(G.v0) - X(G.v2) --float3
local b = X(G.v1) - X(G.v2)	--float3
local c = X(G.v0) - X(G.v3)	--float3
local d = X(G.v1) - X(G.v3)	--float3

a = normalize(a)
b = normalize(b)
c = normalize(c)
d = normalize(d)

--cotangent laplacian; Meyer et al. 03
local w = 0.5*(cot(a,b) + cot(c,d))

--mean-value coordinates; Floater 03
--local w = (ad.tan(angle(a,b)*0.5) + ad.tan(angle(c,d)*0.5)) / length(X(G.v1), X(G.v0)) 
--w = ad.abs(w)

--w = ad.abs(w)
w = ad.select(ad.greater(w, 0.0), w, 0.0001)
w = ad.sqrt(w)

local cost = ad.sumsquared(w_fitSqrt*(X(0,0) - A(0,0)), w_regSqrt*w*(X(G.v1) - X(G.v0)))
return adP:Cost(cost)

