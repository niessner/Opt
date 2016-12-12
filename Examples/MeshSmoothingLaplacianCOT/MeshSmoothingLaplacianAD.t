local adP = ad.ProblemSpec()
local P = adP.P
local N = opt.Dim("N",0)

local w_fitSqrt = adP:Param("w_fit", float, 0)
local w_regSqrt = adP:Param("w_reg", float, 1)
local X = adP:Unknown("X", opt.float3,{N},2)
local A = adP:Image("A", opt.float3,{N},3)
local G = adP:Graph("G", 4, "v0", {N}, 5, --current vertex
                            "v1", {N}, 7, --neighboring vertex
                            "v2", {N}, 9, --prev neighboring vertex
                            "v3", {N}, 11) --next neighboring vertex
P:UsePreconditioner(true)

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

local cost = {w_fitSqrt*(X(0) - A(0)), w_regSqrt*w*(X(G.v1) - X(G.v0))}
return adP:Cost(unpack(cost))

