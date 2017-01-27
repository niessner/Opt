require("helper")
local N = Dim("N",0)
local X = Array1D("X", opt.float3,N,0)
local A = Array1D("A", opt.float3,N,1)
local G = Graph("G", 0, "v0", N, 0, --current vertex
                       "v1", N, 1, --neighboring vertex
                       "v2", N, 2, --prev neighboring vertex
                       "v3", N, 3) --next neighboring vertex

UsePreconditioner(true)

local w_fitSqrt = Param("w_fit", float, 0)
local w_regSqrt = Param("w_reg", float, 1)

function cot(v0, v1) 
	local adotb = Dot(v0, v1)
	local disc = Dot(v0, v0)*Dot(v1, v1) - adotb*adotb
	disc = Select(greater(disc, 0.0), disc,  0.0001)
	return Dot(v0, v1) / sqrt(disc)
end

function normalize(v)
	return v / sqrt(Dot(v, v))
end

function length(v0, v1) 
	local diff = v0 - v1
	return sqrt(Dot(diff, diff))
end


-- fit energy
Energy(w_fitSqrt*(X(0,0) - A(0,0)))

local a = normalize(X(G.v0) - X(G.v2)) --float3
local b = normalize(X(G.v1) - X(G.v2))	--float3
local c = normalize(X(G.v0) - X(G.v3))	--float3
local d = normalize(X(G.v1) - X(G.v3))	--float3

--cotangent laplacian; Meyer et al. 03
local w = 0.5*(cot(a,b) + cot(c,d))
w = sqrt(Select(greater(w, 0.0), w, 0.0001)) 
Energy(w_regSqrt*w*(X(G.v1) - X(G.v0)))

return Result()

