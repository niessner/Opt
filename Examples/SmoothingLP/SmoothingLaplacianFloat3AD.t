local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()

local w_fitSqrt = S:Param("w_fitSqrt", float, 0)
local w_regSqrt = S:Param("w_regSqrt", float, 1)
local pNorm = S:Param("pNorm", float, 2)
local X = S:Unknown("X", opt.float3,{W,H},3)
local X_const = S:Image("X_const", opt.float3,{W,H},3) -- Hack
local A = S:Image("A", opt.float3,{W,H},4)

function length(v)
    -- TODO: check if scalar and just return
    return ad.sqrt(v:dot(v))
end
local L_p_counter = 1
function L_p(diff, diff_const, p)
    local dist_const = length(diff_const)
    local eps = 0.0000001
    local C = ad.pow(dist_const+eps,(pNorm-2))
    local sqrtC = ad.sqrt(C)
    local sqrtCImage = S:ComputedImage("sqrtC_"..tostring(L_p_counter),{W,H},sqrtC)
    L_p_counter = L_p_counter + 1
    return sqrtCImage(0,0)*diff
end

local terms = terralib.newlist()

local offsets = { {1,0}, {-1,0}, {0,1}, {0,-1} }

local useL_p = true

for j,o in ipairs(offsets) do
    local x,y = unpack(o)
    local diff = X(0,0) - X(x,y)
    local laplacianCost = diff
    if useL_p then
    	local diff_const = X_const(0,0) - X_const(x,y)
    	laplacianCost = L_p(diff, diff_const, pNorm)
    end
    local laplacianCostF = ad.select(opt.InBounds(0,0),ad.select(opt.InBounds(x,y), laplacianCost,0),0)
    terms:insert(w_regSqrt*laplacianCostF)
end
local fittingCost = X(0,0) - A(0,0)
terms:insert(w_fitSqrt*fittingCost)


return S:Cost(unpack(terms))
