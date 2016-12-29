require("helper")

package.path = package.path .. ';../shared/?.t;'
require("opt_precision")

if OPT_DOUBLE_PRECISION then
    OPT_FLOAT3 = double3
else
    OPT_FLOAT3 = float3
end

local w_fitSqrt = S:Param("w_fitSqrt", float, 0)
local w_regSqrt = S:Param("w_regSqrt", float, 1)
local pNorm = S:Param("pNorm", float, 2)
local X = S:Unknown("X", OPT_FLOAT3,{W,H},3)
local X_const = S:Image("X_const", OPT_FLOAT3,{W,H},3)
local A = S:Image("A", OPT_FLOAT3,{W,H},4)

local terms = terralib.newlist()

local offsets = { {1,0}, {-1,0}, {0,1}, {0,-1} }

local useL_p = true

for j,o in ipairs(offsets) do
    local x,y = unpack(o)
    local diff = X(0,0) - X(x,y)
    local laplacianCost = diff
    if useL_p then
    	local diff_const = X_const(0,0) - X_const(x,y)
    	laplacianCost = L_p(diff, diff_const, pNorm, S)
    end
    local laplacianCostF = ad.select(opt.InBounds(0,0),ad.select(opt.InBounds(x,y), laplacianCost,0),0)
    terms:insert(w_regSqrt*laplacianCostF)
end
local fittingCost = X(0,0) - A(0,0)
terms:insert(w_fitSqrt*fittingCost)


return S:Cost(unpack(terms))
