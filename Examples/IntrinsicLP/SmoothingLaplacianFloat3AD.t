package.path = package.path .. ';?.t;'
require("helper")

local w_fitSqrt = S:Param("w_fitSqrt", float, 0)
local w_regSqrtAlbedo = S:Param("w_regSqrtAlbedo", float, 1)
local w_regSqrtShading = S:Param("w_regSqrtShading", float, 2)
local pNorm = S:Param("pNorm", float, 4)
local r = S:Unknown("r", opt.float3,{W,H},5)
local r_const = S:Image("r_const", opt.float3,{W,H},5)
local i = S:Image("i", opt.float3,{W,H},6)
local s = S:Unknown("s", float,{W,H},8)

local terms = terralib.newlist()
local offsets = { {1,0}, {-1,0}, {0,1}, {0,-1} }

-- reg Albedo
for j,o in ipairs(offsets) do
    local x,y = unpack(o)
	local diff = (r(0,0) - r(x,y))
	local diff_const = (r_const(0,0) - r_const(x,y))
    local laplacianCost = L_p(diff, diff_const, pNorm, S)
    local laplacianCostF = ad.select(opt.InBounds(0,0),ad.select(opt.InBounds(x,y), laplacianCost,0),0)
    terms:insert(w_regSqrtAlbedo*laplacianCostF)
end

-- reg Shading
for j,o in ipairs(offsets) do
    local x,y = unpack(o)
    local diff = (s(0,0) - s(x,y))
    local laplacianCostF = ad.select(opt.InBounds(0,0),ad.select(opt.InBounds(x,y), diff,0),0)
    terms:insert(w_regSqrtShading*laplacianCostF)
end

-- fit
local fittingCost = r(0,0)+s(0,0)-i(0,0)
terms:insert(w_fitSqrt*fittingCost)

return S:Cost(unpack(terms))
