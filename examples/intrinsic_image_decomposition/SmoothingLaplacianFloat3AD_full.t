package.path = package.path .. ';?.t;'
require("helper")

local w_fitSqrt = S:Param("w_fitSqrt", float, 0)
local w_regSqrtAlbedo = S:Param("w_regSqrtAlbedo", float, 1)
local w_regSqrtShading = S:Param("w_regSqrtShading", float, 2)
local w_regSqrtChroma = S:Param("w_regSqrtChroma", float, 3)
local pNorm = S:Param("pNorm", float, 4)
local r = S:Unknown("r", opt.float3,{W,H},5)
local r_const = S:Image("r_const", opt.float3,{W,H},5)
local i = S:Image("i", opt.float3,{W,H},6)
local I = S:Image("I", opt.float3,{W,H},7)
local s = S:Unknown("s", float,{W,H},8)

function intensity(color)
	return (color[0] + color[1] + color[2])/3.0
end

function chroma(c)
	return c/intensity(c);
end

function wiw(color)
	return 1-0.8*(1-intensity(color))
end

function length(v)
	return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
end

function wcs(color0, color1)
	return pow(2.71828, -15*length(chroma(color0)-chroma(color1)))
end

function color(r, s)
	local S = pow(2, s)
	local R = ad.Vector(pow(2, r[0]), pow(2, r[1]), pow(2, r[2]))
	return R*S
end

local terms = terralib.newlist()
local offsets = { {1,0}, {-1,0}, {0,1}, {0,-1} }

-- reg Albedo
for j,o in ipairs(offsets) do
    local x,y = unpack(o)
	local diff = sqrt(wcs(I(0, 0), I(x, y)))*(r(0,0) - r(x,y))
	local diff_const = sqrt(wcs(I(0, 0), I(x, y)))*(r_const(0,0) - r_const(x,y))
    local laplacianCost = L_p(diff, diff_const, pNorm, S)
    local laplacianCostF = ad.select(opt.InBounds(0,0),ad.select(opt.InBounds(x,y), laplacianCost,0),0)
    terms:insert(w_regSqrtAlbedo*laplacianCostF)
end

-- reg Shading
for j,o in ipairs(offsets) do
    local x,y = unpack(o)
    local diff = sqrt(1.0-wcs(I(0, 0), I(x, y)))*(s(0,0) - s(x,y))
    local laplacianCostF = ad.select(opt.InBounds(0,0),ad.select(opt.InBounds(x,y), diff,0),0)
    terms:insert(w_regSqrtShading*laplacianCostF)
end

-- fit
local fittingCost = r(0,0)+s(0,0)-i(0,0)
terms:insert(w_fitSqrt*sqrt(wiw(I(0, 0)))*fittingCost)

-- fit chroma
local fittingCostChroma = chroma(color(r(0, 0), s(0, 0))) - chroma(I(0, 0))
terms:insert(w_regSqrtChroma*fittingCostChroma)

return S:Cost(unpack(terms))
