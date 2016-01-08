local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()
local X = S:Image("X", opt.float4,W,H,0)
local A = S:Image("A", opt.float4,W,H,1)


local w_fitSqrt = S:Param("w_fitSqrt", float, 0)
local w_regSqrt = S:Param("w_regSqrt", float, 1)

local terms = terralib.newlist()

local offsets = { {1,0}, {-1,0}, {0,1}, {0,-1} }

useprecompute = false
for j,o in ipairs(offsets) do
    local x,y = unpack(o)
    local laplacianCost = X(0,0) - X(x,y)
    if useprecompute then
        local lc = S:ComputedImage("laplacian_"..tostring(j),W,H,laplacianCost)
        laplacianCost = lc(0,0)
    end
    local laplacianCostF = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(x,y,0,0), laplacianCost,0),0)
    terms:insert(w_regSqrt*laplacianCostF)
end
local fittingCost = X(0,0) - A(0,0)
terms:insert(w_fitSqrt*fittingCost)

local cost = ad.sumsquared(unpack(terms))
S:Cost(cost)

return S.P
