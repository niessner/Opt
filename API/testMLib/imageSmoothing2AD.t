local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()
local X = S:Image("X",opt.float2,W,H,0)
local A = S:Image("A",opt.float2,W,H,1)

local zero = S:Param("zero",float,0)

local terms = terralib.newlist()
for i = 0,1 do
    local w_fit = 0.1 -- keep us rational for now
    local w_reg = 1.0
    local laplacianCost = (4*X(0,0,i) - (X(-1,0,i) + X(0,1,i) + X(1,0,i) + X(0,-1,i)))
    local laplacianCostF = ad.select(opt.InBounds(0,0,1,1),laplacianCost,0)
    local reconstructionCost = X(0,0,i) - A(0,0,i) + zero
    terms:insert(math.sqrt(w_reg)*laplacianCostF)
    terms:insert(math.sqrt(w_fit)*reconstructionCost)
end
local cost = ad.sumsquared(unpack(terms))
return S:Cost(cost)
