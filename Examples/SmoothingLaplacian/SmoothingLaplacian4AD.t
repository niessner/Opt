local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()
local X = S:Image("X",float[4],W,H,0)
local A = S:Image("A",float[4],W,H,1)


local w_fit = S:Param("w_fit", float, 0)
local w_reg = S:Param("w_reg", float, 1)

local terms = terralib.newlist()
for i = 0,1 do
	local laplacianCost0 = X(0,0,i) - X(1,0,i)
	local laplacianCost1 = X(0,0,i) - X(-1,0,i)
	local laplacianCost2 = X(0,0,i) - X(0,1,i)
	local laplacianCost3 = X(0,0,i) - X(0,-1,i)
	
	local laplacianCost0F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(1,0,0,0),laplacianCost0,0),0)
	local laplacianCost1F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(-1,0,0,0),laplacianCost1,0),0)
	local laplacianCost2F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(0,1,0,0),laplacianCost2,0),0)
	local laplacianCost3F = ad.select(opt.InBounds(0,0,0,0),ad.select(opt.InBounds(0,-1,0,0),laplacianCost3,0),0)
	
	local fittingCost = X(0,0,i) - A(0,0,i)
	
	terms:insert(ad.sqrt(w_reg)*laplacianCost0F)
	terms:insert(ad.sqrt(w_reg)*laplacianCost1F)
	terms:insert(ad.sqrt(w_reg)*laplacianCost2F)
	terms:insert(ad.sqrt(w_reg)*laplacianCost3F)
	terms:insret(ad.sqrt(w_fit)*fittingCost)
end

local cost = ad.sumsquared(unpack(terms))
return S:Cost(cost)
