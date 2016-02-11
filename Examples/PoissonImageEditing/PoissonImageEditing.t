local IO = terralib.includec("stdio.h")
local adP = ad.ProblemSpec()
local P = adP.P
local W,H = opt.Dim("W",0), opt.Dim("H",1)

P:Image("X", opt.float4,{W,H},0)
P:Image("T", opt.float4,{W,H},1)
P:Image("M", float, {W,H} ,2)

P:Stencil(1)
P:UsePreconditioner(false)	--TODO needs to be implemented (in this file)

local C = terralib.includecstring [[
#include <math.h>
]]

local TUnknownType = P:UnknownType():terratype()
local unknownElement = P:UnknownType():ElementType()
local Index = P:UnknownType().ispace:indextype()


local terra laplacianCost(x : unknownElement, t : unknownElement, idx : Index, xImage : TUnknownType, tImage : TUnknownType) : unknownElement
	
	var res : unknownElement = 0.0f
	
	if idx:InBounds() then
		res = (x - xImage(idx)) - (t - tImage(idx))
	end

	return res	
end

local terra laplacianCostP(x : unknownElement, idx : Index, xImage : TUnknownType) : unknownElement
	
	var res : unknownElement = 0.0f
	
	if idx:InBounds() then
		res = x - xImage(idx)
	end

	return res	
end


local terra cost(idx : Index, self : P:ParameterType()) : float
	
	var m = self.M(idx)(0)
	var x = self.X(idx)
	var t = self.T(idx)	
	
	var res : float = 0.0f
	
	if m == 0 then
		var l0 = laplacianCost(x,t,idx(1,0),self.X,self.T)
		var l1 = laplacianCost(x,t,idx(0,1),self.X,self.T)
		var l2 = laplacianCost(x,t,idx(-1,0),self.X,self.T)
		var l3 = laplacianCost(x,t,idx(0,-1),self.X,self.T)
	
		var l = l0*l0 + l1*l1 + l2*l2 + l3*l3
		res = l(0) + l(1) + l(2) + l(3)
	end
	
	return res
end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
local terra gradient(idx : Index, self : P:ParameterType()) : unknownElement

	var m = self.M(idx)(0)
	var x = self.X(idx)
	var t = self.T(idx)
	
	var res : unknownElement = 0.0f
	
	if m == 0 then	
		var l0 = 2.0f*laplacianCost(x,t,idx(1,0),self.X,self.T)
		var l1 = 2.0f*laplacianCost(x,t,idx(0,1),self.X,self.T)
		var l2 = 2.0f*laplacianCost(x,t,idx(-1,0),self.X,self.T)
		var l3 = 2.0f*laplacianCost(x,t,idx(0,-1),self.X,self.T)	
		
		var laplacian = l0 + l1 + l2 + l3
	
		res = 2.0f*laplacian		
	end
	
	return res
end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
local terra evalJTF(idx : Index, self : P:ParameterType())
	
	var gradient : unknownElement = gradient(idx, self)
	
	var pre : float = 1.0f
	return gradient, pre
end
	
-- eval 2*JtJ (note that we keep the '2' to make it consistent with the gradient
local terra applyJTJ(idx : Index, self : P:ParameterType(), pImage : TUnknownType) : unknownElement
	
	var m = self.M(idx)(0)
	var p = pImage(idx)
	
	var res : unknownElement = 0.0f
	
	if m == 0 then
		var l0 = 2.0f*laplacianCostP(p,idx(1,0),pImage)
		var l1 = 2.0f*laplacianCostP(p,idx(0,1),pImage)
		var l2 = 2.0f*laplacianCostP(p,idx(-1,0),pImage)
		var l3 = 2.0f*laplacianCostP(p,idx(0,-1),pImage)	
		var laplacian = l0 + l1 + l2 + l3
		res = 2.0f*laplacian
	end
	
	return res
end

P:Function("cost", cost)
P:Function("gradient", gradient)
P:Function("evalJTF", evalJTF)
P:Function("applyJTJ", applyJTJ)

return P