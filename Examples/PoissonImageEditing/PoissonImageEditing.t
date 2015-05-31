local IO = terralib.includec("stdio.h")
local P = opt.ProblemSpec()
local W,H = opt.Dim("W",0), opt.Dim("H",1)

P:Image("X", opt.float4,W,H,0)
P:Image("T", opt.float4,W,H,1)
P:Image("M", float, W,H,2)
P:Stencil(1)

local C = terralib.includecstring [[
#include <math.h>
]]


local unknownElement = P:UnknownType().metamethods.typ


local terra inBounds(i : int64, j : int64, xImage : P:UnknownType()) : bool
	return i >= 0 and i < xImage:W() and j >= 0 and j < xImage:H()
end

local terra laplacianCost(x : unknownElement, t : unknownElement, ni : int64, nj : int64, ngi : int64, ngj : int64, xImage : P:UnknownType(), tImage : P:UnknownType()) : unknownElement
	
	var res : unknownElement = 0.0f
	
	if inBounds(ngi,ngj,xImage) then
		res = (x - xImage(ni,nj)) - (t - tImage(ni,nj))
	end

	return res	
end

local terra laplacianCostP(x : unknownElement, ni : int64, nj : int64, ngi : int64, ngj : int64, xImage : P:UnknownType()) : unknownElement
	
	var res : unknownElement = 0.0f
	
	if inBounds(ngi,ngj,xImage) then
		res = x - xImage(ni,nj)
	end

	return res	
end


local terra cost(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType()) : float
	
	var m = self.M(i, j)
	var x = self.X(i, j)
	var t = self.T(i, j)	
	
	var res : float = 0.0f
	
	if m == 0 then
		var l0 = laplacianCost(x,t,i+1,j+0,gi+1,gj+0,self.X,self.T)
		var l1 = laplacianCost(x,t,i+0,j+1,gi+0,gj+1,self.X,self.T)
		var l2 = laplacianCost(x,t,i-1,j+0,gi-1,gj+0,self.X,self.T)
		var l3 = laplacianCost(x,t,i+0,j-1,gi+0,gj-1,self.X,self.T)
	
		var l = l0*l0 + l1*l1 + l2*l2 + l3*l3
		res = l(0) + l(1) + l(2) + l(3)
	end
	
	--if gi == 0 and gj == 0 or gi == 10 and gj == 10 then
	--	printf("cost=%f (%d|%d); x=%f\n", e_reg(0), int(gi), int(gj), x(0)) 
	--end
	
	return res
end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
local terra gradient(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType()) : unknownElement

	var m = self.M(i, j)
	var x = self.X(i, j)
	var t = self.T(i, j)
	
	var res : unknownElement = 0.0f
	
	if m == 0 then	
		var l0 = 2.0f*laplacianCost(x,t,i+1,j+0,gi+1,gj+0,self.X,self.T)
		var l1 = 2.0f*laplacianCost(x,t,i+0,j+1,gi+0,gj+1,self.X,self.T)
		var l2 = 2.0f*laplacianCost(x,t,i-1,j+0,gi-1,gj+0,self.X,self.T)
		var l3 = 2.0f*laplacianCost(x,t,i+0,j-1,gi+0,gj-1,self.X,self.T)	
		
		var laplacian = l0 + l1 + l2 + l3
	
		res = 2.0f*laplacian		
	end
	
	--if gi == 0 and gj == 0 or gi == 10 and gj == 10 then
	--	printf("cost=%f (%d|%d); x=%f\n", e_reg(0), int(gi), int(gj), x(0)) 
	--end
	
	return res
end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
local terra evalJTF(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType())
	
	var gradient : unknownElement = gradient(i, j, gi, gj, self)
	
	var pre : float = 1.0f
	return gradient, pre
end
	
-- eval 2*JtJ (note that we keep the '2' to make it consistent with the gradient
local terra applyJTJ(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType(), pImage : P:UnknownType()) : unknownElement
	
	var m = self.M(i, j)
	var p = pImage(i, j)
	
	var res : unknownElement = 0.0f
	
	if m == 0 then
		var l0 = 2.0f*laplacianCostP(p,i+1,j+0,gi+1,gj+0,pImage)
		var l1 = 2.0f*laplacianCostP(p,i+0,j+1,gi+0,gj+1,pImage)
		var l2 = 2.0f*laplacianCostP(p,i-1,j+0,gi-1,gj+0,pImage)
		var l3 = 2.0f*laplacianCostP(p,i+0,j-1,gi+0,gj-1,pImage)	
		var laplacian = l0 + l1 + l2 + l3
		res = 2.0f*laplacian
	end
	
	return res
end


P:Function("cost", {W,H}, cost)
P:Function("gradient", {W,H}, gradient)
P:Function("evalJTF", {W,H}, evalJTF)
P:Function("applyJTJ", {W,H}, applyJTJ)

return P