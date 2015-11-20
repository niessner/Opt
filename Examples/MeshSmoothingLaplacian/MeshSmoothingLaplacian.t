local IO = terralib.includec("stdio.h")
local P = opt.ProblemSpec()
local W,H = opt.Dim("W",0), opt.Dim("H",1)

P:Image("X", opt.float3,W,H,0)
P:Image("A", opt.float3,W,H,1)
P:Adjacency("G", {W,H}, {W,H}, 0)
P:Stencil(2)

local C = terralib.includecstring [[
#include <math.h>
]]

--local w_fit = P:Param("w_fit", float, 0)
--local w_reg = P:Param("w_reg", float, 1)

-- TODO: this should be factored into a parameter
local w_fit = 50.0
local w_reg = 100.0

local unknownElement = P:UnknownType().metamethods.typ

local terra cost(i : int32, j : int32, gi : int32, gj : int32, self : P:ParameterType()) : float
	
	var x = self.X(i, j)
	var a = self.A(i, j)
	
	var v2 = x - a
	var e_fit = w_fit * v2 * v2	
	
	var laplacian : unknownElement = 0.0f
	for adj in self.G:neighbors(i,j) do
	    var l_n = x - self.X(adj.x,adj.y)
	    -- l_n*l_n gives different (wrong) results. Why?
	    var edgeCost = opt.float3(l_n(0)*l_n(0), l_n(1)*l_n(1), l_n(2)*l_n(2))
	    laplacian = laplacian + edgeCost
	    
	    printf("%d,%d: %f\n", i, adj.x, edgeCost*1000000.0);
	    
	end
		
	var e_reg = w_reg*laplacian
	
	var res : float = 
		e_fit(0) + e_fit(1) + e_fit(2) +
		e_reg(0) + e_reg(1) + e_reg(2) 
	
	--if gi == 0 and gj == 0 or gi == 10 and gj == 10 then
	--	printf("cost=%f (%d|%d); x=%f\n", e_reg(0), int(gi), int(gj), x(0)) 
	--end
	
	return res
end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
local terra gradient(i : int32, j : int32, gi : int32, gj : int32, self : P:ParameterType()) : unknownElement
	
	var x = self.X(i, j)
	var a = self.A(i, j)
	
	var e_fit = w_fit*2.0f * (x - a)
	
	
	var laplacian : unknownElement = 0.0f
	for adj in self.G:neighbors(i,j) do
		var l_n = x - self.X(adj.x,adj.y)
		laplacian = laplacian + 2.0f*l_n
	end
	
	var e_reg = 2.0f*w_reg*laplacian
	
	--if gi == 0 and gj == 0 or gi == 10 and gj == 10 then
	--	printf("cost=%f (%d|%d); x=%f\n", e_reg(0), int(gi), int(gj), x(0)) 
	--end
	
	return e_fit + e_reg
end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
local terra evalJTF(i : int32, j : int32, gi : int32, gj : int32, self : P:ParameterType())
	
	var gradient : unknownElement = gradient(i, j, gi, gj, self)
	
	var pre : float = 1.0f
	return gradient, pre
end
	
-- eval 2*JtJ (note that we keep the '2' to make it consistent with the gradient
local terra applyJTJ(i : int32, j : int32, gi : int32, gj : int32, self : P:ParameterType(), pImage : P:UnknownType()) : unknownElement
 
	var p = pImage(i, j)
	--fit
	var e_fit = w_fit*2.0f*p
	
	--reg
	var laplacian : unknownElement = 0.0f
	for adj in self.G:neighbors(i,j) do
		var l_n = p - pImage(adj.x,adj.y)
		laplacian = laplacian + 2.0f*l_n
	end
	var e_reg = 2.0f*w_reg*laplacian
	
	return e_fit + e_reg
end



P:Function("cost", {W,H}, cost)
P:Function("gradient", {W,H}, gradient)
P:Function("evalJTF", {W,H}, evalJTF)
P:Function("applyJTJ", {W,H}, applyJTJ)


return P
