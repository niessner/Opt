local IO = terralib.includec("stdio.h")
local adP = ad.ProblemSpec()
local P = adP.P
local N = opt.Dim("N",0)

local w_fitSqrt = P:Param("w_fitSqrt", float, 0)
local w_regSqrt = P:Param("w_regSqrt", float, 1)
local X = adP:Unknown("X", opt.float3,{N},2)
local A = adP:Image("A", opt.float3,{N},3)
local G = adP:Graph("G", 4, "v0", {N}, 5, "v1", {N}, 7)
P:Stencil(2)
P:UsePreconditioner(true)

local C = terralib.includecstring [[
#include <math.h>
]]



local NS = opt.toispace { N }
local TUnknownType = P:UnknownType():terratype()
local Index = NS:indextype()

local C = {}
local G = {}

local unknownElement = P:UnknownType():VectorTypeForIndexSpace(NS)

local terra laplacianCost(idx : int32, self : P:ParameterType()) : unknownElement	
    var x0 = self.X(self.G.v0[idx])
    var x1 = self.X(self.G.v1[idx])
    return x0 - x1
end

terra C.cost(idx : Index, self : P:ParameterType()) : float
	var v2 = self.X(idx) - self.A(idx)
	var e_fit = self.w_fitSqrt*self.w_fitSqrt * v2 * v2	
	
	var res : float = e_fit(0) + e_fit(1) + e_fit(2)
	
	return res
end

terra G.cost(idx : int32, self : P:ParameterType()) : float
	var l0 = laplacianCost(idx, self)		
	var e_reg = self.w_regSqrt*self.w_regSqrt*l0*l0
	
	var res : float = e_reg(0) + e_reg(1) + e_reg(2)
	
	return res
end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
terra C.evalJTF(idx : Index, self : P:ParameterType())
	var x = self.X(idx)
	var a = self.A(idx)
	var gradient = self.w_fitSqrt*self.w_fitSqrt*2.0f * (x - a)	
	var pre : float = 1.0f
	if P.usepreconditioner then
		pre = self.w_fitSqrt*self.w_fitSqrt*2.0f
	end
	return gradient, pre
end


terra G.evalJTF(idx : int32, self : P:ParameterType(), r : TUnknownType, preconditioner : TUnknownType)	

	var lap = 2.0f*self.w_regSqrt*self.w_regSqrt*laplacianCost(idx, self)
	var c0 = ( 1.0f)*lap
	var c1 = (-1.0f)*lap
	
    var v0 = self.G.v0[idx]
    var v1 = self.G.v1[idx]
	--write results
	var _residuum0 = -c0
	var _residuum1 = -c1
	r.X:atomicAdd(v0, _residuum0)
	r.X:atomicAdd(v1, _residuum1)
	
	if P.usepreconditioner then
		var pre = 2.0f*(self.w_regSqrt*self.w_regSqrt)
		preconditioner.X:atomicAdd(v0, pre)
		preconditioner.X:atomicAdd(v1, pre)
	end
	
end
	
-- eval 2*JtJ (note that we keep the '2' to make it consistent with the gradient
terra C.applyJTJ(idx : Index, self : P:ParameterType(), pImage : TUnknownType, ctcImage : TUnknownType) : unknownElement 
    return self.w_fitSqrt*self.w_fitSqrt*2.0f*pImage(idx)
end

terra G.applyJTJ(idx : int32, self : P:ParameterType(), pImage : TUnknownType, Ap_X : TUnknownType)
    var v0 = self.G.v0[idx]
    var v1 = self.G.v1[idx]
    
    var p0 = pImage(v0)
    var p1 = pImage(v1)

    -- (1*p0) + (-1*p1)
    var l_n = p0 - p1
    var e_reg = 2.0f*self.w_regSqrt*self.w_regSqrt*l_n

	var c0 = 1.0 *  e_reg
	var c1 = -1.0f * e_reg
	

	Ap_X.X:atomicAdd(v0, c0)
    Ap_X.X:atomicAdd(v1, c1)

    var d = 0.0f
	d = d + pImage(v0):dot(c0)
	d = d + pImage(v1):dot(c1)					
	return d 

end


P:Functions(NS,C)
P:Functions("G",G)

return P
