local IO = terralib.includec("stdio.h")
local adP = ad.ProblemSpec()
local P = adP.P
local W,H = opt.Dim("W",0), opt.Dim("H",1)

local X = 			adP:Image("X", opt.float6,W,H,0)			--vertex.xyz, rotation.xyz <- unknown
local UrShape = 	adP:Image("UrShape", opt.float3,W,H,1)		--urshape: vertex.xyz
local Constraints = adP:Image("Constraints", opt.float3,W,H,2)	--constraints
local G = adP:Graph("G", 0, "v0", W, H, 0, "v1", W, H, 1)
P:Stencil(2)

local w_fitSqrt = P:Param("w_fitSqrt", float, 0)
local w_regSqrt = P:Param("w_regSqrt", float, 1)


local C = terralib.includecstring [[
#include <math.h>
]]

local float_2 = opt.float2
local float_3 = opt.float3
local float_4 = opt.float4
local float_6 = opt.float6
local float_9 = opt.float9

--local float2x2 = vector(float, 4)
local float2x2 = float_4
local float3x3 = float_9

local terra make_float2(x : float, y : float)
	var v : float_2
	v(0) = x
	v(1) = y
	return v
end

local terra make_float3(x : float, y : float, z : float)
	var v : float_3
	v(0) = x
	v(1) = y
	v(2) = z
	return v
end

local terra make_float4(x : float, y : float, z : float, w : float)
	var v : float_4
	v(0) = x
	v(1) = y
	v(2) = z
	v(3) = w
	return v
end

local terra make_float6(	x0 : float, x1 : float, x2 : float, 
							x3 : float, x4 : float, x5 : float)
	var v : float_6
	v(0) = x0
	v(1) = x1
	v(2) = x2
	v(3) = x3
	v(4) = x4
	v(5) = x5
	return v
end

local terra make_float3x3(	x0 : float, x1 : float, x2 : float, 
							x3 : float, x4 : float, x5 : float, 
							x6 : float, x7 : float, x8 : float)
	var v : float3x3
	v(0) = x0
	v(1) = x1
	v(2) = x2
	v(3) = x3
	v(4) = x4
	v(5) = x5
	v(6) = x6
	v(7) = x7
	v(8) = x8
	return v
end


--local w_fit = P:Param("w_fit", float, 0)
--local w_reg = P:Param("w_reg", float, 1)

-- TODO: this should be factored into a parameter
local w_fit = 1.0
local w_reg = 0.0

local unknownElement = P:UnknownType().metamethods.typ

local terra laplacianCost(idx : int32, self : P:ParameterType()) : unknownElement	
    var x0 = self.X(self.G.v0_x[idx], self.G.v0_y[idx])
    var x1 = self.X(self.G.v1_x[idx], self.G.v1_y[idx])
    return x0 - x1
end


local terra  evalRot(CosAlpha : float, CosBeta  : float, CosGamma : float, SinAlpha : float, SinBeta : float, SinGamma : float)
	return make_float3x3(
		CosGamma*CosBeta, 
		-SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha, 
		SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha,
		SinGamma*CosBeta,
		CosGamma*CosAlpha + SinGamma*SinBeta*SinAlpha,
		-CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha,
		-SinBeta,
		CosBeta*SinAlpha,
		CosBeta*CosAlpha)
end
	
local terra evalR(a : float_3)
	return evalRot(opt.math.cos(a(0)), opt.math.cos(a(1)), opt.math.cos(a(2)), opt.math.sin(a(0)), opt.math.sin(a(1)), opt.math.sin(a(2)))
end

local terra mul(matrix: float_9, v: float_3)
	return make_float3(matrix(0)*v(0)+matrix(1)*v(1)+matrix(2)*v(2),matrix(3)*v(0)+matrix(4)*v(1)+matrix(5)*v(2),matrix(6)*v(0)+matrix(7)*v(1)+matrix(8)*v(2))
end

local terra cost(i : int32, j : int32, gi : int32, gj : int32, self : P:ParameterType()) : float
	
	var e : float_3  = make_float3(0.0f, 0.0f, 0.0f)
	var temp : float_6 = self.X(i,j)
	var x : float_3 = make_float3(temp(0), temp(1), temp(2))
	var xHat : float_3 = self.UrShape(i,j)
	var a : float_3 = make_float3(temp(3), temp(4), temp(5))
	var c : float_3 = self.Constraints(i, j)

	--e_fit
	if c(0) >= -999999.9f then
		var e_fit : float_3 = make_float3(x(0) - c(0), x(1) - c(1),	x(2) - c(2))
								
		var tmp = make_float3(
					e(0) + (self.w_fitSqrt*self.w_fitSqrt) * (e_fit(0)*e_fit(0)), 
					e(1) + (self.w_fitSqrt*self.w_fitSqrt) * (e_fit(1)*e_fit(1)),
					e(2) + (self.w_fitSqrt*self.w_fitSqrt) * (e_fit(2)*e_fit(2)))
		e = tmp
		
		
	end
	return e(0) + e(1) + e(2)
end



local terra cost_graph(idx : int32, self : P:ParameterType()) : float
	var w0,h0 = self.G.v0_x[idx], self.G.v0_y[idx]
    var w1,h1 = self.G.v1_x[idx], self.G.v1_y[idx]

	var temp : float_6 = self.X(w0, h0)
	var a : float_3 = make_float3(temp(3), temp(4), temp(5))
	var R : float_9 = evalR(a);
	var p = make_float3(temp(0), temp(1), temp(2))
	var pHat : float_3 = self.UrShape(w0, h0)
	
	temp = self.X(w1, h1)
	var q = make_float3(temp(0), temp(1), temp(2))
	var qHat : float_3 = self.UrShape(w1, h1)
	var d : float_3 = (p - q) - mul(R,(pHat - qHat))
	var e_reg = (self.w_regSqrt*self.w_regSqrt)*d*d
	return e_reg(0) + e_reg(1) + e_reg(2)
end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
local terra evalJTF(i : int32, j : int32, gi : int32, gj : int32, self : P:ParameterType())
	var b : float = 0.0f
	var pre : float = 1.0f

	var ones = make_float3(1.0,1.0,1.0)
	
	var temp : float_6 = self.X(i,j);
	var p : float_3 = make_float3(temp(0), temp(1), temp(2))
	var c : float_3 = self.Constraints(i, j)
	-- TODO: use -inf?
	if c(0) > -9999999.9 then
		b   = b + 2.0*(self.w_fitSqrt*self.w_fitSqrt) * (p - c);
		pre = pre + 2.0*(self.w_fitSqrt*self.w_fitSqrt) * ones;
	end
	var gradient = make_float6(b(0), b(1), b(2), 0.0f, 0.0f, 0.0f)

	return gradient, pre
end

local terra transpose(M : float3x3) : float3x3
	-- TODO implement
	return make_float3x3(	0.0, 0.0, 0.0,
							0.0, 0.0, 0.0,
							0.0, 0.0, 0.0)
end

local terra evalDerivativeRotationTimesVector(dRAlpha : float3x3, dRBeta : float3x3, dRGamma : float3x3, pMinQ : float_3) : float3x3
	-- TODO implement
	return make_float3x3(	0.0, 0.0, 0.0,
							0.0, 0.0, 0.0,
							0.0, 0.0, 0.0)
end

local terra evalDerivativeRotationMatrix(a : float_3, dRAlpha : float3x3, dRBeta : float3x3, dRGamma : float3x3) : float3x3
	return make_float3x3(0,0,0,0,0,0,0,0,0)
 -- TODO: Implement
end

local terra evalJTF_graph(idx : int32, self : P:ParameterType(), pImage : P:UnknownType(), r : P:UnknownType())
	
	var w0,h0 = self.G.v0_x[idx], self.G.v0_y[idx]
    var w1,h1 = self.G.v1_x[idx], self.G.v1_y[idx]
	
	var ones = make_float3(1.0,1.0,1.0)

	var temp : float_6 = self.X(w0,h0)
	var p : float_3 = make_float3(temp(0), temp(1), temp(2))
	var a : float_3 = make_float3(temp(3), temp(4), temp(5))

	var pHat : float_3 = self.UrShape(w0,h0)
	var R_i : float3x3 = evalR(a);
	var dRAlpha : float3x3, dRBeta : float3x3, dRGamma : float3x3
	evalDerivativeRotationMatrix(a, dRAlpha, dRBeta, dRGamma)

	temp = self.X(w1,h1)
	var q : float_3 = make_float3(temp(0), temp(1), temp(2))
	var a_neighbor : float_3 = make_float3(temp(3), temp(4), temp(5))
	var qHat  : float_3 = self.UrShape(w1,h1)
	var R_j  : float3x3 = evalR(a_neighbor);
	var D : float3x3  	= -evalDerivativeRotationTimesVector(dRAlpha, dRBeta, dRGamma, pHat - qHat);
	var P : float3x3 	= (self.w_fitSqrt*self.w_fitSqrt)*mul(transpose(D),(D));
	
	var e_reg		= 2.0f*(p - q) - (R_i+R_j)*(pHat - qHat);
	var pre			= 2.0f*(self.w_regSqrt*self.w_regSqrt)*ones;
	var e_reg_angle = transpose(D)*((p - q) - R_i*(pHat - qHat))
	var preA		= P
	
	-- TODO: Missing 2?
	var b  = self.w_regSqrt*self.w_regSqrt*e_reg;
	var bA = self.w_regSqrt*self.w_regSqrt*e_reg_angle;
	
	pre  = ones		-- TODO!!!
	preA = ones -- TODO!!!
	
	-- TODO: Preconditioner
	--[[
	if (fabs(pre(0)) > FLOAT_EPSILON && fabs(pre(1)) > FLOAT_EPSILON && fabs(pre(2)) > FLOAT_EPSILON) { pre(0) = 1.0f/pre(0);  pre(1) = 1.0f/pre(1);  pre(2) = 1.0f/pre(2); } else { pre = ones; }
	state.d_precondioner[variableIdx] = make_float3(pre(0), pre(1), pre(2));

	if (preA.det() > FLOAT_EPSILON) { preA = preA.getInverse(); } else { preA.setIdentity(); }
	state.d_precondionerA[variableIdx] = make_float3(preA(0, 0), preA(1, 1), preA(2, 2));
	]]

	var c = make_float6(b(0), b(1), b(2), bA(0), bA(1), bA(2))

	-- TODO: is this right?
	var c0 = ( 1.0f)*c
	var c1 = (-1.0f)*c

    
	--write results
	var _residuum0 = -c0
	var _residuum1 = -c1
	r:atomicAdd(w0, h0, _residuum0)
	r:atomicAdd(w1, h1, _residuum1)
	
	var _pre0 = make_float6(pre(0), pre(1), pre(2), preA(0), preA(1), preA(2))
	var _pre1 = _pre0
	-- TODO: Preconditioner
	--preconditioner:atomicAdd(w0, h0, _pre0)
	--preconditioner:atomicAdd(w1, h1, _pre1)
	
	var _p0 = _pre0*_residuum0
	var _p1 = _pre1*_residuum1
	pImage:atomicAdd(w0, h0, _p0)
	pImage:atomicAdd(w1, h1, _p1)
	
end
	
-- eval 2*JtJ (note that we keep the '2' to make it consistent with the gradient
local terra applyJTJ(i : int32, j : int32, gi : int32, gj : int32, self : P:ParameterType(), pImage : P:UnknownType()) : unknownElement 
	return 0.0
    --return w_fit*2.0f*pImage(i,j)
end

local terra applyJTJ_graph(idx : int32, self : P:ParameterType(), pImage : P:UnknownType(), Ap_X : P:UnknownType())
    --var w0,h0 = self.G.v0_x[idx], self.G.v0_y[idx]
    --var w1,h1 = self.G.v1_x[idx], self.G.v1_y[idx]
    --
    --var p0 = pImage(w0,h0)
    --var p1 = pImage(w1,h1)
    --
    ---- (1*p0) + (-1*p1)
    --var l_n = p0 - p1
    --var e_reg = 2.0f*w_reg*l_n
    --
	--var c0 = 1.0 *  e_reg
	--var c1 = -1.0f * e_reg
	--
    --
	--Ap_X:atomicAdd(w0, h0, c0)
    --Ap_X:atomicAdd(w1, h1, c1)
    --
    --var d = 0.0f
	--d = d + opt.Dot(pImage(w0,h0), c0)
	--d = d + opt.Dot(pImage(w1,h1), c1)					
	--return d 
	return 0.0
end

P:Function("cost", 		cost, "G", cost_graph)
P:Function("evalJTF", 	evalJTF, "G", evalJTF_graph)
P:Function("applyJTJ", 	applyJTJ, "G", applyJTJ_graph)


return P
