local IO = terralib.includec("stdio.h")

local P = opt.ProblemSpec()
local W = opt.Dim("W",0)
local H = opt.Dim("H",1)

local X = 			P:Image("X", opt.float3,W,H,0)				--uv, a <- unknown
local UrShape = 	P:Image("UrShape", opt.float2,W,H,1)		--urshape
local Constraints = P:Image("Constraints", opt.float2,W,H,2)	--constraints
local Mask = 		P:Image("Mask", float, W,H,3)				--validity mask for constraints

local w_fitSqrt = P:Param("w_fitSqrt", float, 0)
local w_regSqrt = P:Param("w_regSqrt", float, 1)

P:Stencil(2)

local unknownElement = P:UnknownType().metamethods.typ

local C = terralib.includecstring [[
#include <math.h>
]]

-- e = sum_{i,j} [ e_reg_{i,j}^2 + e_fit_{i,j}^2 ]
-- e_reg_{i,j} = I(i,j) - 0.25 * [ I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1) ]
-- e_fit_{i,j} = || I(i,j) - \hat{I}(i,j) ||_2

local float_2 = opt.float2
local float_3 = opt.float3
local float_4 = opt.float4

--local float2x2 = vector(float, 4)
local float2x2 = float_4

local terra make_float2(x : float, y : float)
	var v : float_2
	v(0) = x
	v(1) = y
	return v
	--return float_2(x, y)
end

local terra make_float3(x : float, y : float, z : float)
	var v : float_3
	v(0) = x
	v(1) = y
	v(2) = z
	return v
	--return float_3(x, y, z)
end

local terra make_float4(x : float, y : float, z : float, w : float)
	var v : float_4
	v(0) = x
	v(1) = y
	v(2) = z
	v(3) = w
	return v
	--return float_4(x, y, z, w)
end

local terra inBounds(i : int64, j : int64, xImage : P:UnknownType()) : bool
	return i >= 0 and i < xImage:W() and j >= 0 and j < xImage:H()
end

local terra evalRot(CosAlpha : float, SinAlpha: float)
	--return vector(CosAlpha, -SinAlpha, SinAlpha, CosAlpha)
	return make_float4(CosAlpha, -SinAlpha, SinAlpha, CosAlpha)	
end

local terra evalR (angle : float)
	return evalRot(opt.math.cos(angle), opt.math.sin(angle))
end

local terra mul(matrix : float2x2, v : float_2) : float_2
	return float_2(matrix(0)*v(0)+matrix(1)*v(1), matrix(2)*v(0)+matrix(3)*v(1))
end

local terra getXFloat2(i : int64, j : int64, self : P:ParameterType())
	return make_float2(self.X(i,j)(0), self.X(i,j)(1))
end

local terra eval_dR(cosAlpha : float, sinAlpha : float) 
	return make_float4(-sinAlpha, -cosAlpha,
					 cosAlpha,  -sinAlpha)
end

local terra evalR_dR(angle : float) 
	return eval_dR(opt.math.cos(angle), opt.math.sin(angle))
end

local terra cost(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType()) : float
	var e : float_2  = make_float2(0.0f, 0.0f)

	var x = make_float2(self.X(i, j)(0), self.X(i, j)(1))
	var xHat : float_2 = self.UrShape(i,j)
	var a = self.X(i, j)(2)
	var m = self.Mask(i,j)
	var c = self.Constraints(i, j)
	
	--e_fit
	if c(0) >= 0 and c(1) >= 0 and m == 0 then
		var e_fit : float_2 = make_float2(x(0) - self.Constraints(i,j)(0), x(1) - self.Constraints(i,j)(1))
		
		var tmp = make_float2(e(0) + (self.w_fitSqrt*self.w_fitSqrt) * (e_fit(0)*e_fit(0)), e(1) + (self.w_fitSqrt*self.w_fitSqrt) * (e_fit(1)*e_fit(1)))
		e = tmp
	end
	
	var R : float2x2 = evalR(a)
	var e_reg : float_2 = make_float2(0.0f, 0.0f)
	
	--e_reg
	if inBounds(gi+1,gj+0,self.X) and self.Mask(i+1,j+0) == 0.0f then
		var n : float_2 = make_float2(self.X(i+1,j+0)(0), self.X(i+1,j+0)(1))
		var d : float_2 = (x - n) - mul(R, (xHat - self.UrShape(i+1,j+0)))
		e_reg = e_reg + d*d
	end
	if inBounds(gi-1,gj+0,self.X) and self.Mask(i-1,j+0) == 0.0f then
		var n : float_2 = make_float2(self.X(i-1,j+0)(0), self.X(i-1,j+0)(1))
		var d : float_2 = (x - n) - mul(R, (xHat - self.UrShape(i-1,j+0)))
		e_reg = e_reg + d*d
	end
	if inBounds(gi+0,gj+1,self.X) and self.Mask(i+0,j+1) == 0.0f then
		var n : float_2 = make_float2(self.X(i+0,j+1)(0), self.X(i+0,j+1)(1))
		var d : float_2 = (x - n) - mul(R, (xHat - self.UrShape(i+0,j+1)))
		e_reg = e_reg + d*d
	end
	if inBounds(gi+0,gj-1,self.X) and self.Mask(i+0,j-1) == 0.0f then
		var n : float_2 = make_float2(self.X(i+0,j-1)(0), self.X(i+0,j-1)(1))
		var d : float_2 = (x - n) - mul(R, (xHat - self.UrShape(i+0,j-1)))
		e_reg = e_reg + d*d
	end
	
	e = e + (self.w_regSqrt*self.w_regSqrt)*e_reg
	

	var res : float = e(0) + e(1)
	return res
end






-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
local terra evalJTF(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType())
	var b = make_float2(0.0f, 0.0f)
	var bA = 0.0f
	var pre = make_float2(0.0f, 0.0f)
	var preA = 0.0f
	
	-- fit/pos
	var constraintUV = self.Constraints(i,j)	
	var validConstraint = (constraintUV(0) >= 0 and constraintUV(1) >= 0) and self.Mask(i,j) == 0.0f
	if validConstraint then
	 	b 	= b + (2.0f*self.w_fitSqrt*self.w_fitSqrt)*(getXFloat2(i,j,self) - constraintUV)
	 	pre = pre + (2.0f*self.w_fitSqrt*self.w_fitSqrt)*make_float2(1.0f, 1.0f) 
	end

	var a = self.X(i,j)(2)
	-- reg/pos
	var	 p : float_2    = getXFloat2(i,j,self)
	var	 pHat : float_2 = self.UrShape(i,j)
	var R_i : float2x2  = evalR(a)
	var e_reg : float_2 	= make_float2(0.0f, 0.0f);

	var valid0 = inBounds(gi+0, gj-1, self.X) and self.Mask(i+0, j-1) == 0.0f
	var valid1 = inBounds(gi+0, gj+1, self.X) and self.Mask(i+0, j+1) == 0.0f
	var valid2 = inBounds(gi-1, gj+0, self.X) and self.Mask(i-1, j+0) == 0.0f
	var valid3 = inBounds(gi+1, gj+0, self.X) and self.Mask(i+1, j+0) == 0.0f

	if valid0 then
		var q : float_2 	= getXFloat2(	i+0, j-1, self)
		var qHat : float_2 	= self.UrShape(	i+0, j-1)
		var R_j : float2x2 	= evalR(self.X(	i+0, j-1)(2)) 
		e_reg 				= e_reg + 2 * (p - q) - mul(R_i + R_j, pHat - qHat)
		pre 				= pre + 2.0f*self.w_regSqrt*self.w_regSqrt
	end
	if valid1 then
		var q : float_2 	= getXFloat2(	i+0, j+1, self)
		var qHat : float_2 	= self.UrShape(	i+0, j+1)
		var R_j : float2x2 	= evalR(self.X(	i+0, j+1)(2)) 
		e_reg 				= e_reg + 2 * (p - q) - mul(R_i + R_j, pHat - qHat)
		pre 				= pre + 2.0f*self.w_regSqrt*self.w_regSqrt
	end
	if valid2 then
		var q : float_2 	= getXFloat2(	i-1, j+0, self)
		var qHat : float_2 	= self.UrShape(	i-1, j+0)
		var R_j : float2x2 	= evalR(self.X(	i-1, j+0)(2)) 
		e_reg 				= e_reg + 2 * (p - q) - mul(R_i + R_j, pHat - qHat)
		pre 				= pre + 2.0f*self.w_regSqrt*self.w_regSqrt
	end
	if valid3 then
		var q : float_2 	= getXFloat2(	i+1, j+0, self)
		var qHat : float_2 	= self.UrShape(	i+1, j+0)
		var R_j : float2x2 	= evalR(self.X(	i+1, j+0)(2)) 
		e_reg 				= e_reg + 2 * (p - q) - mul(R_i + R_j, pHat - qHat)
		pre 				= pre + 2.0f*self.w_regSqrt*self.w_regSqrt
	end
			
		
	b = b + (2.0f * self.w_regSqrt*self.w_regSqrt) * e_reg
	
	-- reg/angle
	var R : float2x2 = evalR(a)
	var dR : float2x2 = evalR_dR(a)
	var e_reg_angle = 0.0f
		
	if valid0 then
		var q : float_2 	= getXFloat2(	i+0, j-1, self)
		var qHat : float_2 	= self.UrShape(	i+0, j-1)
		var D : float_2 	= -mul(dR,(pHat - qHat))
		e_reg_angle 		= e_reg_angle 	+ D:dot((p - q) - mul(R,(pHat - qHat)))
		preA 				= preA 			+ D:dot(D) * self.w_regSqrt*self.w_regSqrt
	end

	if valid1 then
		var q : float_2 	= getXFloat2(	i+0, j+1, self)
		var qHat : float_2 	= self.UrShape(	i+0, j+1)
		var D : float_2 	= -mul(dR,(pHat - qHat))
		e_reg_angle 		= e_reg_angle 	+ D:dot((p - q) - mul(R,(pHat - qHat)))
		preA 				= preA 			+ D:dot(D) * self.w_regSqrt*self.w_regSqrt
	end

	if valid2 then
		var q : float_2 	= getXFloat2(	i-1, j+0, self)
		var qHat : float_2 	= self.UrShape(	i-1, j+0)
		var D : float_2 	= -mul(dR,(pHat - qHat))
		e_reg_angle 		= e_reg_angle 	+ D:dot((p - q) - mul(R,(pHat - qHat)))
		preA 				= preA 			+ D:dot(D) * self.w_regSqrt*self.w_regSqrt
	end

	if valid3 then
		var q : float_2 	= getXFloat2(	i+1, j+0, self)
		var qHat : float_2 	= self.UrShape(	i+1, j+0)
		var D : float_2 	= -mul(dR,(pHat - qHat))
		e_reg_angle 		= e_reg_angle 	+ D:dot((p - q) - mul(R,(pHat - qHat)))
		preA 				= preA 			+ D:dot(D) * self.w_regSqrt*self.w_regSqrt
	end
			

	bA = bA + (2.0f*self.w_regSqrt*self.w_regSqrt)*e_reg_angle

	-- disable preconditioner
	pre = make_float2(1.0f, 1.0f)
	preA = 1.0f
	-- we actually just computed negative gradient, so negate to return positive gradient
	-- Should we multiply by 2?
	return (make_float3(b(0), b(1), bA)), make_float3(pre(0), pre(1), preA)

end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
local terra gradient(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType())
	return evalJTF(i, j, gi, gj, self)._0
end
	
local terra getP(pImage : P:UnknownType(), i : int64, j : int64) 
	var p = pImage(i,j)
	return make_float2(p(0), p(1))
end

-- eval 2*JtJ (note that we keep the '2' to make it consistent with the gradient
local terra applyJTJ(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType(), pImage : P:UnknownType())
 	var b = make_float2(0.0f, 0.0f)
	var bA = 0.0f

	-- fit/pos
	var constraintUV = self.Constraints(i,j)	
	var validConstraint = (constraintUV(0) >= 0 and constraintUV(1) >= 0) and self.Mask(i,j) == 0.0f
	if validConstraint then
	 	b 	= b + (2.0f*self.w_fitSqrt*self.w_fitSqrt)*getP(pImage,i,j)
	end

	-- pos/reg
	var e_reg = make_float2(0.0f, 0.0f);
	var p00 = getP(pImage, i, j)

	var valid0 = inBounds(gi+0, gj-1, self.X) and self.Mask(i+0, j-1) == 0.0f
	var valid1 = inBounds(gi+0, gj+1, self.X) and self.Mask(i+0, j+1) == 0.0f
	var valid2 = inBounds(gi-1, gj+0, self.X) and self.Mask(i-1, j+0) == 0.0f
	var valid3 = inBounds(gi+1, gj+0, self.X) and self.Mask(i+1, j+0) == 0.0f
				
	if valid0 then
		e_reg = e_reg + 2 * (p00 - getP(pImage,i+0, j-1))
	end
	if valid1 then
		e_reg = e_reg + 2 * (p00 - getP(pImage,i+0, j+1))
	end
	if valid2 then
		e_reg = e_reg + 2 * (p00 - getP(pImage,i-1, j+0))
	end
	if valid3 then
		e_reg = e_reg + 2 * (p00 - getP(pImage,i+1, j+0))
	end
	
	b = b + (2.0f*self.w_regSqrt*self.w_regSqrt)*e_reg;

	-- angle/reg
	var e_reg_angle = 0.0f;
	var dR : float2x2 = evalR_dR(self.X(i,j)(2))
	var angleP = pImage(i,j)(2)
	var pHat :float_2 = self.UrShape(i,j)
		
	if valid0 then
		var qHat : float_2 = self.UrShape(i+0, j-1)
		var D : float_2 = mul(dR,(pHat - qHat))
		e_reg_angle = e_reg_angle + D:dot(D)*angleP 
	end
	if valid1 then
		var qHat : float_2 = self.UrShape(i+0, j+1)
		var D : float_2 = mul(dR,(pHat - qHat))
		e_reg_angle = e_reg_angle + D:dot(D)*angleP 
	end
	if valid2 then
		var qHat : float_2 = self.UrShape(i-1, j+0)
		var D : float_2 = mul(dR,(pHat - qHat))
		e_reg_angle = e_reg_angle + D:dot(D)*angleP 
	end
	if valid3 then
		var qHat : float_2 = self.UrShape(i+1, j+0)
		var D : float_2 = mul(dR,(pHat - qHat))
		e_reg_angle = e_reg_angle + D:dot(D)*angleP 
	end
			
	bA = bA + (2.0f*self.w_regSqrt*self.w_regSqrt)*e_reg_angle;

	return make_float3(b(0), b(1), bA)
end

P:Function("cost", {W,H}, cost)
P:Function("gradient", {W,H}, gradient)
P:Function("evalJTF", {W,H}, evalJTF)
P:Function("applyJTJ", {W,H}, applyJTJ)

return P