local IO = terralib.includec("stdio.h")
local P = opt.ProblemSpec()
local N = opt.Dim("N",0)

local w_fitSqrt = P:Param("w_fitSqrt", float, 0)
local w_regSqrt = P:Param("w_regSqrt", float, 1)
local X = 			P:Unknown("X", opt.float6,{N},2)			--vertex.xyz, rotation.xyz <- unknown
local UrShape = 	P:Image("UrShape", opt.float3,{N},3)		--urshape: vertex.xyz
local Constraints = P:Image("Constraints", opt.float3,{N},4)	--constraints
local G =   P:Graph("G", 5, "v0", {N}, 6,
                            "v1", {N}, 8)
P:Stencil(2)
P:UsePreconditioner(true)

local NS = opt.toispace { N }
local TUnknownType = P:UnknownType():terratype()

local unknownElement = P:UnknownType():VectorTypeForIndexSpace(NS)

local Index = NS:indextype()

local C = {}
local G = {}


local float_2 = opt.float2
local float_3 = opt.float3
local float_4 = opt.float4
local float_6 = opt.float6
local float_9 = opt.float9

local sin = opt.math.sin
local cos = opt.math.cos

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

local terra make_float3x3_diag(x0 : float, x1 : float, x2 : float)
	var v : float3x3
	v(0) = x0
	v(1) = 0
	v(2) = 0
	v(3) = 0
	v(4) = x1
	v(5) = 0
	v(6) = 0
	v(7) = 0
	v(8) = x2
	return v
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
	return evalRot(cos(a(0)), cos(a(1)), cos(a(2)), sin(a(0)), sin(a(1)), sin(a(2)))
end

local terra mul(matrix : float3x3, v : float_3) : float_3
	return make_float3(matrix(0)*v(0)+matrix(1)*v(1)+matrix(2)*v(2),matrix(3)*v(0)+matrix(4)*v(1)+matrix(5)*v(2),matrix(6)*v(0)+matrix(7)*v(1)+matrix(8)*v(2))
end

local terra matmul(A: float3x3, B: float3x3) : float3x3
	var result : float3x3
	escape
		--https://en.wikipedia.org/wiki/Matrix_multiplication#Matrix_product_.28two_matrices.29
		for i=0,2 do
			for j=0,2 do
				emit quote
					result(i*3+j) = A(i*3+0)*B(0*3+j) + A(i*3+1)*B(1*3+j) + A(i*3+2)*B(2*3+j)
				end
			end
		end
	end

	return result
end

terra C.cost(idx : Index, self : P:ParameterType()) : float
	
	var e : float_3  = make_float3(0.0f, 0.0f, 0.0f)
	var temp : float_6 = self.X.X(idx)
	var x : float_3 = make_float3(temp(0), temp(1), temp(2))
	var xHat : float_3 = self.UrShape(idx)
	var a : float_3 = make_float3(temp(3), temp(4), temp(5))
	var c : float_3 = self.Constraints(idx)

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


terra G.cost(idx : int32, self : P:ParameterType()) : float
	var n0 = self.G.v0[idx]
    var n1 = self.G.v1[idx]

	var temp : float_6 = self.X.X(n0)
	var a : float_3 = make_float3(temp(3), temp(4), temp(5))
	var R : float3x3 = evalR(a);
	var p = make_float3(temp(0), temp(1), temp(2))
	var pHat : float_3 = self.UrShape(n0)
	
	temp = self.X.X(n1)
	var q = make_float3(temp(0), temp(1), temp(2))
	var qHat : float_3 = self.UrShape(n1)
	var d : float_3 = (p - q) - mul(R,(pHat - qHat))
	var e_reg = (self.w_regSqrt*self.w_regSqrt)*d*d
	return e_reg(0) + e_reg(1) + e_reg(2)
end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
terra C.evalJTF(idx : Index, self : P:ParameterType())
	var b : float_3 = make_float3(0.0f, 0.0f, 0.0f)
	var pre : float_6 = make_float6(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
	var preT : float_3 = make_float3(0.0f, 0.0f, 0.0f)

	var ones = make_float3(1.0,1.0,1.0)
	
	var temp : float_6 = self.X.X(idx);
	var p : float_3 = make_float3(temp(0), temp(1), temp(2))
	var c : float_3 = self.Constraints(idx)
	-- TODO: use -inf?
	if c(0) > -9999999.9 then
		b   = b + 2.0*(self.w_fitSqrt*self.w_fitSqrt) * (p - c);
		preT = preT + 2.0*(self.w_fitSqrt*self.w_fitSqrt) * ones;
	end
	var gradient = make_float6(b(0), b(1), b(2), 0.0f, 0.0f, 0.0f)

	if P.usepreconditioner then	
		pre(0) = preT(0)
		pre(1) = preT(1)
		pre(2) = preT(2)
		pre(3) = 0.0
		pre(4) = 0.0
		pre(5) = 0.0
	else
		preT = make_float3(1,1,1)
		pre(0) = preT(0)
		pre(1) = preT(1)
		pre(2) = preT(2)
		pre(3) = 1.0
		pre(4) = 1.0
		pre(5) = 1.0
	end
	
	return gradient, pre
end

local terra transpose(M : float3x3) : float3x3
	return make_float3x3(	M(0), M(3), M(6),
							M(1), M(4), M(7),
							M(2), M(5), M(8))
end

local terra evalDerivativeRotationTimesVector(dRAlpha : float3x3, dRBeta : float3x3, dRGamma : float3x3, d : float_3) : float3x3
	var R : float3x3 
	-- TODO: Is this the right direction (did I screw up row major vs col major)
	var b : float_3 = mul(dRAlpha,d) 
	R(0*3+0) = b(0) 
	R(1*3+0) = b(1) 
	R(2*3+0) = b(2)
	
	b = mul(dRBeta,d)
	R(0*3+1) = b(0)
	R(1*3+1) = b(1) 
	R(2*3+1) = b(2)
	
	b = mul(dRGamma,d) 
	R(0*3+2) = b(0) 
	R(1*3+2) = b(1) 
	R(2*3+2) = b(2)

	return R;
end

local terra evalRMat_dAlpha(CosAlpha : float, CosBeta : float, CosGamma : float, SinAlpha : float, SinBeta : float,  SinGamma : float) : float3x3
	-- TODO: Is this the right direction (did I screw up row major vs col major)
	var R : float3x3
	R(0*3+0) = 0.0f;
	R(0*3+1) = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;
	R(0*3+2) = SinGamma*CosAlpha - CosGamma*SinBeta*SinAlpha;

	R(1*3+0) = 0.0f;
	R(1*3+1) = -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha;
	R(1*3+2) = -CosGamma*CosAlpha - SinGamma*SinBeta*SinAlpha;

	R(2*3+0) = 0.0f;
	R(2*3+1) = CosBeta*CosAlpha;
	R(2*3+2) = -CosBeta*SinAlpha;

	return R;
end

local terra evalR_dAlpha(angles : float_3) -- angles = [alpha, beta, gamma]
	return evalRMat_dAlpha(cos(angles(0)), cos(angles(1)), cos(angles(2)), sin(angles(0)), sin(angles(1)), sin(angles(2)))
end

-- Rotation Matrix dBeta
local terra evalRMat_dBeta(CosAlpha : float, CosBeta : float, CosGamma : float, SinAlpha : float, SinBeta : float,  SinGamma : float) : float3x3
	-- TODO: Is this the right direction (did I screw up row major vs col major)
	var R : float3x3
	R(0*3+0) = -CosGamma*SinBeta;
	R(0*3+1) = CosGamma*CosBeta*SinAlpha;
	R(0*3+2) = CosGamma*CosBeta*CosAlpha;

	R(1*3+0) = -SinGamma*SinBeta;
	R(1*3+1) = SinGamma*CosBeta*SinAlpha;
	R(1*3+2) = SinGamma*CosBeta*CosAlpha;

	R(2*3+0) = -CosBeta;
	R(2*3+1) = -SinBeta*SinAlpha;
	R(2*3+2) = -SinBeta*CosAlpha;

	return R;
end

local terra evalR_dBeta(angles : float_3) -- angles = [alpha, beta, gamma]
	return evalRMat_dBeta(cos(angles(0)), cos(angles(1)), cos(angles(2)), sin(angles(0)), sin(angles(1)), sin(angles(2)))
end

-- Rotation Matrix dGamma
local terra evalRMat_dGamma(CosAlpha : float, CosBeta : float, CosGamma : float, SinAlpha : float, SinBeta : float,  SinGamma : float) : float3x3
	var R : float3x3
	R(0*3+0) = -SinGamma*CosBeta
	R(0*3+1) = -CosGamma*CosAlpha - SinGamma*SinBeta*SinAlpha
	R(0*3+2) = CosGamma*SinAlpha - SinGamma*SinBeta*CosAlpha

	R(1*3+0) = CosGamma*CosBeta
	R(1*3+1) = -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha
	R(1*3+2) = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha

	R(2*3+0) = 0.0f
	R(2*3+1) = 0.0f
	R(2*3+2) = 0.0f

	return R
end

local terra evalR_dGamma(angles : float_3) -- angles = [alpha, beta, gamma]
	return evalRMat_dGamma(cos(angles(0)), cos(angles(1)), cos(angles(2)), sin(angles(0)), sin(angles(1)), sin(angles(2)))
end



local terra evalDerivativeRotationMatrix(angles : float_3, dRAlpha : &float3x3, dRBeta : &float3x3, dRGamma : &float3x3) : float3x3
	var cosAlpha = cos(angles(0)) 
	var cosBeta  = cos(angles(1)) 
	var cosGamma = cos(angles(2))
	var sinAlpha = sin(angles(0)) 
	var sinBeta  = sin(angles(1)) 
	var sinGamma = sin(angles(2))

	@dRAlpha = evalRMat_dAlpha(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
	@dRBeta  = evalRMat_dBeta(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
	@dRGamma = evalRMat_dGamma(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
end

terra G.evalJTF(idx : int32, self : P:ParameterType(), r : TUnknownType, preconditioner : TUnknownType)
	
	var ones = make_float3(1.0,1.0,1.0)
	var identity = make_float3x3_diag(1.0, 1.0, 1.0)
	
	var n0 = self.G.v0[idx]
    var n1 = self.G.v1[idx]
	
	var tmp : float_6


	tmp = self.X.X(n0)
	var p : float_3 = make_float3(tmp(0), tmp(1), tmp(2))
	var a_i : float_3 = make_float3(tmp(3), tmp(4), tmp(5))
	var pHat : float_3 = self.UrShape(n0)
	var R_i : float3x3 = evalR(a_i)
	
	var dRAlpha : float3x3, dRBeta : float3x3, dRGamma : float3x3
	evalDerivativeRotationMatrix(a_i, &dRAlpha, &dRBeta, &dRGamma)

	tmp = self.X.X(n1)
	var q : float_3 = make_float3(tmp(0), tmp(1), tmp(2))
	var a_j : float_3 = make_float3(tmp(3), tmp(4), tmp(5))
	var qHat  : float_3 = self.UrShape(n1)
	var R_j  : float3x3 = evalR(a_j)
	
	var D_i : float3x3 = -evalDerivativeRotationTimesVector(dRAlpha, dRBeta, dRGamma, pHat - qHat)
	var f_i : float_3 = (p - q) - mul(R_i,(pHat - qHat))
	
	var c0 : float_6
	var c1 : float_6
	
	var pre0 : float_6
	var pre1 : float_6
	
	--1st part of the residual (d/d_i)
	do
		var e_reg		: float_3 = 1.0f*f_i
		var e_reg_angle : float_3 = mul(transpose(D_i), f_i)
	
		var b  : float_3 = 2.0f*(self.w_regSqrt*self.w_regSqrt)*e_reg;
		var bA : float_3 = 2.0f*(self.w_regSqrt*self.w_regSqrt)*e_reg_angle;
		var c = make_float6(b(0), b(1), b(2), bA(0), bA(1), bA(2))
		c0 = c
		
		var pre : float_3 = 2.0f*(self.w_regSqrt*self.w_regSqrt)*ones
		var preA : float3x3 = 2.0f*(self.w_regSqrt*self.w_regSqrt)*matmul(transpose(D_i),D_i)
		pre0 = make_float6(pre(0), pre(1), pre(2), preA(0), preA(4), preA(8))
	end
	--1st part of the residual (d/d_j)
	do
		var e_reg		: float_3 = -1.0f*f_i
		var e_reg_angle : float_3 = make_float3(0,0,0)
	
		var b  : float_3 = 2.0f*(self.w_regSqrt*self.w_regSqrt)*e_reg;
		var bA : float_3 = 2.0f*(self.w_regSqrt*self.w_regSqrt)*e_reg_angle;
		var c = make_float6(b(0), b(1), b(2), bA(0), bA(1), bA(2))
		c1 = c
		
		var pre : float_3 = 2.0f*(self.w_regSqrt*self.w_regSqrt)*ones
		pre1 = make_float6(pre(0), pre(1), pre(2), 0, 0, 0)
	end	
    
	--write results
	var _residuum0 = -c0
	var _residuum1 = -c1
	r.X:atomicAdd(n0, _residuum0)
	r.X:atomicAdd(n1, _residuum1)
	
	if P.usepreconditioner then	
		preconditioner.X:atomicAdd(n0, pre0)
		preconditioner.X:atomicAdd(n1, pre1)
	end
end
	
-- eval 2*JtJ (note that we keep the '2' to make it consistent with the gradient
terra C.applyJTJ(idx : Index, self : P:ParameterType(), pImage : TUnknownType) : unknownElement 
	
	var b 	= make_float3(0.0f, 0.0f, 0.0f)
	var tmp = pImage(idx)
	var p 	= make_float3(tmp(0), tmp(1), tmp(2))

	-- fit/pos
	var c : float_3 = self.Constraints(idx)
	-- TODO: use -inf?
	if c(0) >  -999999.9f then
		b = b + 2.0f*(self.w_fitSqrt*self.w_fitSqrt)*p;
	end

	return make_float6(b(0), b(1), b(2), 0.0f, 0.0f, 0.0f)
end

terra G.applyJTJ(idx : int32, self : P:ParameterType(), pImage : TUnknownType, Ap_X : TUnknownType)
    var n0 = self.G.v0[idx]
    var n1 = self.G.v1[idx]
	
	var tmp : float_6
	tmp = pImage(n0)
	var p : float_3      = make_float3(tmp(0), tmp(1), tmp(2))
	var pAngle : float_3 = make_float3(tmp(3), tmp(4), tmp(5))

	tmp = self.X.X(n0)
	var a_i : float_3 = make_float3(tmp(3), tmp(4), tmp(5))

	var dRAlpha : float3x3, dRBeta : float3x3, dRGamma : float3x3
	evalDerivativeRotationMatrix(a_i, &dRAlpha, &dRBeta, &dRGamma)

	var pHat : float_3 = self.UrShape(n0)
	var qHat : float_3 = self.UrShape(n1)
	var D : float3x3 = -evalDerivativeRotationTimesVector(dRAlpha, dRBeta, dRGamma, pHat - qHat)

	tmp = self.X.X(n1)
	var a_j : float_3 = make_float3(tmp(3), tmp(4), tmp(5))

	var dRAlphaJ : float3x3, dRBetaJ : float3x3, dRGammaJ : float3x3
	evalDerivativeRotationMatrix(a_j, &dRAlphaJ, &dRBetaJ, &dRGammaJ)
	var D_j :float3x3 = -evalDerivativeRotationTimesVector(dRAlphaJ, dRBetaJ, dRGammaJ, pHat - qHat)

	tmp = pImage(n1)
	var q  : float_3  = make_float3(tmp(0), tmp(1), tmp(2))
	
	var c0 : float_6
	var c1 : float_6
	
	do
		var e_reg = p - q
		var e_reg_angle = mul(matmul(transpose(D),D),pAngle)
		e_reg = e_reg + mul(D,pAngle)
		e_reg_angle = e_reg_angle + mul(transpose(D),(p - q))
		
		var b  = 2.0f*(self.w_regSqrt*self.w_regSqrt)*e_reg
		var bA = 2.0f*(self.w_regSqrt*self.w_regSqrt)*e_reg_angle
		var c : float_6 = make_float6(b(0), b(1), b(2), bA(0), bA(1), bA(2))
		c0 = c
	end
	
	do 
		var e_reg = q - p
		var e_reg_angle = make_float3(0,0,0)
		e_reg = e_reg - mul(D,pAngle)
		
		var b  = 2.0f*(self.w_regSqrt*self.w_regSqrt)*e_reg
		var bA = 2.0f*(self.w_regSqrt*self.w_regSqrt)*e_reg_angle
		var c : float_6 = make_float6(b(0), b(1), b(2), bA(0), bA(1), bA(2))
		c1 = c
	end
    
	Ap_X.X:atomicAdd(n0, c0)
    Ap_X.X:atomicAdd(n1, c1)
    
    var d = 0.0f
	d = d + pImage(n0):dot(c0)
	d = d + pImage(n1):dot(c1)	
	return d
end

P:Functions(NS,C)
P:Functions("G",G)

return P

