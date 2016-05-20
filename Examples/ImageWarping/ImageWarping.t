local IO = terralib.includec("stdio.h")

local P = opt.ProblemSpec()
local W = opt.Dim("W",0)
local H = opt.Dim("H",1)

local Offset = P:Unknown("Offset",opt.float2,{W,H},0)
local Angle = P:Unknown("Angle",float,{W,H},1)
local UrShape = 	P:Image("UrShape", opt.float2,{W,H},2)		--urshape
local Constraints = P:Image("Constraints", opt.float2,{W,H},3)	--constraints
local Mask = 		P:Image("Mask", float, {W,H} ,4)				--validity mask for constraints
local w_fitSqrt = P:Param("w_fitSqrt", float, 5)
local w_regSqrt = P:Param("w_regSqrt", float, 6)

P:Stencil(2)
P:UsePreconditioner(true)
P:Dia

local WH = opt.toispace {W,H}

local TUnknownType = P:UnknownType():terratype()
local Index = WH:indextype()

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


local terra evalRot(CosAlpha : float, SinAlpha: float)
	return make_float4(CosAlpha, -SinAlpha, SinAlpha, CosAlpha)
	--return make_float4(CosAlpha, SinAlpha, -SinAlpha, CosAlpha)		
end

local terra evalR (angle : float)
	return evalRot(opt.math.cos(angle), opt.math.sin(angle))
end

local terra mul(matrix : float2x2, v : float_2) : float_2
	return make_float2(matrix(0)*v(0)+matrix(1)*v(1), matrix(2)*v(0)+matrix(3)*v(1))
end

local terra getXFloat2(idx : Index, self : P:ParameterType())
	return self.X.Offset(idx)
end

local terra eval_dR(cosAlpha : float, sinAlpha : float) 
	return make_float4(-sinAlpha, -cosAlpha, cosAlpha,  -sinAlpha)
	--return make_float4(-sinAlpha, cosAlpha, -cosAlpha,  -sinAlpha)
end

local terra evalR_dR(angle : float) 
	return eval_dR(opt.math.cos(angle), opt.math.sin(angle))
end

local F = {}


terra F.cost(idx : Index, self : P:ParameterType()) : float
	var e : float_2  = make_float2(0.0f, 0.0f)

	var x =  self.X.Offset(idx)
	var xHat : float_2 = self.UrShape(idx)
	var a = self.X.Angle(idx)(0)
	var m = self.Mask(idx)(0)
	var c = self.Constraints(idx)
	
	--e_fit
	if c(0) >= 0 and c(1) >= 0 and m == 0 then
		var e_fit : float_2 = make_float2(x(0) - self.Constraints(idx)(0), x(1) - self.Constraints(idx)(1))
		
		var tmp = make_float2(e(0) + (self.w_fitSqrt*self.w_fitSqrt) * (e_fit(0)*e_fit(0)), e(1) + (self.w_fitSqrt*self.w_fitSqrt) * (e_fit(1)*e_fit(1)))
		e = tmp
	end
	
	var right,left,up,down = idx(1,0),idx(-1,0),idx(0,1),idx(0,-1)
	
	var R : float2x2 = evalR(a)
	var e_reg : float_2 = make_float2(0.0f, 0.0f)
	
	--e_reg
	if right:InBounds() and self.Mask(right)(0) == 0.0f then
		var n : float_2 = getXFloat2(right,self) 
		var d : float_2 = (x - n) - mul(R, (xHat - self.UrShape(right)))
		e_reg = e_reg + d*d
	end
	if left:InBounds() and self.Mask(left)(0) == 0.0f then
		var n : float_2 = getXFloat2(left,self) 
		var d : float_2 = (x - n) - mul(R, (xHat - self.UrShape(left)))
		e_reg = e_reg + d*d
	end
	if up:InBounds() and self.Mask(up)(0) == 0.0f then
		var n : float_2 = getXFloat2(up,self) 
		var d : float_2 = (x - n) - mul(R, (xHat - self.UrShape(up)))
		e_reg = e_reg + d*d
	end
	if down:InBounds() and self.Mask(down)(0) == 0.0f then
		var n : float_2 = getXFloat2(down,self)
		var d : float_2 = (x - n) - mul(R, (xHat - self.UrShape(down)))
		e_reg = e_reg + d*d
	end
	
	e = e + (self.w_regSqrt*self.w_regSqrt)*e_reg
	

	var res : float = e(0) + e(1)
	return res
end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
terra F.evalJTF(idx : Index, self : P:ParameterType())
	var b = make_float2(0.0f, 0.0f)
	var bA = 0.0f
	var pre = make_float2(0.0f, 0.0f)
	var preA = 0.0f
	
	-- fit/pos
	var constraintUV = self.Constraints(idx)	
	var validConstraint = (constraintUV(0) >= 0 and constraintUV(1) >= 0) and self.Mask(idx)(0) == 0.0f
	if validConstraint then
	 	b 	= b + (2.0f*self.w_fitSqrt*self.w_fitSqrt)*(getXFloat2(idx,self) - constraintUV)
	 	pre = pre + (2.0f*self.w_fitSqrt*self.w_fitSqrt)*make_float2(1.0f, 1.0f) 
	end

	var a = self.X.Angle(idx)(0)
	-- reg/pos
	var	 p : float_2    = getXFloat2(idx,self)
	var	 pHat : float_2 = self.UrShape(idx)
	var R_i : float2x2  = evalR(a)
	var e_reg : float_2 	= make_float2(0.0f, 0.0f)

	var valid  = idx(0,0):InBounds() and (self.Mask(idx(0,0))(0) == 0.0f)
	var valid0 = idx(0,-1):InBounds() and (self.Mask(idx(0,-1))(0) == 0.0f)
	var valid1 = idx(0,1):InBounds() and (self.Mask(idx(0,1))(0) == 0.0f)
	var valid2 = idx(-1,0):InBounds() and (self.Mask(idx(-1,0))(0) == 0.0f)
	var valid3 = idx(1,0):InBounds() and (self.Mask(idx(1,0))(0) == 0.0f)

	var b_  = idx(0,0):InBounds()
	var b0 = idx(0,-1):InBounds()
	var b1 = idx(0,1):InBounds()
	var b2 = idx(-1,0):InBounds()
	var b3 = idx(1,0):InBounds()	
	
	b0 = b0 and b_
	b1 = b1 and b_
	b2 = b2 and b_
	b3 = b3 and b_

	var m  = (self.Mask(idx(0,0))(0) == 0.0f)
	var m0 = valid0
	var m1 = valid1
	var m2 = valid2
	var m3 = valid3
	
	if b0 then
		var q : float_2 	= getXFloat2(idx(0,-1), self)
		var qHat : float_2 	= self.UrShape(idx(0,-1))
		var R_j : float2x2 	= evalR(self.X.Angle(idx(0,-1))(0)) 
		if m0 then
			e_reg 			= e_reg + (p - q) - mul(R_i, pHat - qHat)
			pre 			= pre + (2.0f*self.w_regSqrt*self.w_regSqrt)*make_float2(1.0f, 1.0f) 
		end 
		if m then
			e_reg 			= e_reg + (p - q) - mul(R_j, pHat - qHat)
			pre 			= pre + (2.0f*self.w_regSqrt*self.w_regSqrt)*make_float2(1.0f, 1.0f) 
		end 
	end
	if b1 then
		var q : float_2 	= getXFloat2(idx(0,1), self)
		var qHat : float_2 	= self.UrShape(idx(0,1))
		var R_j : float2x2 	= evalR(self.X.Angle(idx(0,1))(0)) 
		if m1 then
			e_reg 			= e_reg + (p - q) - mul(R_i, pHat - qHat)
			pre 			= pre + (2.0f*self.w_regSqrt*self.w_regSqrt)*make_float2(1.0f, 1.0f) 
		end 
		if m then
			e_reg 			= e_reg + (p - q) - mul(R_j, pHat - qHat)
			pre 			= pre + (2.0f*self.w_regSqrt*self.w_regSqrt)*make_float2(1.0f, 1.0f) 
		end 
	end
	if b2 then
		var q : float_2 	= getXFloat2(idx(-1,0), self)
		var qHat : float_2 	= self.UrShape(idx(-1,0))
		var R_j : float2x2 	= evalR(self.X.Angle(idx(-1,0))(0)) 
		if m2 then
			e_reg 			= e_reg + (p - q) - mul(R_i, pHat - qHat)
			pre 			= pre + (2.0f*self.w_regSqrt*self.w_regSqrt)*make_float2(1.0f, 1.0f) 
		end 
		if m then
			e_reg 			= e_reg + (p - q) - mul(R_j, pHat - qHat)
			pre 			= pre + (2.0f*self.w_regSqrt*self.w_regSqrt)*make_float2(1.0f, 1.0f) 
		end 
	end
	if b3 then
		var q : float_2 	= getXFloat2(idx(1,0), self)
		var qHat : float_2 	= self.UrShape(idx(1,0))
		var R_j : float2x2 	= evalR(self.X.Angle(idx(1,0))(0)) 
		if m3 then
			e_reg 			= e_reg + (p - q) - mul(R_i, pHat - qHat)
			pre 			= pre + (2.0f*self.w_regSqrt*self.w_regSqrt)*make_float2(1.0f, 1.0f) 
		end 
		if m then
			e_reg 			= e_reg + (p - q) - mul(R_j, pHat - qHat)
			pre 			= pre + (2.0f*self.w_regSqrt*self.w_regSqrt)*make_float2(1.0f, 1.0f) 
		end 
	end
	
	b = b + (2.0f * self.w_regSqrt*self.w_regSqrt) * e_reg
	
	-- reg/angle
	var R : float2x2 = evalR(a)
	var dR : float2x2 = evalR_dR(a)
	var e_reg_angle = 0.0f
	
	
	if valid0 then
		var q : float_2 	= getXFloat2(idx(0,-1), self)
		var qHat : float_2 	= self.UrShape(idx(0,-1))
		var D : float_2 	= -mul(dR,(pHat - qHat))

		e_reg_angle 		= e_reg_angle 	+ D:dot((p - q) - mul(R,(pHat - qHat)))
		preA 				= preA 			+ D:dot(D) * self.w_regSqrt*self.w_regSqrt
	end

	if valid1 then
		var q : float_2 	= getXFloat2(idx(0,1), self)
		var qHat : float_2 	= self.UrShape(idx(0,1))
		var D : float_2 	= -mul(dR,(pHat - qHat))
		e_reg_angle 		= e_reg_angle 	+ D:dot((p - q) - mul(R,(pHat - qHat)))
		preA 				= preA 			+ D:dot(D) * self.w_regSqrt*self.w_regSqrt
	end

	if valid2 then
		var q : float_2 	= getXFloat2(idx(-1,0), self)
		var qHat : float_2 	= self.UrShape(idx(-1,0))
		var D : float_2 	= -mul(dR,(pHat - qHat))
		e_reg_angle 		= e_reg_angle 	+ D:dot((p - q) - mul(R,(pHat - qHat)))
		preA 				= preA 			+ D:dot(D) * self.w_regSqrt*self.w_regSqrt
	end

	if valid3 then
		var q : float_2 	= getXFloat2(idx(1,0), self)
		var qHat : float_2 	= self.UrShape(idx(1,0))
		var D : float_2 	= -mul(dR,(pHat - qHat))		
		e_reg_angle 		= e_reg_angle 	+ D:dot((p - q) - mul(R,(pHat - qHat)))
		preA 				= preA 			+ D:dot(D) * self.w_regSqrt*self.w_regSqrt
	end
	
	preA = 2.0f* preA

	
	bA = bA + (2.0f*self.w_regSqrt*self.w_regSqrt)*e_reg_angle

	
	if not P.usepreconditioner then		--no pre-conditioner
		pre = make_float2(1.0f, 1.0f)
		preA = 1.0f
	end
	
	
	-- we actually just computed negative gradient, so negate to return positive gradient
	return (make_float3(b(0), b(1), bA)), make_float3(pre(0), pre(1), preA)

end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
terra F.gradient(idx : Index, self : P:ParameterType())
	return F.evalJTF(idx, self)._0
end
	
local terra getP(pImage : TUnknownType, idx : Index) 
	return pImage.Offset(idx)
end

-- eval 2*JtJ (note that we keep the '2' to make it consistent with the gradient
terra F.applyJTJ(idx : Index, self : P:ParameterType(), pImage : TUnknownType)
 	var b = make_float2(0.0f, 0.0f)
	var bA = 0.0f

	-- fit/pos
	var constraintUV = self.Constraints(idx)	
	var validConstraint = (constraintUV(0) >= 0 and constraintUV(1) >= 0) and self.Mask(idx)(0) == 0.0f
	if validConstraint then
	 	b = b + (2.0f*self.w_fitSqrt*self.w_fitSqrt)*getP(pImage,idx)
	end

	-- pos/reg
	var e_reg = make_float2(0.0f, 0.0f)
	var p00 = getP(pImage, idx)

	var valid0 = idx(0,-1):InBounds() and self.Mask(idx(0,-1))(0) == 0.0f
	var valid1 = idx(0,1):InBounds() and self.Mask(idx(0,1))(0) == 0.0f
	var valid2 = idx(-1,0):InBounds() and self.Mask(idx(-1,0))(0) == 0.0f
	var valid3 = idx(1,0):InBounds() and self.Mask(idx(1,0))(0) == 0.0f
	
	var b_  = idx(0,0):InBounds()
	var b0 = idx(0,-1):InBounds()
	var b1 = idx(0,1):InBounds()
	var b2 = idx(-1,0):InBounds()
	var b3 = idx(1,0):InBounds()	
	
	b0 = b0 and b_
	b1 = b1 and b_
	b2 = b2 and b_
	b3 = b3 and b_

	var m  = (self.Mask(idx(0,0))(0) == 0.0f)
	var m0 = valid0
	var m1 = valid1
	var m2 = valid2
	var m3 = valid3
				
	if b0 then
		if m then
			e_reg = e_reg + p00
			e_reg = e_reg - getP(pImage, idx(0, -1))
		end
		if m0 then
			e_reg = e_reg + p00
			e_reg = e_reg - getP(pImage, idx(0, -1))
		end	
		--e_reg = e_reg + 2 * (p00 - getP(pImage,i+0, j-1))
	end
	if b1 then
		if m then
			e_reg = e_reg + p00
			e_reg = e_reg - getP(pImage, idx(0, 1))
		end
		if m1 then
			e_reg = e_reg + p00
			e_reg = e_reg - getP(pImage, idx(0, 1))
		end	
		--e_reg = e_reg + 2 * (p00 - getP(pImage,i+0, j+1))
	end
	if b2 then
		if m then
			e_reg = e_reg + p00
			e_reg = e_reg - getP(pImage,idx(-1, 0))
		end
		if m2 then
			e_reg = e_reg + p00
			e_reg = e_reg - getP(pImage,idx(-1, 0))
		end		
		--e_reg = e_reg + 2 * (p00 - getP(pImage,idx(-1, 0)))
	end
	if b3 then
		if m then
			e_reg = e_reg + p00
			e_reg = e_reg - getP(pImage,idx(1, 0))
		end
		if m3 then
			e_reg = e_reg + p00
			e_reg = e_reg - getP(pImage,idx(1, 0))
		end		
		--e_reg = e_reg + 2 * (p00 - getP(pImage,idx(1, 0)))
	end
	
	b = b + (2.0f*self.w_regSqrt*self.w_regSqrt)*e_reg

	-- angle/reg
	var e_reg_angle = 0.0f;
	var dR : float2x2 = evalR_dR(self.X.Angle(idx)(0))
	var angleP = pImage.Angle(idx)(0)
	var pHat : float_2 = self.UrShape(idx)
		
	if valid0 then
		var qHat : float_2 = self.UrShape(idx(0,-1))
		var D : float_2 = mul(dR,(pHat - qHat))
		e_reg_angle = e_reg_angle + D:dot(D)*angleP 
	end
	if valid1 then
		var qHat : float_2 = self.UrShape(idx(0,1))
		var D : float_2 = mul(dR,(pHat - qHat))
		e_reg_angle = e_reg_angle + D:dot(D)*angleP 
	end
	if valid2 then
		var qHat : float_2 = self.UrShape(idx(-1,0))
		var D : float_2 = mul(dR,(pHat - qHat))
		e_reg_angle = e_reg_angle + D:dot(D)*angleP 
	end
	if valid3 then
		var qHat : float_2 = self.UrShape(idx(1,0))
		var D : float_2 = mul(dR,(pHat - qHat))
		e_reg_angle = e_reg_angle + D:dot(D)*angleP 
	end
			
	bA = bA + (2.0f*self.w_regSqrt*self.w_regSqrt)*e_reg_angle
	
	
	
	-- upper right block
	e_reg = make_float2(0.0f, 0.0f)	
	if b0 then
		var ni = 0
		var nj = -1
		var qHat : float_2 = self.UrShape(idx(ni,nj))
		var dR_j = evalR_dR(self.X.Angle(idx(ni,nj))(0))
		var D : float_2 	= -mul(dR,(pHat - qHat))
		var D_j : float_2	= mul(dR_j,(pHat - qHat))
		--e_reg = e_reg + (D*pImage(idx)(2) - D_j*pImage(ni,nj)(2))
		if m0 then 
			e_reg = e_reg + D*pImage.Angle(idx)(0)
		end
		if m then
			e_reg = e_reg - D_j*pImage.Angle(idx(ni,nj))(0)
		end
	end
	if b1 then
		var ni = 0
		var nj = 1
		var qHat : float_2 = self.UrShape(idx(ni,nj))
		var dR_j = evalR_dR(self.X.Angle(idx(ni,nj))(0))
		var D : float_2 	= -mul(dR,(pHat - qHat))
		var D_j : float_2	= mul(dR_j,(pHat - qHat))
		--e_reg = e_reg + (D*pImage(idx)(2) - D_j*pImage(ni,nj)(2))
		if m1 then 
			e_reg = e_reg + D*pImage.Angle(idx)(0)
		end
		if m then
			e_reg = e_reg - D_j*pImage.Angle(idx(ni,nj))(0)
		end
	end
	if b2 then
		var ni = -1
		var nj = 0
		var qHat : float_2 = self.UrShape(idx(ni,nj))
		var dR_j = evalR_dR(self.X.Angle(idx(ni,nj))(0))
		var D : float_2 	= -mul(dR,(pHat - qHat))
		var D_j : float_2	= mul(dR_j,(pHat - qHat))
		--e_reg = e_reg + (D*pImage(idx)(2) - D_j*pImage(ni,nj)(2))
		if m2 then 
			e_reg = e_reg + D*pImage.Angle(idx)(0)
		end
		if m then
			e_reg = e_reg - D_j*pImage.Angle(idx(ni,nj))(0)
		end
	end
	if b3 then
		var ni = 1
		var nj = 0
		var qHat : float_2 = self.UrShape(idx(ni,nj))
		var dR_j = evalR_dR(self.X.Angle(idx(ni,nj))(0))
		var D : float_2 	= -mul(dR,(pHat - qHat))
		var D_j : float_2	= mul(dR_j,(pHat - qHat))
		--e_reg = e_reg + (D*pImage(idx)(2) - D_j*pImage(ni,nj)(2))
		if m3 then 
			e_reg = e_reg + D*pImage.Angle(idx)(0)
		end
		if m then
			e_reg = e_reg - D_j*pImage.Angle(idx(ni,nj))(0)
		end
	end
	b = b + (2.0f*self.w_regSqrt*self.w_regSqrt)*e_reg
	
	-- lower left block
	e_reg_angle = 0.0f
	if valid0 then
		var ni = 0
		var nj = -1
		var qHat : float_2	= self.UrShape(idx(ni,nj))
		var D : float_2		= -mul(dR,(pHat - qHat))
		var diff : float_2	= getP(pImage,idx) - getP(pImage,idx(ni,nj))
		e_reg_angle = e_reg_angle + D:dot(diff)
	end
	if valid1 then
		var ni = 0
		var nj = 1
		var qHat : float_2	= self.UrShape(idx(ni,nj))
		var D : float_2		= -mul(dR,(pHat - qHat))
		var diff : float_2	= getP(pImage,idx) - getP(pImage,idx(ni,nj))
		e_reg_angle = e_reg_angle + D:dot(diff)
	end
	if valid2 then
		var ni = -1
		var nj = 0
		var qHat : float_2	= self.UrShape(idx(ni,nj))
		var D : float_2		= -mul(dR,(pHat - qHat))
		var diff : float_2	= getP(pImage,idx) - getP(pImage,idx(ni,nj))
		e_reg_angle = e_reg_angle + D:dot(diff)
	end
	if valid3 then
		var ni = 1
		var nj = 0
		var qHat : float_2	= self.UrShape(idx(ni,nj))
		var D : float_2		= -mul(dR,(pHat - qHat))
		var diff : float_2	= getP(pImage,idx) - getP(pImage,idx(ni,nj))
		e_reg_angle = e_reg_angle + D:dot(diff)
	end
	bA = bA + (2.0f*self.w_regSqrt*self.w_regSqrt)*e_reg_angle
	
	
	return make_float3(b(0), b(1), bA)
end

P:Functions(WH, F)

return P