local IO = terralib.includec("stdio.h")

local P = opt.ProblemSpec()
local W = opt.Dim("W",0)
local H = opt.Dim("H",1)

P:Image("X",float,W,H,0)
P:Image("A",float,W,H,1)
P:Param("Foo",int,0)
P:Stencil(2)

local C = terralib.includecstring [[
#include <math.h>
]]

-- e = sum_{i,j} [ e_reg_{i,j}^2 + e_fit_{i,j}^2 ]
-- e_reg_{i,j} = I(i,j) - 0.25 * [ I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1) ]
-- e_fit_{i,j} = || I(i,j) - \hat{I}(i,j) ||_2


-- TODO: this should be factored into a parameter
local w_fit = 0.1
local w_reg = 1.0


local terra inLaplacianBounds(i : int64, j : int64, xImage : P:UnknownType()) : bool
	return i > 0 and i < xImage:W()-1 and j > 0 and j < xImage:H()-1
end

local terra laplacian(i : int64, j : int64, gi : int64, gj : int64, xImage : P:UnknownType()) : float
	if not inLaplacianBounds(gi, gj, xImage) then
		return 0
	end

	var x = xImage(i, j)
	var n0 = xImage(i - 1, j)
    var n1 = xImage(i + 1, j)
    var n2 = xImage(i, j - 1)
    var n3 = xImage(i, j + 1)

	var v = 4*x - (n0 + n1 + n2 + n3)
	
	return v
end

-- diagonal of JtJ; i.e., (Jt)^2
local terra laplacianPreconditioner(gi : int64, gj : int64, xImage : P:UnknownType()) : float

	var p : float = 0.0
	if inLaplacianBounds(gi+0, gj+0, xImage) then	p = p + 4*4			end
	if inLaplacianBounds(gi+1, gj+0, xImage) then	p = p + (-1)*(-1)	end
	if inLaplacianBounds(gi+0, gj+1, xImage) then	p = p + (-1)*(-1)	end
	if inLaplacianBounds(gi-1, gj+0, xImage) then	p = p + (-1)*(-1)	end
	if inLaplacianBounds(gi+0, gj-1, xImage) then	p = p + (-1)*(-1)	end
	
	if gi == 0 and gj == 0 then
		printf("p=%f\n", p)
	end
	
	
	return p
end

local terra cost(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType())
	
	var x = self.X(i, j)
	var a = self.A(i, j)

	var v2 = x - a
	var e_fit = v2 * v2
	
	var v = laplacian(i, j, gi, gj, self.X)
	var e_reg = v * v

	var res = (float)(w_fit*e_fit + w_reg*e_reg)
	return res
	
end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
local terra gradient(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType(), outPre : &float)
	
	var x = self.X(i, j)
	var a = self.A(i, j)
	
	var e_fit = 2 * (x - a)

	var e_reg = 
		4*laplacian(i, j, gi, gj, self.X)
		-laplacian(i + 1, j, gi + 1, gj, self.X)
		-laplacian(i - 1, j, gi - 1, gj, self.X)
		-laplacian(i, j + 1, gi, gj + 1, self.X)
		-laplacian(i, j - 1, gi, gj - 1, self.X)
	e_reg = 2.0*e_reg

	
	if outPre ~= nil then
		--pre-conditioner
		var p_fit : float = 2.0	
		var p_reg : float = 2.0 * laplacianPreconditioner(gi, gj, self.X)
		
		var pre = w_fit*p_fit + w_reg*p_reg
		if pre > 0.0001 then
			pre = 1.0 / pre
			if pre > 1 and gi < 10 and gj < 10 then
				--printf("pre=%f (%d|%d)\t", pre, gi, gj)
			end
		else 
			pre = 1.0
		end
		
		var expected : float = 1.0f / (2.0f*w_fit + 2.0f*20.0f*w_reg)
		--if 	gi == 0 and gj == 0 or
		--	gi == 0 and gj == self.X:H()-1 or
		--	gi == self.X:W()-1 and gj == 0 or
		--	gi == self.X:W()-1 and gj == self.X:H()-1		then		
		if gi == 0 or gi == self.X:W()-1 or gj == 0 or gj == self.X:H()-1 then
			--pre = expected
			--printf("reg=%f; pre=%f\n; exp=%f\n",e_reg,pre,expected)
		end		
		--pre = expected
		--pre = 1.0f
		@outPre = pre
	end
	
	
	return w_fit*e_fit + w_reg*e_reg
	
end

-- eval 2*JtJ (note that we keep the '2' to make it consistent with the gradient
local terra applyJTJ(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType(), pImage : P:UnknownType())
 
	--fit
	var e_fit = 2.0f*pImage(i, j)
	
	--reg
	var e_reg = 
		4*laplacian(i + 0, j + 0, gi + 0, gj + 0, pImage)
		-laplacian(i + 1, j + 0, gi + 1, gj + 0, pImage)
		-laplacian(i - 1, j + 0, gi - 1, gj + 0, pImage)
		-laplacian(i + 0, j + 1, gi + 0, gj + 1, pImage)
		-laplacian(i + 0, j - 1, gi + 0, gj - 1, pImage)
	e_reg = 2.0*e_reg
	
	return w_fit*e_fit + w_reg*e_reg
end

P:Function("cost", {W,H}, cost)
P:Function("gradient", {W,H}, gradient)
P:Function("applyJTJ", {W,H}, applyJTJ)


return P