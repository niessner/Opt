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


local terra inLaplacianBounds(i : int64, j : int64, xImage : P:UnknownType())
	return i > 0 and i < xImage:W() - 1 and j > 0 and j < xImage:H() - 1
end

local terra laplacian(i : int64, j : int64, gi : int64, gj : int64, xImage : P:UnknownType())
	if not inLaplacianBounds(gi, gj, xImage) then
		return 0
	end

	var x = xImage(i, j)
	var n0 = xImage(i - 1, j)
    var n1 = xImage(i + 1, j)
    var n2 = xImage(i, j - 1)
    var n3 = xImage(i, j + 1)

	var v = 4 * x - (n0 + n1 + n2 + n3)
	--IO.printf("laplacian (%d,%d) = %f\n", i, j, v)
	return v
end

local terra cost(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType())
	
	var x = self.X(i, j)
	var a = self.A(i, j)

	var v = laplacian(i, j, gi, gj, self.X)
	var laplacianCost = v * v

	var v2 = x - a
	var reconstructionCost = v2 * v2

	var res = (float)(w_reg*laplacianCost + w_fit*reconstructionCost)
	return res
	
end

local terra gradient(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType())
	
	var x = self.X(i, j)
	var a = self.A(i, j)
	
	var reconstructionGradient = 2 * (x - a)

	var laplacianGradient = 
		4*laplacian(i, j, gi, gj, self.X)
		-laplacian(i + 1, j, gi + 1, gj, self.X)
		-laplacian(i - 1, j, gi - 1, gj, self.X)
		-laplacian(i, j + 1, gi, gj + 1, self.X)
		-laplacian(i, j - 1, gi, gj - 1, self.X)
	laplacianGradient = 2.0*laplacianGradient

	return w_reg*laplacianGradient + w_fit*reconstructionGradient


end

local terra gradientPreconditioner(i : int64, j : int64)
	return w_reg*24.0f + w_fit*2.0f
end

-- eval 2*JTJ (note that we keep the '2' to make it consistent with the gradient
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

---------------
---------------
---------------

local terra inLaplacianBounds_global(i : int64, j : int64, xImage : P:UnknownType(false))
	return i > 0 and i < xImage:W() - 1 and j > 0 and j < xImage:H() - 1
end

local terra laplacian_global(i : int64, j : int64, gi : int64, gj : int64, xImage : P:UnknownType(false))
	if not inLaplacianBounds_global(gi, gj, xImage) then
		return 0
	end

	var x = xImage(i, j)
	var n0 = xImage(i - 1, j)
    var n1 = xImage(i + 1, j)
    var n2 = xImage(i, j - 1)
    var n3 = xImage(i, j + 1)

	var v = 4 * x - (n0 + n1 + n2 + n3)
	--IO.printf("laplacian (%d,%d) = %f\n", i, j, v)
	return v
end

local terra cost_global(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType(false))
	
	var x = self.X(i, j)
	var a = self.A(i, j)

	var v = laplacian_global(i, j, gi, gj, self.X)
	var laplacianCost = v * v

	var v2 = x - a
	var reconstructionCost = v2 * v2

	var res = (float)(w_reg*laplacianCost + w_fit*reconstructionCost)
	return res
	
end

local terra gradient_global(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType(false))
	
	var x = self.X(i, j)
	var a = self.A(i, j)
	
	var reconstructionGradient = 2 * (x - a)

	var laplacianGradient = 
		4*laplacian_global(i, j, gi, gj, self.X)
		-laplacian_global(i + 1, j, gi + 1, gj, self.X)
		-laplacian_global(i - 1, j, gi - 1, gj, self.X)
		-laplacian_global(i, j + 1, gi, gj + 1, self.X)
		-laplacian_global(i, j - 1, gi, gj - 1, self.X)
	laplacianGradient = 2.0*laplacianGradient

	return w_reg*laplacianGradient + w_fit*reconstructionGradient


end

local terra gradientPreconditioner(i : int64, j : int64)
	return w_reg*24.0f + w_fit*2.0f
end

-- eval 2*JTJ (note that we keep the '2' to make it consistent with the gradient
local terra applyJTJ_global(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType(false), pImage : P:UnknownType(false))
 
	var e_fit = 2.0f*pImage(i, j)
	
	--reg
	var e_reg = 
		4*laplacian_global(i + 0, j + 0, gi + 0, gj + 0, pImage)
		-laplacian_global(i + 1, j + 0, gi + 1, gj + 0, pImage)
		-laplacian_global(i - 1, j + 0, gi - 1, gj + 0, pImage)
		-laplacian_global(i + 0, j + 1, gi + 0, gj + 1, pImage)
		-laplacian_global(i + 0, j - 1, gi + 0, gj - 1, pImage)
	e_reg = 2.0 * e_reg
	
	return w_fit*e_fit + w_reg*e_reg
end


P:Function("cost", {W,H}, cost)
P:Function("gradient", {W,H}, gradient)
P:Function("applyJTJ", {W,H}, applyJTJ)
P:Function("cost_global", {W,H}, cost_global)
P:Function("gradient_global", {W,H}, gradient_global)
P:Function("applyJTJ_global", {W,H}, applyJTJ_global)
P.gradientPreconditioner = gradientPreconditioner

return P