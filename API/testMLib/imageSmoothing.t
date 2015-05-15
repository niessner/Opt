local IO = terralib.includec("stdio.h")

local P = opt.ProblemSpec()
local W = opt.Dim("W",0)
local H = opt.Dim("H",1)

P:Image("X",float,W,H,0)
P:Image("A",float,W,H,1)

local C = terralib.includecstring [[
#include <math.h>
]]

-- e = sum_{i,j} [ e_reg_{i,j}^2 + e_fit_{i,j}^2 ]
-- e_reg_{i,j} = I(i,j) - 0.25 * [ I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1) ]
-- e_fit_{i,j} = || I(i,j) - \hat{I}(i,j) ||_2


-- TODO: this should be factored into a parameter
local w_fit = 0.1
local w_reg = 1.0

local terra inLaplacianBounds(i : uint64, j : uint64, xImage : P:UnknownType())
	return i > 0 and i < xImage:W() - 1 and j > 0 and j < xImage:H() - 1
end

local terra inImageBounds(i : uint64, j : uint64, xImage : P:UnknownType())
	return i >= 0 and i <= xImage:W() - 1 and j >= 0 and j <= xImage:H() - 1
end

local terra laplacian(i : uint64, j : uint64, xImage : P:UnknownType())
	if not inLaplacianBounds(i, j, xImage) then
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

local terra cost(i : uint64, j : uint64, self : P:ParameterType())
	
	var x = self.X(i, j)
	var a = self.A(i, j)

	var v = laplacian(i, j, self.X)
	var laplacianCost = v * v

	var v2 = x - a
	var reconstructionCost = v2 * v2

	return (float)(w_reg*laplacianCost + w_fit*reconstructionCost)
end

local terra gradient(i : uint64, j : uint64, self : P:ParameterType())
	var x = self.X(i, j)
	var a = self.A(i, j)

	var reconstructionGradient = 2 * (x - a)

	var laplacianGradient = 4*laplacian(i, j, self.X)-laplacian(i + 1, j, self.X)-laplacian(i - 1, j, self.X)-laplacian(i, j + 1, self.X)-laplacian(i, j - 1, self.X);
	laplacianGradient = 2.0*laplacianGradient;

	return w_reg*laplacianGradient + w_fit*reconstructionGradient
end

local terra gradientPreconditioner(i : uint64, j : uint64)
	return w_reg*24.0f + w_fit*2.0f
end

-- eval 2*JTJ (note that we keep the '2' to make it consistent with the gradient
local terra applyJTJ(i : uint64, j : uint64, self : P:ParameterType(), pImage : P:UnknownType())
 
	var e_fit = 2.0f*pImage(i, j)
	
	--reg
	var e_reg = 4*laplacian(i + 0, j + 0, pImage)-laplacian(i + 1, j + 0, pImage)-laplacian(i - 1, j, pImage)-laplacian(i, j + 1, pImage)-laplacian(i, j - 1, pImage)
	e_reg = 2.0 * e_reg;
	
	return w_fit*e_fit + w_reg*e_reg
end

P:Function("cost", {W,H}, {1,1}, cost)
P:Function("gradient", {W,H}, {2,2}, gradient)
P:Function("applyJTJ", {W,H}, {2,2}, applyJTJ)
P.gradientPreconditioner = gradientPreconditioner

return P