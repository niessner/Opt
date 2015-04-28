local IO = terralib.includec("stdio.h")

local W = opt.Dim("W",0)
local H = opt.Dim("H",1)

local X = opt.Image(float,W,H,0)
local A = opt.Image(float,W,H,1)

local C = terralib.includecstring [[
#include <math.h>
]]

-- e = sum_{i,j} [ e_reg_{i,j}^2 + e_fit_{i,j}^2 ]
-- e_reg_{i,j} = I(i,j) - 0.25 * [ I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1) ]
-- e_fit_{i,j} = || I(i,j) - \hat{I}(i,j) ||_2


-- TODO: this should be factored into a parameter
local w_fit = 0.1
local w_reg = 1.0

local terra inLaplacianBounds(i : uint64, j : uint64, xImage : X)
	return i > 0 and i < xImage:W() - 1 and j > 0 and j < xImage:H() - 1
end

local terra inImageBounds(i : uint64, j : uint64, xImage : X)
	return i >= 0 and i <= xImage:W() - 1 and j >= 0 and j <= xImage:H() - 1
end

local terra laplacian(i : uint64, j : uint64, xImage : X)
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

local terra cost(i : uint64, j : uint64, xImage : X, aImage : A)
	var x = xImage(i, j)
	var a = aImage(i, j)

	--IO.printf("image (%f,%f)\n", x, a)
	--IO.getchar()
    
	var v = laplacian(i, j, xImage)
	var laplacianCost = v * v

	var v2 = x - a
	var reconstructionCost = v2 * v2

	return (float)(w_reg*laplacianCost + w_fit*reconstructionCost)
end

local terra gradient(i : uint64, j : uint64, xImage : X, aImage : A)
	var x = xImage(i, j)
	var a = aImage(i, j)

	var laplacianGradient = 0.0f

	laplacianGradient = laplacianGradient + 8 * laplacian(i, j, xImage)

	laplacianGradient = laplacianGradient + -2 * laplacian(i + 1, j, xImage)
	laplacianGradient = laplacianGradient + -2 * laplacian(i - 1, j, xImage)
	laplacianGradient = laplacianGradient + -2 * laplacian(i, j + 1, xImage)
	laplacianGradient = laplacianGradient + -2 * laplacian(i, j - 1, xImage)

	var reconstructionGradient = 2 * (x - a)

	return (float)(w_reg*laplacianGradient + w_fit*reconstructionGradient)
end

local terra gradientPreconditioner(i : uint64, j : uint64)
	return w_reg*24.0f + w_fit*2.0f
end

-- eval 2*JTJ (note that we keep the '2' to make it consistent with the gradient
local terra applyJTJ(i : uint64, j : uint64, pImage : X, xImage : X, aImage : A)
 
	--fit
	var e_fit = 2.0f*pImage(i, j)
	
	--reg
	var e_reg = 4*laplacian(i + 0, j + 0, pImage)-laplacian(i + 1, j + 0, pImage)-laplacian(i - 1, j + 0, pImage)-laplacian(i + 0, j + 1, pImage)-laplacian(i + 0, j - 1, pImage)
	e_reg = 2.0 * e_reg;
	
	
	return w_fit*e_fit + w_reg*e_reg
end

return {
         cost = { dimensions = {W,H}, boundary = cost, interior = cost, stencil = {1,1} },
         gradient = { dimensions = {W,H}, boundary = gradient, interior = gradient, stencil = {2,2} },
		 applyJTJ = { dimensions = {W,H}, boundary = applyJTJ, interior = applyJTJ, stencil = {2,2} },
		 gradientPreconditioner = gradientPreconditioner }
