local IO = terralib.includec("stdio.h")

local W = opt.Dim("W")
local H = opt.Dim("H")

local X = opt.Image(float,W,H)
local A = opt.Image(float,W,H)

local C = terralib.includecstring [[
#include <math.h>
]]

-- e = sum_{i,j} [ e_reg_{i,j}^2 + e_fit_{i,j}^2 ]
-- e_reg_{i,j} = I(i,j) - 0.25 * [ I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1) ]
-- e_fit_{i,j} = || I(i,j) - \hat{I}(i,j) ||_2


-- TODO: this should be factored into a parameter
local w = 0.1

local terra inLaplacianBounds(i : uint64, j : uint64, xImage : X)
	return i > 0 and i < xImage.W - 1 and j > 0 and j < xImage.H - 1
end

local terra inImageBounds(i : uint64, j : uint64, xImage : X)
	return i >= 0 and i <= xImage.W - 1 and j >= 0 and j <= xImage.H - 1
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
	var reconstructionCost = w * v2 * v2

	return (float)(laplacianCost + reconstructionCost)
end

local terra gradient(i : uint64, j : uint64, xImage : X, aImage : A)
	var x = xImage(i, j)
	var a = aImage(i, j)

	var laplacianGradient = 0.0

	laplacianGradient = laplacianGradient + 8 * laplacian(i, j, xImage)

	laplacianGradient = laplacianGradient + -2 * laplacian(i + 1, j, xImage)
	laplacianGradient = laplacianGradient + -2 * laplacian(i - 1, j, xImage)
	laplacianGradient = laplacianGradient + -2 * laplacian(i, j + 1, xImage)
	laplacianGradient = laplacianGradient + -2 * laplacian(i, j - 1, xImage)

	var reconstructionGradient = w * 2 * (x - a)

	return (float)(laplacianGradient + reconstructionGradient)
end

local terra gradientPreconditioner(i : uint64, j : uint64)
	return 24 + w * 2
end

return { dimensions = { W, H },
         cost = { dimensions = {W,H}, boundary = cost, interior = cost, stencil = {1,1} },
         gradient = { dimensions = {W,H}, boundary = gradient, interior = gradient, stencil = {2,2} },
		 gradientPreconditioner = gradientPreconditioner }
