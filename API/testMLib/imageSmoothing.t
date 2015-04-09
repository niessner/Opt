local IO = terralib.includec("stdio.h")

local W = opt.Dim("W")
local H = opt.Dim("H")

local X = opt.Image(double,W,H)

local A = opt.Image(double,W,H)

-- TODO: this should be factored into a parameter
local w = 0.1

local terra laplacian(i : uint64, j : uint64, xImage : X)
	var x = xImage(i, j)
	var n0 = xImage(i - 1, j)
    var n1 = xImage(i + 1, j)
    var n2 = xImage(i, j - 1)
    var n3 = xImage(i, j + 1)

	return 4 * x - (n0 + n1 + n2 + n3)
end

local terra cost(i : uint64, j : uint64, xImage : X, aImage : A)
    var x = xImage(i, j)
	var a = aImage(i, j)

	var laplacianCost = 0.0

	if i > 0 and i < aImage.W - 1 and j > 0 and j < aImage.H - 1 then
		var v = laplacian(i, j, xImage)
		laplacianCost = v * v
	end

	var v2 = x - a
	var reconstructionCost = w * v2 * v2

	return laplacianCost + reconstructionCost
end

local terra gradient(i : uint64, j : uint64, xImage : X, aImage : A)
	var x = xImage(i, j)
	var a = aImage(i, j)

	var laplacianGradient = 0.0

	if i > 0 and i < aImage.W - 1 and j > 0 and j < aImage.H - 1 then
		var v = laplacian(i - 1, j, xImage)
		laplacianGradient = laplacianGradient + 8 * v * x
	end

	if j > 0 and j < aImage.H - 1 then
		if i > 0 then
			var v = laplacian(i - 1, j, xImage)
			laplacianGradient = laplacianGradient + -2 * v
		end

		if i < aImage.width - 1 then
			var v = laplacian(i + 1, j, xImage)
			laplacianGradient = laplacianGradient + -2 * v
		end
	end

	if i > 0 and i < aImage.W - 1 then
		if j > 0 then
			var v = laplacian(i, j - 1, xImage)
			laplacianGradient = laplacianGradient + -2 * v
		end

		if j < aImage.height - 1 then
			var v = laplacian(i, j + 1, xImage)
			laplacianGradient = laplacianGradient + -2 * v
		end
	end

	var v2 = x - a
	var reconstructionGradient = w * 2 * v2

	return laplacianGradient + reconstructionGradient
end

return { dims = { W, H },
         cost = { dim = {W,H}, fn = cost },
         gradient = gradient }
