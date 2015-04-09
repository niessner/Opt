local IO = terralib.includec("stdio.h")

W = opt.Dim("W")
H = opt.Dim("H")

A = opt.Image(double,W,1)
B = opt.Image(double,W,1)
C = opt.Image(double,W,1)
X = opt.Image(double,W,1)

--
-- cost = sum_i (r_i^2)
-- r_i = a * x * x + b * x + c
-- g_i = 2(ax^2 + bx + c) * (2ax + b)
-- minimum when 2(ax^2 + bx + c) * (2ax + b) = 0, x = -b / 2a
-- note that this depends on whether g_i should be the gradient of r_i or r_i^2?
--

local terra cost(i : uint64, j : uint64, xImage : X, aImage : A, bImage : B, cImage : C)
	var x = xImage(i, j)
	var a = aImage(i, j)
	var b = bImage(i, j)
	var c = cImage(i, j)
	--IO.printf("image (%d,%d)\n", xImage.impl.stride, xImage.impl.elemsize)
	--IO.printf("cost (%d,%d): x=%f, a=%f, b=%f, c=%f\n", i, j, x, a, b, c)

	var v = a * x * x + b * x + c 
	return v * v
end

local terra gradient(i : uint64, j : uint64, xImage : X, aImage : A, bImage : B, cImage : C)
	var x = xImage(i, j)
	var a = aImage(i, j)
	var b = bImage(i, j)
	var c = cImage(i, j)
	return 2.0 * (a * x * x + b * x + c) * (2.0 * a * x + b);
end

return { dims = { W, H },
         cost = { dim = {W,1}, fn = cost },
         gradient = gradient }
