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
-- note that this depends on whether g_i should be the gradient of r_i or r_i^2?
--

terra cost(i : uint64, j : uint64, xImage : X, aImage : A, bImage : B, cImage : C)
    double x = xImage(i, j)
	double a = aImage(i, j)
	double b = bImage(i, j)
	double c = cImage(i, j)
	IO.printf("the cost function")
	return a * x * x + b * x + c
end

terra gradient(i : uint64, j : uint64, xImage : X, aImage : A, bImage : B, cImage : C)
    double x = xImage(i, j)
	double a = aImage(i, j)
	double b = bImage(i, j)
	double c = cImage(i, j)
	IO.printf("the gradient function")
	return 2.0 * (a * x * x + b * x + c) * (2.0 * a * x + b);
end

return { dims = { W, H },
         cost = { dim = {W,1}, fn = cost },
         gradient = gradient }
