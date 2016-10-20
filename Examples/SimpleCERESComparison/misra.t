local OPT_FLOAT2 = float2

local N,U = opt.Dim("N",0), opt.Dim("U",1)
local funcParams =   Unknown("funcParams", OPT_FLOAT2, {U}, 0) -- a,b
local data =         Image("data", OPT_FLOAT2, {N}, 1) -- a,b

local G = Graph("G", 2,
                "d", {N}, 3, 
				"ab", {U}, 5)

UsePreconditioner(true)

local d = data(G.d)
local ab = funcParams(G.ab)
local x_i = d(0)
local y_i = d(1)
local a = ab(0)
local b = ab(1)

--Energy(y_i - (a*cos(b*x_i)+b*sin(a*x_i))) -- works
--Energy(y_i - ad.cos(x_i)) -- works
--Energy(y_i - ad.exp(x_i)) -- doesn't work
Energy(y_i - a*(1.0 - ad.pow(2.7182818284590452353603, -b*x_i))) -- doesn't work

-- Hack to get example to work with no image domain energy
local zero = 0.0
local zeroIm = ComputedImage("zero",{U},zero)
Energy(zeroIm(0)*(funcParams(0)(0) - funcParams(0)(1)))