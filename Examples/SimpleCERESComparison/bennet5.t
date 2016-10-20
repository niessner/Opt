local OPT_FLOAT2 = float2
local OPT_FLOAT3 = float3

local N,U = opt.Dim("N",0), opt.Dim("U",1)
local funcParams =   Unknown("funcParams", OPT_FLOAT3, {U}, 0) -- a,b,c
local data =         Image("data", OPT_FLOAT2, {N}, 1) -- a,b

local G = Graph("G", 2,
                "d", {N}, 3, 
				"abc", {U}, 5)

UsePreconditioner(true)

local d = data(G.d)
local abc = funcParams(G.abc)
local x_i = d(0)
local y_i = d(1)
local a = abc(0)
local b = abc(1)
local c = abc(2)

--Energy(y_i - (a*cos(b*x_i)+b*sin(a*x_i))) -- works
--Energy(y_i - ad.cos(x_i)) -- works
Energy(y_i - a * ad.pow(b + x_i, -1.0 / c))
--Energy(y_i - a*(1.0 - ad.exp(-b*x_i))) -- doesn't work

-- Hack to get example to work with no image domain energy
local zero = 0.0
local zeroIm = ComputedImage("zero",{U},zero)
Energy(zeroIm(0)*(funcParams(0)(0) - funcParams(0)(1)))