require("opt_precision")
OPT_FLOAT = {}
if OPT_DOUBLE_PRECISION then
    OPT_FLOAT[2] = double2
    OPT_FLOAT[3] = double3
    OPT_FLOAT[4] = double4
    OPT_FLOAT[5] = double5
    OPT_FLOAT[6] = double6
    OPT_FLOAT[7] = double7
    OPT_FLOAT[8] = double8
    OPT_FLOAT[9] = double9
else
    OPT_FLOAT[2] = float2
    OPT_FLOAT[3] = float3
    OPT_FLOAT[4] = float4
    OPT_FLOAT[5] = float5
    OPT_FLOAT[6] = float6
    OPT_FLOAT[7] = float7
    OPT_FLOAT[8] = float8
    OPT_FLOAT[9] = float9
end

if not DATA_DIMENSION then
    DATA_DIMENSION = 2
end

-- Useful constant
pi = 3.141592653589793238462643383279

function sqr(x)
    return x*x
end

local N,U = opt.Dim("N",0), opt.Dim("U",1)
local funcParams = Unknown("b", OPT_FLOAT[UNKNOWN_COUNT], {U}, 0) 
local data =         Image("data", OPT_FLOAT[DATA_DIMENSION], {N}, 1) -- a,b

local G = Graph("G", 2,
                "d", {N}, 3, 
				"b", {U}, 5)

UsePreconditioner(true)

local d = data(G.d)
b = funcParams(G.b)
x = d(0)
y = d(1)

if DATA_DIMENSION >= 3 then
    y  = d(0)
    x1 = d(1)
    x2 = d(2)
    if (DATA_DIMENSION) >= 4 then
        x3 = d(3)
    end
end

b1 = b(0)
b2 = b(1)
b3 = b(2)
b4 = b(3)
b5 = b(4)
b6 = b(5)
b7 = b(6)
b8 = b(7)
b9 = b(8)
