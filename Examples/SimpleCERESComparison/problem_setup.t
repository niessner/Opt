require("opt_precision")
S = ad.ProblemSpec()

function sqr(x)
    return x*x
end

x = 0
y = 0
b1 = 0
b2 = 0
b3 = 0
b4 = 0
b5 = 0
b6 = 0
b7 = 0
b8 = 0
b9 = 0
terms = terralib.newlist()
function problemsetup(UNKNOWN_COUNT, DATA_DIMENSION)
    local OPT_FLOAT = {}
    if OPT_DOUBLE_PRECISION then
        OPT_FLOAT[2] = opt.double2
        OPT_FLOAT[3] = opt.double3
        OPT_FLOAT[4] = opt.double4
        OPT_FLOAT[5] = opt.double5
        OPT_FLOAT[6] = opt.double6
        OPT_FLOAT[7] = opt.double7
        OPT_FLOAT[8] = opt.double8
        OPT_FLOAT[9] = opt.double9
    else
        OPT_FLOAT[2] = opt.float2
        OPT_FLOAT[3] = opt.float3
        OPT_FLOAT[4] = opt.float4
        OPT_FLOAT[5] = opt.float5
        OPT_FLOAT[6] = opt.float6
        OPT_FLOAT[7] = opt.float7
        OPT_FLOAT[8] = opt.float8
        OPT_FLOAT[9] = opt.float9
    end

    local N,U = opt.Dim("N",0), opt.Dim("U",1)
    local funcParams =   S:Unknown("funcParams", OPT_FLOAT[UNKNOWN_COUNT], {U}, 0)
    local data =         S:Image("data", OPT_FLOAT[DATA_DIMENSION], {N}, 1)

    local G = S:Graph("G", 2,
                    "d", {N}, 3, 
                    "b", {U}, 5)

    S:UsePreconditioner(true)

    local d = data(G.d)
    local b = funcParams(G.b)
    x = d(0)
    y = d(1)
    b1 = b(0)
    b2 = b(1)
    if UNKNOWN_COUNT > 2 then
        b3 = b(2)
    end
    if UNKNOWN_COUNT > 3 then
        b4 = b(3)
    end
    if UNKNOWN_COUNT > 4 then
        b5 = b(4)
    end
    if UNKNOWN_COUNT > 5 then
        b6 = b(5)
    end
    if UNKNOWN_COUNT > 6 then
        b7 = b(6)
    end
    if UNKNOWN_COUNT > 7 then
        b8 = b(7)
    end
    if UNKNOWN_COUNT > 8 then
        b9 = b(8)
    end
    
    -- Hack to get example to work with no image domain energy
    local zero = 0.0
    local zeroIm = S:ComputedImage("zero",{U},zero)
    local sumOfParams = funcParams(0)(0)
    for i=1,UNKNOWN_COUNT-1 do
        sumOfParams = sumOfParams + funcParams(0)(i)
    end
    local hack = zeroIm(0)*sumOfParams

    terms:insert(hack)
end