W,H = opt.Dim("W",0), opt.Dim("H",1)
S = ad.ProblemSpec()

function L_2_norm(v)
    -- TODO: check if scalar and just return
    return ad.sqrt(v:dot(v))
end
local L_p_counter = 1
function L_p(diff, diff_const, p, S)
    local dist_const = L_2_norm(diff_const)
    local eps = 0.0000001
    local C = ad.pow(dist_const+eps,(p-2))
    local sqrtC = ad.sqrt(C)
    local sqrtCImage = S:ComputedImage("sqrtC_"..tostring(L_p_counter),{W,H},sqrtC)
    L_p_counter = L_p_counter + 1
    return sqrtCImage(0,0)*diff
end