require("helper")
local W,H = Dim("W",0), Dim("H",1)
local X = Array2D("X", opt.float2,W,H,0)
local I = Array2D("I",float,W,H,1)
local I_hat_im = Array2D("I_hat",float,W,H,2)
local I_hat_dx = Array2D("I_hat_dx",float,W,H,3)
local I_hat_dy = Array2D("I_hat_dy",float,W,H,4)
local I_hat = SampledImage(I_hat_im,I_hat_dx,I_hat_dy)

local i,j = W:index(),H:index()
UsePreconditioner(false)
local w_fitSqrt = Param("w_fit", float, 0)
local w_regSqrt = Param("w_reg", float, 1)

local e_fit = w_fitSqrt*(I(0,0) - I_hat(i + X(0,0,0),j + X(0,0,1)))
Energy(e_fit)

for nx,ny in Stencil { {1,0}, {-1,0}, {0,1}, {0,-1} } do
	local e_reg = w_regSqrt*(X(0,0) - X(nx,ny))
    Energy(Select(InBounds(nx,ny),e_reg,0))
end

return Result() --P:Cost(ad.sumsquared(unpack(terms)))