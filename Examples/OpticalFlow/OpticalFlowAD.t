local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()

local X = S:Image("X", opt.float2,W,H,0)

local I = S:Image("I",float,W,H,1)

local I_hat_im = S:Image("I_hat",float,W,H,2)
local I_hat_dx = S:Image("I_hat_dx",float,W,H,3)
local I_hat_dy = S:Image("I_hat_dy",float,W,H,4)
local I_hat = ad.sampledimage(I_hat_im,I_hat_dx,I_hat_dy)

local i,j = W:index(),H:index()

S:UsePreconditioner(false)

local C = terralib.includecstring [[
#include <math.h>
]]

--local w_fitSqrt = S:Param("w_fit", float, 0)
--local w_regSqrt = S:Param("w_reg", float, 1)

local w_fitSqrt = 1.0
local w_regSqrt = 1.0


local terms = terralib.newlist()

local e_fit = w_fitSqrt*(I(0,0) - I_hat(i + X(0,0,0),j + X(0,0,1)))
terms:insert(e_fit)

--[[
local offsets = { {1,0}, {-1,0}, {0,1}, {0,-1} }
for ii,o in ipairs(offsets) do
    local nx,ny = unpack(o)
    local a = I(nx,ny) - I(0,0)
    local b = I_hat(i + X(0,0,0),j + X(0,0,1)) - I_hat(i + nx + X(nx,ny,0),j + ny + X(nx,ny,1))
	local e_reg = w_regSqrt*(a - b)
    terms:insert(e_reg)
end
--]]


local offsets = { {1,0}, {-1,0}, {0,1}, {0,-1} }
for ii,o in ipairs(offsets) do
    local nx,ny = unpack(o)
    local l = X(0,0) - X(nx,ny)
	local e_reg = w_regSqrt*l
   -- terms:insert(e_reg)
end


return S:Cost(ad.sumsquared(unpack(terms)))