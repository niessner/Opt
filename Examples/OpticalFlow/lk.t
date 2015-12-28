local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()

local X = S:Image("X", opt.float2,W,H,0)

local I = S:Image("I",opt.float3,W,H,1)

local I_hat_im = S:Image("I_hat",opt.float3,W,H,2)
local I_hat_dx = S:Image("I_hat_dx",opt.float3,W,H,3)
local I_hat_dy = S:Image("I_hat_dy",opt.float3,W,H,4)
local I_hat = ad.sampledimage(I_hat_im,I_hat_dx,I_hat_dy)

local i,j = W:index(),H:index()

local terms = terralib.newlist()

terms:insert(I(0,0) - I_hat(i + X(0,0,0),j + X(0,0,1)))

local offsets = { {1,0}, {-1,0}, {0,1}, {0,-1} }
for ii,o in ipairs(offsets) do
    local nx,ny = unpack(o)
    local a = I(nx,ny) - I(0,0)
    local b = I_hat(i + X(0,0,0),j + X(0,0,1)) - I_hat(i + nx + X(nx,ny,0),j + ny + X(nx,ny,1))
    terms:insert(a - b)
end

return S:Cost(ad.sumsquared(unpack(terms)))