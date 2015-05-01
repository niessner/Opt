local W,H = opt.Dim("W",0), opt.Dim("H",1)
local D = ad.Image("D",W,H,0) -- Refined Depth
local D_i = ad.Image("D_i",W,H,1) -- Depth input
local I = ad.Image("I",W,H,2) -- Color input
-- local B = ad.Image("B",W,H,2) --Rendered image
local P = opt.InBounds(1,1)

local posX = W:index()
local posY = H:index()

-- TODO: accept these 4 as input uniforms?
local f_x = 0.1
local f_y = 0.1
local u_x = 320
local u_y = 240

function magnitude(point) 
	return ad.sqrt(point[1]*point[1] + point[2]*point[2] + point[3]*point[3])
end

function times(number,point)
	return {number*point[1], number*point[2], number*point[3]}
end

function plus(p0,p1)
	return {p0[1]+p1[1], p0[2]+p1[2], p0[3]+p1[3]}
end

function p(offX,offY) 
    local d = D(0+offX,0+offY)
    local i = offX + posX
    local j = offY + posY
    local point = {((i-u_x)/f_x)*d, ((i-u_y)/f_y)*d, d}
    return point
end


local w_g = 1.0
local w_s = 1.0
local w_p = 1.0 
local w_r = 1.0

local E_g_h = 0 --B(0,0) - B(1,0) - (I(0,0) - I(1,0))
local E_g_v = 0 --B(0,0) - B(0,1) - (I(0,0) - I(0,1))

local crossStencilValid = ad.greater(D_i(0,0)+D_i(1,0)+D_i(0,1)+D_i(-1,0)+D_i(0,-1), 0.0)
local pointValid = ad.greater(D_i(0,0), 0.0)

--local E_s = p(0,0) - 0.25*(p(-1,0) + p(0,-1) + p(1, 0) + p(0,1))

local E_s = ad.select(crossStencilValid, magnitude(plus(p(0,0), times(-0.25, plus(plus(p(-1,0), p(0,-1)),plus(p(1, 0), p(0,1))))))	,0)
local E_p = ad.select(pointValid, D(0,0) - D_i(0,0)																					,0)

local E_r = 0 --temporal constraint, unimplemented

local cost = ad.sumsquared(w_g*E_g_h, w_g*E_g_v, w_s*E_s, w_p*E_p, w_r*E_r)

return ad.Cost(cost)
