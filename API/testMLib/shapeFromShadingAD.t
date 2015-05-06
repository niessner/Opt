local W,H = opt.Dim("W",0), opt.Dim("H",1)
local D = ad.Image("D",W,H,0) -- Refined Depth
local D_i = ad.Image("D_i",W,H,1) -- Depth input
local I = ad.Image("I",W,H,2) -- Color input
--local k = ad.Image("k",W,H,2) --Albedo image
local P = opt.InBounds(1,1)

local util = require("util")

local posX = W:index()
local posY = H:index()

-- TODO: accept these 4 as input uniforms?
local f_x = 532.569458
local f_y = 531.541077
local u_x = 320.0
local u_y = 240.0

-- Lighting coefficients
local L = {
	1.0,
	1.0,
	1.0,
	1.0,
	1.0,
	1.0,
	1.0,
	1.0
}

function sqMagnitude(point) 
	return 
point[1]*point[1] + point[2]*point[2] + 
point[3]*point[3]
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

function n(offX, offY)
	local i = offX + posX
    local j = offY + posY

    local n_x = D(offX, offY - 1) * (D(offX, offY) - D(offX - 1, offY)) / f_y
    local n_y = D(offX - 1, offY) * (D(offX, offY) - D(offX, offY - 1)) / f_x
    local n_z = (n_x * (u_x - i) / f_x) + (n_y * (u_y - j) / f_y) - (D(offX-1, offY)*D(off) / f_x*f_y)
    return {n_x, n_y, n_z}
end

function B(offX, offY)
	local normal = n(offX, offY)
	local n_x = normal[1]
	local n_y = normal[2]
	local n_z = normal[3]

	local lighting = L[1] +
					 L[2]*n_y + L[3]*n_z + L[4]*n_x  +
					 L[5]*n_x*n_y + L[6]*n_y*n_z + L[7]*(-n_x*n_x - n_y*n_y + 2*n_z*n_z) + L[8]*n_z*n_x + L[9]*(n_x*n_x-n*y*n_y)

	return k(offX, offY)*lighting
end


local w_g = 1.0
local w_s = 400.0 --1000.0
local w_p = 10.0 
local w_r = 100.0


local E_g_h = B(0,0) - B(1,0) - (I(0,0) - I(1,0))
local E_g_v = B(0,0) - B(0,1) - (I(0,0) - I(0,1))


local crossStencilValid = ad.greater(D_i(0,0)+D_i(1,0)+D_i(0,1)+D_i(-1,0)+D_i(0,-1), 0.0)
local pointValid = ad.greater(D_i(0,0), 0.0)

--local E_s = p(0,0) - 0.25*(p(-1,0) + p(0,-1) + p(1, 0) + p(0,1)) 
local E_s_beforeSelect = sqMagnitude(plus(p(0,0), times(-0.25, plus(plus(p(-1,0), p(0,-1)),plus(p(1, 0), p(0,1))))))
local E_s = ad.select(P,ad.select(crossStencilValid, E_s_beforeSelect ,0),0)

local E_p = ad.select(pointValid, D(0,0) - D_i(0,0)	,0)

local E_r = 0 --temporal constraint, unimplemented

local cost = ad.sumsquared(w_g*E_g_h, w_g*E_g_v, w_s*E_s, w_p*E_p, w_r*E_r)

return ad.Cost(cost)
