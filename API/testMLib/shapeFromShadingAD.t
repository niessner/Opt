local W,H 	= opt.Dim("W",0), opt.Dim("H",1)
local S 	= ad.ProblemSpec()
local D 	= S:Image("X",W,H,0) -- Refined Depth
local D_i 	= S:Image("Y",W,H,1) -- Depth input
local I 	= S:Image("I",W,H,2) -- Target Intensity
local D_p 	= S:Image("Z",W,H,3) -- Previous Depth
--local edgeMask 	= S:Image("edgeMask",W,H,3) -- Edge mask. Currently unused

local inBounds = opt.InBounds(3,3)

local epsilon = 0.00001

-- See TerraSolverParameters
local w_p						= S:Param("w_p",float,0)-- Is initialized by the solver!

local w_s		 				= S:Param("w_s",float,1)-- Regularization weight
local w_r						= S:Param("w_r",float,2)-- Prior weight
w_s = 1.0
local w_g						= S:Param("w_g",float,3)-- Shading weight
w_g = 1000.0
local weightShadingStart		= S:Param("weightShadingStart",float,4)-- Starting value for incremental relaxation
local weightShadingIncrement	= S:Param("weightShadingIncrement",float,5)-- Update factor

local weightBoundary			= S:Param("weightBoundary",float,6)-- Boundary weight

local f_x						= S:Param("f_x",float,7)
local f_y						= S:Param("f_y",float,8)
local u_x 						= S:Param("u_x",float,9)
local u_y 						= S:Param("u_y",float,10)
    
local offset = 10;
local deltaTransform = {}
for i=1,16 do
	deltaTransform[i] = S:Param("deltaTransform" .. i .. "",float,offset+i)
end
offset = offset + 16

local L = {}
for i=1,9 do
	L[i] = S:Param("L" .. i .. "",float,offset+i)
end
offset = offset + 9
local nNonLinearIterations 	= S:Param("nNonLinearIterations",uint,offset+1) -- Steps of the non-linear solver	
local nLinIterations 		= S:Param("nLinIterations",uint,offset+2) -- Steps of the linear solver
local nPatchIterations 		= S:Param("nPatchIterations",uint,offset+3) -- Steps on linear step on block level


local util = require("util")

local posX = W:index()
local posY = H:index()


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

-- equation 8
function p(offX,offY) 
    local d = D(offX,offY)
    local i = offX + posX
    local j = offY + posY
    local point = {((i-u_x)/f_x)*d, ((i-u_y)/f_y)*d, d}
    return point
end

-- equation 10
function n(offX, offY)
	local i = offX + posX -- good
    local j = offY + posY -- good
    --f_x good, f_y good

    local n_x = D(offX, offY - 1) * (D(offX, offY) - D(offX - 1, offY)) / f_y
    local n_y = D(offX - 1, offY) * (D(offX, offY) - D(offX, offY - 1)) / f_x
    local n_z = (n_x * (u_x - i) / f_x) + (n_y * (u_y - j) / f_y) - (D(offX-1, offY)*D(offX, offY-1) / f_x*f_y)
    local inverseMagnitude = 1.0/ad.sqrt(n_x*n_x + n_y*n_y + n_z*n_z + epsilon)
    return times(inverseMagnitude, {n_x, n_y, n_z})
    --return {inverseMagnitude,inverseMagnitude,inverseMagnitude}
end

function B(offX, offY)
	local normal = n(offX, offY)
	local n_x = normal[1]
	local n_y = normal[2]
	local n_z = normal[3]

	local lighting = L[1] +
					 L[2]*n_y + L[3]*n_z + L[4]*n_x  +
					 L[5]*n_x*n_y + L[6]*n_y*n_z + L[7]*(-n_x*n_x - n_y*n_y + 2*n_z*n_z) + L[8]*n_z*n_x + L[9]*(n_x*n_x-n_y*n_y)

	return 1.0*lighting -- replace 1.0 with estimated albedo in slower version
end

local squareStencilSum = 0
for i=-3,3 do
	for j=-3,3 do
		squareStencilSum = squareStencilSum + D_i(i,j)			
	end
end

local squareStencilValid = ad.greater(squareStencilSum, 0.0)

local crossStencilValid = ad.greater(D_i(0,0)+D_i(1,0)+D_i(0,1)+D_i(-1,0)+D_i(0,-1), 0.0)
local pointValid = ad.greater(D_i(0,0), 0.0)

local E_g_h_beforeSelect = B(0,0) - B(1,0) - (I(0,0) - I(1,0))
local E_g_h = ad.select(inBounds, ad.select(squareStencilValid, E_g_h_beforeSelect, 0), 0)

local E_g_v_beforeSelect = B(0,0) - B(0,1) - (I(0,0) - I(0,1))
local E_g_v = ad.select(inBounds, ad.select(squareStencilValid, E_g_v_beforeSelect, 0), 0)

local E_s_beforeSelect = sqMagnitude(plus(p(0,0), times(-0.25, plus(plus(p(-1,0), p(0,-1)),plus(p(1, 0), p(0,1))))))
local E_s = ad.select(inBounds,ad.select(crossStencilValid, E_s_beforeSelect ,0),0)

local E_p = ad.select(pointValid, D(0,0) - D_i(0,0),0)
--local E_p = ad.select(pointValid, D(0,0) - D_i(0,0)	,0)
--local E_p = ad.select(squareStencilValid, D(0,0) - squareStencilSum/10000.0,0)
--local E_p = ad.select(pointValid, D(0,0) - 0.9	, D(0,0) - 0.1)

--local E_p = ad.select(squareStencilValid, D(0,0) 	,0)

local E_r_h = 0 --temporal constraint, unimplemented
local E_r_v = 0 --temporal constraint, unimplemented
local E_r_d = 0 --temporal constraint, unimplemented

local cost = ad.sumsquared(w_g*E_g_h, w_g*E_g_v, w_s*E_s, w_p*E_p, w_r*E_r_h, w_r*E_r_v, w_r*E_r_d)
--local cost = ad.sumsquared(E_p)
return S:Cost(cost)
