local IO = terralib.includec("stdio.h")

local W = opt.Dim("W", 0)
local H = opt.Dim("H", 1)

local X = opt.Image(float, W, H, 0)
local A = opt.Image(float, W, H, 1)

local imageAdj = opt.Adjacency( {W, H}, {W, H}, 0)

local weights = opt.EdgeValues (float, imageAdj, 0)

local C = terralib.includecstring [[
#include <math.h>
]]

local w_fit = 0.1
local w_reg = 1.0

local terra laplacian(i : uint64, j : uint64, xImage : X, w : weights, iAdj : imageAdj)

	var x = xImage(i, j)

	var sumValue = 0.0
	var sumWeights = 0.0
	for a in iAdj:neighbors(i, j) do
		sum = sum + X(a.x, a.y) * w(a)
		sumWeights = sumWeights + w(a)
	end

	--iAdj:count(i, j)
	var v = sumWeights * x - sum
	return v
end

local terra cost(i : uint64, j : uint64, xImage : X, aImage : A, w : weights, iAdj : imageAdj)
	var x = xImage(i, j)
	var a = aImage(i, j)

	var v = laplacian(i, j, xImage, w, iAdj)
	var laplacianCost = v * v

	var v2 = x - a
	var reconstructionCost = v2 * v2

	return (float)(w_reg*laplacianCost + w_fit*reconstructionCost)
end

return
{
	cost = { dimensions = {W, H}, boundary = cost, interior = cost, stencil = { uses = {{A,0,0},{X,0,0}}, indirections = {{imageAdj, { uses = {{X,0,0}}}, {weights} }} }
}
