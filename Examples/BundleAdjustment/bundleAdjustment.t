
local cameraCount = Dim("cameraCount", 0)
local correspondenceCount = Dim("correspondenceCount", 1)

local unknownType = double6
local weightType = float

local cameras =         Unknown("cameras", unknownType, {cameraCount}, 0) -- translation.xyz, rotation.xyz
local correspondences =	Image("correspondences", unknownType, {correspondenceCount}, 1) -- ptA.xyz, ptB.xyz
local anchorWeights =   Image("anchorWeights", weightType, {cameraCount}, 2)

local G = Graph("G", 3,
                "cameraA", {cameraCount}, 4, 
                "cameraB", {cameraCount}, 5,
				"correspondence", {correspondenceCount}, 6)

local function cross(ptA, ptB)
	return Vector( ptA(1) * ptB(2) - ptA(2) * ptB(1),
				   ptA(2) * ptB(0) - ptA(0) * ptB(2),
				   ptA(0) * ptB(1) - ptA(1) * ptB(0) )
end

local function rotatePoint(axisAngle, pt)
	local theta2 = axisAngle:dot(axisAngle)
	
	local theta = sqrt(theta2)
	local cosTheta = cos(theta)
	local sinTheta = sin(theta)
	local thetaInverse = 1.0 / theta

	local w = axisAngle * thetaInverse
	local crossWPt = cross(w, pt)
	local tmp = w:dot(pt) * (1.0 - cosTheta)

	local full = pt * cosTheta + crossWPt * sinTheta + w * tmp
	local simple = pt + cross(axisAngle, pt)

	return Select(greatereq(theta2, 1e-6),full,simple)
end

local function apply(camera, pt)
	local world = rotatePoint(Vector(camera(0), camera(1), camera(2)), pt)
	return world + Vector(camera(3), camera(4), camera(5))
end

UsePreconditioner(true)

--
-- correspondence energy
--
local cameraA = cameras(G.cameraA)
local cameraB = cameras(G.cameraB)
local correspondence = correspondences(G.correspondence)

local worldA = apply(cameraA, Vector(correspondence(0), correspondence(1), correspondence(2)))
local worldB = apply(cameraB, Vector(correspondence(3), correspondence(4), correspondence(5)))

Energy(worldA - worldB)

--
-- anchor energy
--
local anchorWeight = anchorWeights(0)
Energy(cameras(0) * anchorWeight)
