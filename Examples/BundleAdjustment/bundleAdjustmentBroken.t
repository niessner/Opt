
local cameraCount = Dim("cameraCount", 0)
local correspondenceCount = Dim("correspondenceCount", 1)

--local anchorWeight = Param("anchorWeight", float, XX)

local cameras =         Unknown("cameras", float6, {cameraCount}, 0) -- translation.xyz, rotation.xyz
local correspondences =	Image("correspondences", float6, {correspondenceCount}, 1) -- ptA.xyz, ptB.xyz
local anchorWeights =   Image("anchorWeights", float, {cameraCount}, 2)

local G = Graph("G", 3,
                "cameraA", {cameraCount}, 4, 
                "cameraB", {cameraCount}, 5,
				"correspondence", {correspondenceCount}, 6)

local function cross(ptA, ptB)
	return Vector( ptA(1) * ptB(2) - ptA(2) * ptB(1),
				   ptA(2) * ptB(0) - ptA(0) * ptB(2),
				   ptA(0) * ptB(1) - ptA(1) * ptB(0) )
end

local function breakCodeFloat(a1, a2, b0)
	
	local theta2 = a1
	local theta = sqrt(theta2)
	local thetaInverse = 1.0 / theta

	local crossWPt = a1 * thetaInverse - thetaInverse
	
	return Select(greatereq(theta2, 1e-6), crossWPt, b0)
end

UsePreconditioner(true)

local cameraA = cameras(G.cameraA)
local valueA = breakCodeFloat(cameraA(1), cameraA(2), cameraA(3))
Energy(valueA)

--
-- anchor energy
--
Energy(cameras(0))

