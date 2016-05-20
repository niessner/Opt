
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

local function breakCodeVec(axisAngle, pt)
	
	local theta2 = axisAngle:dot(axisAngle)
	local theta = sqrt(theta2)
	local thetaInverse = 1.0 / theta

	local w = axisAngle * thetaInverse
	local crossWPt = cross(w, pt)
	
	return Select(greatereq(theta2, 1e-6),crossWPt,pt)
end


UsePreconditioner(true)

local cameraA = cameras(G.cameraA)
local valueA = breakCodeVec( Vector(cameraA(0), cameraA(1), cameraA(2)), Vector(cameraA(3), cameraA(4), cameraA(5)))
Energy(valueA)

--
-- anchor energy
--
Energy(cameras(0))

local function breakCodeFloat(a0, a1, a2, b0, b1, b2)
	
	local theta2 = a0 * a0 + a1 * a1 + a2 * a2
	local theta = sqrt(theta2)
	local thetaInverse = 1.0 / theta

	local w = Vector(a0, a1, a2) * thetaInverse
	local crossWPt = cross(w, Vector(b0, b1, b2))
	
	return Select(greatereq(theta2, 1e-6), crossWPt, Vector(b0, b1, b2))
end
