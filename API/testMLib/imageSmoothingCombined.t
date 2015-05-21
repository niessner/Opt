local M = require("imageSmoothing")
local A = require("imageSmoothingAD")


local oldJTJ = A.functions.applyJTJ.boundary
A.functions.applyJTJ.boundary = terra(i : int64, j : int64, gi : int64, gj : int64, self : A:ParameterType(), pImage : A:UnknownType())
	var a : float = oldJTJ(i,j,gi,gj,self,pImage)
	var b : float = M.functions.applyJTJ.boundary(i,j,gi,gj,@[&M:ParameterType()](&self),@[&M:UnknownType()](&pImage))
	--if gi > 64 and gj > 64 and gi < 32 and gj < 32 then
	--if a ~= 0.0f then
	var d : float = a - b
	if d < 0.0f then d = -d end
	if d > 0.001f then
		printf("%d,%d: ad=%f b=%f\n",int(gi),int(gj),a,b)
	end
	return b
end

return A