local M = require("imageSmoothing")
local A = require("imageSmoothingAD")

local w_fit = 0.1
local w_reg = 1.0


local oldJTJ = A.functions.applyJTJ.boundary
A.functions.applyJTJ.boundary = terra(i : int32, j : int32, gi : int32, gj : int32, self : A:ParameterType(), pImage : A:UnknownType())
	var a : float = oldJTJ(i,j,gi,gj,self,pImage)
	var b : float = M.functions.applyJTJ.boundary(i,j,gi,gj,@[&M:ParameterType()](&self),@[&M:UnknownType()](&pImage))
	--if gi > 64 and gj > 64 and gi < 32 and gj < 32 then
	--if a ~= 0.0f then
	var d : float = a - b
	if d < 0.0f then d = -d end
	if d > 0.1f then
		--printf("%d,%d: ad=%f b=%f\n",int(gi),int(gj),a,b)
	end
	--[[
	if gi == 0 and gj == 7 then
	    var special = A.functions.special.boundary(i,j,gi,gj,self,pImage)
		printf("ad=%f, b=%f, s=%f\n", a, b,special)
		printf("-4*%f + 1*%f + 1*%f + 1*%f+ 1*%f\n",pImage(gi+1,gj+0),pImage(gi+0,gj+0),pImage(gi+2,gj+0),pImage(gi+1,gj-1),pImage(gi+1,gj+1))
		var e_reg : float = -4*pImage(gi+1,gj+0)+pImage(gi+0,gj+0)+pImage(gi+2,gj+0)+pImage(gi+1,gj-1)+pImage(gi+1,gj+1)
		var e_fit : float = pImage(gi,gj)
		printf("res=%f\n", 2.0f*e_fit*w_fit +  2.0f*e_reg*w_reg)
	end
	--]]	
	return a
end

local oldJTF = A.functions.evalJTF.boundary
A.functions.evalJTF.boundary = terra(i : int32, j : int32, gi : int32, gj : int32, self : A:ParameterType())
	var a,pa = oldJTF(i, j, gi, gj, self)	--auto-diff
	var b,pb = M.functions.evalJTF.boundary(i,j,gi,gj,@[&M:ParameterType()](&self))

	var d : float = pa - pb
	if d < 0.0f then d = -d end
	if d > 0.1f then
		--if gi == 10 and gj == 10 then
			printf("%d,%d: pa=%f pb=%f\n",int(gi),int(gj),pa,pb)
		--end
	end
	
	return a,pa
end

local cost = A.functions.cost.boundary
A.functions.cost.boundary = terra(i : int32, j : int32, gi : int32, gj : int32, self : A:ParameterType())
	var a = cost(i, j, gi, gj, self)	--auto-diff
	var b = M.functions.cost.boundary(i,j,gi,gj,@[&M:ParameterType()](&self))

	var d : float = a - b
	if d < 0.0f then d = -d end
	if d > 0.1f then
		--printf("%d,%d: ad=%f b=%f\n",int(gi),int(gj),a,b)
	end
	
	return a
end

return A