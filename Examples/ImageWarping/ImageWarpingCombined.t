local M = require("ImageWarping")
local A = require("ImageWarpingAD")


local oldJTJ = A.functions.applyJTJ.boundary
A.functions.applyJTJ.boundary = terra(i : int32, j : int32, gi : int32, gj : int32, self : A:ParameterType(), pImage : A:UnknownType())
	var a = oldJTJ(i,j,gi,gj,self,pImage)
	var b = M.functions.applyJTJ.boundary(i,j,gi,gj,@[&M:ParameterType()](&self),@[&M:UnknownType()](&pImage))

	var diff = a - b
	var d : float = diff:dot(diff)
	if d < 0.0f then d = -d end
	if d > 0.1f then
		--printf("%d,%d: ad=%f b=%f\n",int(gi),int(gj),a(0),b(0))
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

	var diff = a - b
	var d : float = diff:dot(diff)
	if d < 0.0f then d = -d end
	if d > 0.001f then
	--if gi == 240 and gj == 80 then
		printf("JTF (%d|%d): pa=%f %f %f pb=%f %f %f\n",int(gi),int(gj),a(0),a(1),a(2),b(0),b(1),b(2))
	end
	
	if a(0) ~= 0.0f or a(1) ~= 0.0f or a(2) ~= 0.0f then
		--printf("(%d|%d): pa=%f pb=%f\n",int(gi),int(gj),a(0),b(0))
	end
	if b(0) ~= 0.0f or b(1) ~= 0.0f or b(2) ~= 0.0f then
		--printf("JTF (%d|%d): pa=%f %f %f pb=%f %f %f\n",int(gi),int(gj),a(0),a(1),a(2),b(0),b(1),b(2))
	end
	
	--pa(2) = -1.0f
	--pb(2) = -1.0f
	
	var diffPre = pa - pb
	var diffPreF : float = diffPre:dot(diffPre)
	if diffPreF < 0.0f then diffPreF = -diffPreF end
	if diffPreF > 0.1f then
		printf("(%d|%d): precond: pad=%f %f %f pb=%f %f % f\n",int(gi),int(gj),pa(0),pa(1),pa(2),pb(0),pb(1),pb(2))
	end	
	return a,pa
end


local cost = A.functions.cost.boundary
A.functions.cost.boundary = terra(i : int32, j : int32, gi : int32, gj : int32, self : A:ParameterType())
	var a = cost(i, j, gi, gj, self)	--auto-diff
	var b = M.functions.cost.boundary(i,j,gi,gj,@[&M:ParameterType()](&self))

	var weightFit : float = 100.0f
	var weightReg : float = 0.01f
	
	var d : float = a - b
	if d < 0.0f then d = -d end
	if d > 0.1f then
		var expected0 : float = self.X(i,j)(0) - self.Constraints(i,j)(0) 
		expected0 = weightFit*expected0*expected0
		var expected1 : float = self.X(i,j)(1) - self.Constraints(i,j)(1) 
		expected1 = weightFit*expected1*expected1
		var expected : float = expected0 + expected1
		
		printf("cost (%d|%d): ad=%f b=%f\n",int(gi),int(gj),a,b)
		printf("x: %f %f %f;  c: %f %f;   e: %f\n", self.X(i,j)(0), self.X(i,j)(1), self.X(i,j)(2), self.Constraints(i,j)(0), self.Constraints(i,j)(1), expected)
	end
	
	--if gi == 0 and gj == 0 then
	--	printf("bla\n")
	--end
	
	return a
end

return A