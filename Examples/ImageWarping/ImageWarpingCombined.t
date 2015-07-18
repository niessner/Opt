local M = require("ImageWarping")
local A = require("ImageWarpingAD")

local smallestnormalfloat = `1.1754943508222875081e-38f

local terra relerror(a : float, b : float) : float
    var absA,absB = opt.math.abs(a),opt.math.abs(b)
    var diff = opt.math.abs(a - b)
    if a == b then return 0.f
    elseif (absA <= smallestnormalfloat and absB <= smallestnormalfloat) or diff <= smallestnormalfloat then
        return diff -- too close to zero to divide
    else
        return diff / opt.math.fmax(absA,absB)
    end
end

A.functions.applyJTJ.boundary = terra(i : int32, j : int32, gi : int32, gj : int32, self : A:ParameterType(), pImage : A:UnknownType())
	var a = A.functions.applyJTJ.boundary(i,j,gi,gj,self,pImage)
	var b = M.functions.applyJTJ.boundary(i,j,gi,gj,@[&M:ParameterType()](&self),@[&M:UnknownType()](&pImage))

	
	
	var constraintUV = self.Constraints(i,j)	
	var validConstraint = (constraintUV(0) >= 0 and constraintUV(1) >= 0)
	var valid  = (self.Mask(i+0, j+0) == 0.0f)
	var valid0 = (self.Mask(i+0, j-1) == 0.0f)
	var valid1 = (self.Mask(i+0, j+1) == 0.0f)
	var valid2 = (self.Mask(i-1, j+0) == 0.0f)
	var valid3 = (self.Mask(i+1, j+0) == 0.0f)
	
	for i = 0,3 do
	    if relerror(a(i), b(i)) > 1e-3  and b(i) > 1e-5 then
    		printf("JTJ (%d,%d)[%d]: a = %e, b = %e, err = %e (%d,%d,%d,%d,%d,%d)\n",int(gi),int(gj),i,a(i),b(i),relerror(a(i), b(i)), int(validConstraint),int(valid),int(valid0),int(valid1),int(valid2),int(valid3))
    	end
	end
	return b
end

A.functions.evalJTF.boundary = terra(i : int32, j : int32, gi : int32, gj : int32, self : A:ParameterType())
	var a,pa = A.functions.evalJTF.boundary(i, j, gi, gj, self)	--auto-diff
	var b,pb = M.functions.evalJTF.boundary(i,j,gi,gj,@[&M:ParameterType()](&self))
	var c,pc = M.functions.evalJTFNumeric.boundary(i,j,gi,gj,@[&M:ParameterType()](&self))
	
	for i = 0,3 do
	    if relerror(a(i), b(i)) > 1e-3  and b(i) > 1e-5 then
    		printf("JTF (%d,%d)[%d]: a = %e, b = %e, c = %e, err = %e\n",int(gi),int(gj),i,a(i),b(i),c(i),relerror(a(i), b(i)))
    	end
		
		if relerror(pa(i), pb(i)) > 1e-3  and pb(i) > 1e-5 then
    		printf("JTF (%d,%d)[%d]: pa = %e, pb = %e, pc = %e, err = %e\n",int(gi),int(gj),i,pa(i),pb(i),pc(i),relerror(pa(i), pb(i)))
    	end
	end

    return b,pb
end

A.functions.cost.boundary = terra(i : int32, j : int32, gi : int32, gj : int32, self : A:ParameterType())
	var a = A.functions.cost.boundary(i, j, gi, gj, self)	--auto-diff
	var b = M.functions.cost.boundary(i,j,gi,gj,@[&M:ParameterType()](&self))

	var weightFit : float = 100.0f
	var weightReg : float = 0.01f
	
	var d : float = a - b
	if d < 0.0f then d = -d end
	if d > 0.1f then
		printf("cost (%d|%d): ad=%f b=%f\n",int(gi),int(gj),a,b)
	end
	
	return b
end

return A