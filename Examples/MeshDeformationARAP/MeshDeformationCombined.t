package.terrapath = package.terrapath..";./?.t"

local A = require("MeshDeformationAD")
local M = require("MeshDeformation")

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

A.functions.applyJTJ.unknownfunction = terra(i : int32, j : int32, gi : int32, gj : int32, self : A:ParameterType(), pImage : A:UnknownType())
	--var a = A.functions.applyJTJ.unknownfunction(i,j,gi,gj,self,pImage)
	var b = M.functions.applyJTJ.unknownfunction(i,j,gi,gj,@[&M:ParameterType()](&self),@[&M:UnknownType()](&pImage))
	return b
end

A.functions.applyJTJ.graphfunctions[1].implementation = terra(idx : int32, self : A:ParameterType(), pImage : A:UnknownType(), Ap_X : A:UnknownType())
	--var a = [A.functions.applyJTJ.graphfunctions[1].implementation](idx, self, pImage, r)	--auto-diff
	var b = [M.functions.applyJTJ.graphfunctions[1].implementation](idx,@[&M:ParameterType()](&self), @[&M:UnknownType()](&pImage), @[&M:UnknownType()](&Ap_X))  
	return b
end



A.functions.evalJTF.unknownfunction = terra(i : int32, j : int32, gi : int32, gj : int32, self : A:ParameterType())
	--var a,pa = A.functions.evalJTF.unknownfunction(i, j, gi, gj, self)	--auto-diff
	--return a,pa
	var b,pb = M.functions.evalJTF.unknownfunction(i,j,gi,gj,@[&M:ParameterType()](&self))   
	return b,pb
end


A.functions.evalJTF.graphfunctions[1].implementation = terra(idx : int32, self : A:ParameterType(), pImage : A:UnknownType(), r : A:UnknownType())
	--[A.functions.evalJTF.graphfunctions[1].implementation](idx, self, pImage, r)	--auto-diff
	[M.functions.evalJTF.graphfunctions[1].implementation](idx,@[&M:ParameterType()](&self), pImage, r) 
end


A.functions.cost.unknownfunction = terra(i : int32, j : int32, gi : int32, gj : int32, self : A:ParameterType())
	--var a = A.functions.cost.unknownfunction(i, j, gi, gj, self)	--auto-diff
	--return a
	var b = M.functions.cost.unknownfunction(i,j,gi,gj,@[&M:ParameterType()](&self))
	return b
end


return A