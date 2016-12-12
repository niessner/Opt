local IO = terralib.includec("stdio.h")
local adP = ad.ProblemSpec()
local P = adP.P
local W,H = opt.Dim("W",0), opt.Dim("H",1)

P:Unknown("X", opt.float4,{W,H},0)
P:Image("T", opt.float4,{W,H},1)
P:Image("M", float, {W,H} ,2)

P:Stencil(1)
P:UsePreconditioner(false)	--TODO needs to be implemented (in this file)


local WH = opt.toispace {W,H}

local TUnknownType = P:UnknownType():terratype()
local Index = WH:indextype()
local unknownElement = opt.float4

local ImageType = P:ImageType(opt.float4,WH):terratype()
local C = {}

local terra laplacianCost(x : unknownElement, t : unknownElement, idx : Index, xImage : ImageType, tImage : ImageType) : unknownElement
	
	var res : unknownElement = 0.0f
	
	if idx:InBounds() then
		res = (x - xImage(idx)) - (t - tImage(idx))
	end

	return res	
end

local terra laplacianCostP(x : unknownElement, idx : Index, xImage : ImageType) : unknownElement
	
	var res : unknownElement = 0.0f
	
	if idx:InBounds() then
		res = x - xImage(idx)
	end

	return res	
end


terra C.cost(idx : Index, self : P:ParameterType()) : float
	
	var m = self.M(idx)(0)
	var x = self.X(idx)
	var t = self.T(idx)	
	
	var res : float = 0.0f
	
	if m == 0 then
		var l0 = laplacianCost(x,t,idx(1,0),self.X.X,self.T)
		var l1 = laplacianCost(x,t,idx(0,1),self.X.X,self.T)
		var l2 = laplacianCost(x,t,idx(-1,0),self.X.X,self.T)
		var l3 = laplacianCost(x,t,idx(0,-1),self.X.X,self.T)
	
		var l = l0*l0 + l1*l1 + l2*l2 + l3*l3
		res = l(0) + l(1) + l(2) + l(3)
	end
	
	return res
end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
terra C.gradient(idx : Index, self : P:ParameterType()) : unknownElement

	var m = self.M(idx)(0)
	var x = self.X(idx)
	var t = self.T(idx)
	
	var res : unknownElement = 0.0f
	
	if m == 0 then	
		var l0 = 2.0f*laplacianCost(x,t,idx(1,0),self.X.X,self.T)
		var l1 = 2.0f*laplacianCost(x,t,idx(0,1),self.X.X,self.T)
		var l2 = 2.0f*laplacianCost(x,t,idx(-1,0),self.X.X,self.T)
		var l3 = 2.0f*laplacianCost(x,t,idx(0,-1),self.X.X,self.T)	
		
		var laplacian = l0 + l1 + l2 + l3
	
		res = 2.0f*laplacian		
	end
	
	return res
end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
terra C.evalJTF(idx : Index, self : P:ParameterType())
	
	var gradient : unknownElement = C.gradient(idx, self)
	
	var pre : float = 1.0f
	return gradient, pre
end
	
-- eval 2*JtJ (note that we keep the '2' to make it consistent with the gradient
terra C.applyJTJ(idx : Index, self : P:ParameterType(), pImage : TUnknownType, ctcImage : TUnknownType) : unknownElement
	
	var m = self.M(idx)(0)
	var p = pImage(idx)
	
	var res : unknownElement = 0.0f
	
	if m == 0 then
		var l0 = 2.0f*laplacianCostP(p,idx(1,0),pImage.X)
		var l1 = 2.0f*laplacianCostP(p,idx(0,1),pImage.X)
		var l2 = 2.0f*laplacianCostP(p,idx(-1,0),pImage.X)
		var l3 = 2.0f*laplacianCostP(p,idx(0,-1),pImage.X)	
		var laplacian = l0 + l1 + l2 + l3
		res = 2.0f*laplacian
	end
	
	return res
end

P:Functions(WH,C)
return P