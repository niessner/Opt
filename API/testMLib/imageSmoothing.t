local IO = terralib.includec("stdio.h")

local P = opt.ProblemSpec()
local W = opt.Dim("W",0)
local H = opt.Dim("H",1)

P:Image("X",float,W,H,0)
P:Image("A",float,W,H,1)
P:Param("Foo",int,0)
P:Stencil(2)

local C = terralib.includecstring [[
#include <math.h>
]]

-- e = sum_{i,j} [ e_reg_{i,j}^2 + e_fit_{i,j}^2 ]
-- e_reg_{i,j} = I(i,j) - 0.25 * [ I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1) ]
-- e_fit_{i,j} = || I(i,j) - \hat{I}(i,j) ||_2


-- TODO: this should be factored into a parameter
local w_fit = 0.1
local w_reg = 1.0


local terra inLaplacianBounds(i : int64, j : int64, xImage : P:UnknownType()) : bool
	return opt.InBoundsCalc(i,j,xImage:W(),xImage:H(),1,1)
	--return i > 0 and i < xImage:W()-1 and j > 0 and j < xImage:H()-1
end



--local terra laplacian(i : int64, j : int64, gi : int64, gj : int64, xImage : P:UnknownType()) : float
local laplacian = macro(function(i, j , gi, gj, xImage)
    return quote
        var r : float
	    if inLaplacianBounds(gi, gj, xImage) then
		    r = xImage(i, j)*4 - (xImage(i - 1, j) + xImage(i, j + 1) + xImage(i + 1, j) + xImage(i, j - 1))
        else
		    r = 0
	    end
	in r end
end)

-- diagonal of JtJ; i.e., (Jt)^2
local terra laplacianPreconditioner(gi : int64, gj : int64, xImage : P:UnknownType()) : float

	var p : float = 0.0
	if inLaplacianBounds(gi+0, gj+0, xImage) then	p = p + 4*4			end
	if inLaplacianBounds(gi+1, gj+0, xImage) then	p = p + (-1)*(-1)	end
	if inLaplacianBounds(gi+0, gj+1, xImage) then	p = p + (-1)*(-1)	end
	if inLaplacianBounds(gi-1, gj+0, xImage) then	p = p + (-1)*(-1)	end
	if inLaplacianBounds(gi+0, gj-1, xImage) then	p = p + (-1)*(-1)	end
	
	return p
end

local terra cost(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType())
	
	var x = self.X(i, j)
	var a = self.A(i, j)

	var v2 = x - a
	var e_fit = v2 * v2
	
	var v = laplacian(i, j, gi, gj, self.X)
	var e_reg = v * v

	var res = (float)(w_fit*e_fit + w_reg*e_reg)
	return res
	
end

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
local terra gradient(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType())
	
	var x = self.X(i, j)
	var a = self.A(i, j)
	
	var e_fit = (x - a)

    var x0 = laplacian(i, j, gi, gj, self.X)*4
    var x1 = laplacian(i + 1, j, gi + 1, gj, self.X)
    var x2 = laplacian(i - 1, j, gi - 1, gj, self.X)
    var x3 = laplacian(i, j + 1, gi, gj + 1, self.X)
    var x4 = laplacian(i, j - 1, gi, gj - 1, self.X)

	var e_reg = 
		x0
		- x1
		- x2
		- x3
		- x4
	e_reg = e_reg

	return 2*(w_fit*e_fit + w_reg*e_reg)
end


gradient:printpretty(true,false)

-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
local terra evalJTF(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType())
	
	var gradient = gradient(i, j, gi, gj, self)
	
	var pre : float = 1.0f
	if P.usepreconditioner then		--pre-conditioner
		var p_fit : float = 2.0	
		var p_reg : float = 2.0 * laplacianPreconditioner(gi, gj, self.X)
			
		pre = w_fit*p_fit + w_reg*p_reg
		if pre > 0.0001 then
			pre = 1.0 / pre
		else 
			pre = 1.0
		end
	end
	return gradient, pre
end
	

-- eval 2*JtJ (note that we keep the '2' to make it consistent with the gradient
local terra applyJTJ(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType(), pImage : P:UnknownType())
 
	--fit
	var e_fit = 2.0f*pImage(i, j)
	
	--reg
	var e_reg = 
		4*laplacian(i + 0, j + 0, gi + 0, gj + 0, pImage)
		-laplacian(i + 1, j + 0, gi + 1, gj + 0, pImage)
		-laplacian(i - 1, j + 0, gi - 1, gj + 0, pImage)
		-laplacian(i + 0, j + 1, gi + 0, gj + 1, pImage)
		-laplacian(i + 0, j - 1, gi + 0, gj - 1, pImage)
	e_reg = 2.0*e_reg
	
	return w_fit*e_fit + w_reg*e_reg
end

P:Function("cost", {W,H}, cost)
P:Function("gradient", {W,H}, gradient)
P:Function("evalJTF", {W,H}, evalJTF)
P:Function("applyJTJ", {W,H}, applyJTJ)


return P