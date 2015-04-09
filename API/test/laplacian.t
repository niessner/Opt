local IO = terralib.includec("stdio.h")

W = opt.Dim("W")
H = opt.Dim("H")

X = opt.Image(float,W,H)


-- e = sum_{i,j} [ e_reg_{i,j}^2 + e_fit_{i,j}^2 ]
-- e_reg_{i,j} = I(i,j) - 0.25 * [ I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1) ]
-- e_fit_{i,j} = || I(i,j) - \hat{I}(i,j) ||_2

local var wReg = 1.0f
local var wFit = 1.0f
	
local terra cost(i : uint64, j : uint64, xImage : X, oImage : X)
	if i < 1 or i >= W-1 then return 0.0f	end
	if j < 1 or j >= H-1 then return 0.0f	end
	
	var eReg = xImage(i, j) - 0.25f * ( xImage(i+1,j) + xImage(i-1,j) + xImage(i,j+1) + xImage(i,j-1))
    var eFit = xImage(i, j) - oImage(i, j)	

	--IO.printf("image (%d,%d)\n", xImage.impl.stride, xImage.impl.elemsize)
	--IO.printf("cost (%d,%d): x=%f, a=%f, b=%f, c=%f\n", i, j, x, a, b, c)
	return wReg*wReg * eReg*eReg + wFit*wFit * eFit*eFit;
end

local terra gradient(i : uint64, j : uint64, xImage : X, oImage : X)
	if i < 1 or i >= W-1 then return 0.0f	end
	if j < 1 or j >= H-1 then return 0.0f	end
	
	var DeReg = 2.0f * (xImage(i, j) - 0.25f * ( xImage(i+1,j) + xImage(i-1,j) + xImage(i,j+1) + xImage(i,j-1))
	var DeFit = 2.0f * (xImage(i, j) - oImage(i, j))
	
	return wReg * DeReg + wFit * DeFit;
end

return { dims = { W, H },
         cost = { dim = {W,H}, fn = cost },
         gradient = gradient }
