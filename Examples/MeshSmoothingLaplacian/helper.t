P = ad.ProblemSpec()
local One = opt.Dim("H",1)
local terms = terralib.newlist()
function Energy(...)
    for i,e in ipairs {...} do
        terms:insert(Select(opt.InBounds(0,0,0,0),e,0))
    end
end
function Result() return P:Cost(ad.sumsquared(unpack(terms))) end

function evalRot(CosAlpha, CosBeta, CosGamma, SinAlpha, SinBeta, SinGamma)
	return ad.Vector(
		CosGamma*CosBeta, 
		-SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha, 
		SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha,
		SinGamma*CosBeta,
		CosGamma*CosAlpha + SinGamma*SinBeta*SinAlpha,
		-CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha,
		-SinBeta,
		CosBeta*SinAlpha,
		CosBeta*CosAlpha)
end
	
function evalR(alpha, beta, gamma)
	return evalRot(ad.cos(alpha), ad.cos(beta), ad.cos(gamma), ad.sin(alpha), ad.sin(beta), ad.sin(gamma))
end
	
function mul(matrix, v)
	return ad.Vector(
			matrix(0)*v(0)+matrix(1)*v(1)+matrix(2)*v(2),
			matrix(3)*v(0)+matrix(4)*v(1)+matrix(5)*v(2),
			matrix(6)*v(0)+matrix(7)*v(1)+matrix(8)*v(2))
end

function Rotate(a,vector)
    return mul(evalR(a(0), a(1), a(2)), vector)
end

function Array2D(n,t,W,H,idx)
    return P:Image(n,t,W,H,idx)
end
function Array1D(n,t,W,idx)
    return P:Image(n,t,W,One,idx)
end
function Param(name,t,idx) return P:Param(name,t,idx) end

function Slice(im,s,e)
    return setmetatable({},{
            __call = function(self,x,y)
                if s + 1 == e then return im(x,y,s) end
                local t = terralib.newlist()
                for i = s,e - 1 do
                    t:insert(im(x,y)(i))
                end
                return ad.Vector(unpack(t))
            end })
end
function InBounds(x,y,z,w) return opt.InBounds(x,y,z or 0,w or 0) end
function Select(a,b,c) return ad.select(a,b,c) end
function Stencil (lst)
    local i = 0
    return function()
        i = i + 1
        if not lst[i] then return nil
        else return lst[i][1],lst[i][2] end
    end
end
eq = ad.eq
greatereq = ad.greatereq
greater = ad.greater
function All(v)
    local r = 1
    for i = 0,v:size() - 1 do
        r = r * v(i)
    end
    return r
end
function And(...)
    local r = 1
    for i,e in ipairs {...} do
        r = r * e
    end
    return r
end
sqrt = ad.sqrt
function sqMagnitude(point) 
	return (point*point):sum()
end
local name = 0
function ComputedArray(W,H,exp)
    name = name + 1
    return P:ComputedImage("CI"..tostring(name),W,H, exp)
end
Vector = ad.Vector
function Exclude(exp)
    P:Exclude(exp)
end
less = ad.less
abs = ad.abs

function UsePreconditioner(x) P:UsePreconditioner(x) end

function Graph(n,i,...)
    local args = terralib.newlist()
    for i = 1, select("#",...),3 do
        local name,dim,idx = select(i,...)
        args:insert(name)
        args:insert(dim)
        args:insert(One)
        args:insert(idx)
    end
    return P:Graph(n,i,unpack(args))
end

Not = ad.not_
Dim = opt.Dim
SampledImage = ad.sampledimage