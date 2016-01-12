local W,H = opt.Dim("W",0), opt.Dim("H",1)
local S = ad.ProblemSpec()
S:UsePreconditioner(true)

local terms = terralib.newlist()
function Energy(t)
    terms:insert(Select(opt.InBounds(0,0,0,0),t,0))
end

function evalRot(CosAlpha, SinAlpha)
	return ad.Vector(CosAlpha, -SinAlpha, SinAlpha, CosAlpha)
end

function evalR (angle)
	return evalRot(ad.cos(angle), ad.sin(angle))
end

function mul(matrix, v)
	return ad.Vector(matrix(0)*v(0)+matrix(1)*v(1), matrix(2)*v(0)+matrix(3)*v(1))
end

function Array2D(n,t,W,H,idx)
    return S:Image(n,t,W,H,idx)
end
function Param(name,t,idx) return S:Param(name,t,idx) end

function Rotate(angle, vector)
    return mul(evalR(angle),vector)
end 
function Slice(im,s,e)
    return setmetatable({},{
            __call = function(self,x,y)
                if s + 1 == e then return im(x,y,s) end
                local t = terralib.newlist()
                for i = s,e - 1 do
                    t:insert(im(x,y,i))
                end
                return ad.Vector(unpack(t))
            end })
end
function InBounds(x,y) return opt.InBounds(x,y,0,0) end
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
function All(v)
    local r = 1
    for i = 0,v:size() - 1 do
        r = r * v(i)
    end
    return r
end

-----------------------------------------------------------------------------------------
local X = Array2D("X", opt.float3,W,H,0)				--uv, a <- unknown
local UrShape = Array2D("UrShape", opt.float2,W,H,1)		--urshape
local Constraints = Array2D("Constraints", opt.float2,W,H,2)	--constraints
local Mask = Array2D("Mask", float, W,H,3)				--validity mask for constraints
local w_fitSqrt = Param("w_fitSqrt", float, 0)
local w_regSqrt = Param("w_regSqrt", float, 1)
local Offsets,Angle = Slice(X,0,2), Slice(X,2,3)

--regularization
for x,y in Stencil { {1,0}, {-1,0}, {0,1}, {0, -1} } do
    local e_reg = w_regSqrt*((Offsets(0,0) - Offsets(x,y)) 
                             - Rotate(Angle(0,0),(UrShape(0,0) - UrShape(x,y))))          
    local valid = InBounds(x,y) * eq(Mask(x,y),0)
    Energy(Select(valid,e_reg,0))
end

--fitting
local e_fit = w_fitSqrt*(Offsets(0,0)- Constraints(0,0))
local valid = All(greatereq(Constraints(0,0),0))
Energy(Select(valid, e_fit , 0.0))
-----------------------------------------------------------------------------------------

return S:Cost(ad.sumsquared(unpack(terms)))


