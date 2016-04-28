return function(P)
    local terms = terralib.newlist()

    local L = {}

    function L.Energy(...)
        for i,e in ipairs {...} do
            terms:insert(e)
        end
    end

    function L.Result() return P:Cost(unpack(terms)) end
    function L.All(v)
        local r = 1
        for i = 0,v:size() - 1 do
            r = r * v(i)
        end
        return r
    end

    function L.Reduce(fn,init)
        return function(...)
            local r = init
            for _,e in ipairs {...} do
                r = fn(r,e)
            end
            return r
        end
    end
    L.And = L.Reduce(1,ad.and_)
    L.Or = L.Reduce(0,ad.or_)
    -- alas for Image/Array
    function L.Array(...) return P:Image(...) end
    function L.ComputedArray(...) return P:ComputedImage(...) end
--[[
    function L.Rotate3D(a,v)
        local alpha, beta, gamma = a(0), a(1), a(2)
        local  CosAlpha, CosBeta, CosGamma, SinAlpha, SinBeta, SinGamma = ad.cos(alpha), ad.cos(beta), ad.cos(gamma), ad.sin(alpha), ad.sin(beta), ad.sin(gamma)
        local matrix = ad.Vector(
            CosGamma*CosBeta, 
            -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha, 
            SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha,
            SinGamma*CosBeta,
            CosGamma*CosAlpha + SinGamma*SinBeta*SinAlpha,
            -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha,
            -SinBeta,
            CosBeta*SinAlpha,
            CosBeta*CosAlpha)
        return  ad.Vector(
                matrix(0)*v(0)+matrix(1)*v(1)+matrix(2)*v(2),
                matrix(3)*v(0)+matrix(4)*v(1)+matrix(5)*v(2),
                matrix(6)*v(0)+matrix(7)*v(1)+matrix(8)*v(2))
    end
]]
    L.Select = ad.select
    function L.Stencil (lst)
        local i = 0
        return function()
            i = i + 1
            if not lst[i] then return nil
            else return unpack(lst[i]) end
        end
    end
    setmetatable(L,{__index = function(self,key)
        if type(P[key]) == "function" then
            return function(...) return P[key](P,...) end
        end
        if key ~= "select" and ad[key] then return ad[key] end
        if opt[key] then return opt[key] end
        return _G[key]
    end})
    return L
end