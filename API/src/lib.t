return function(P)
    local terms = terralib.newlist()

    local L = {}
    L.inf = math.huge
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
    L.Not = ad.not_
    
    function L.UsePreconditioner(...) return P:UsePreconditioner(...) end
    -- alas for Image/Array
    function L.Array(...) return P:Image(...) end
    function L.ComputedArray(...) return P:ComputedImage(...) end

    function L.Matrix3x3Mul(matrix,v)
        return ad.Vector(
            matrix(0)*v(0)+matrix(1)*v(1)+matrix(2)*v(2),
            matrix(3)*v(0)+matrix(4)*v(1)+matrix(5)*v(2),
            matrix(6)*v(0)+matrix(7)*v(1)+matrix(8)*v(2))
    end

    function L.Dot3(v0,v1)
        return v0(0)*v1(0)+v0(1)*v1(1)+v0(2)*v1(2)
    end

    function L.Dot4(v0,v1)
        return v0(0)*v1(0)+v0(1)*v1(1)+v0(2)*v1(2)+v0(3)*v1(3)
    end

    function L.Sqrt(v)
        return ad.sqrt(v)
    end

    function L.normalize(v)
        return v / ad.sqrt(L.Dot3(v, v))
    end

    function L.length(v0, v1) 
        local diff = v0 - v1
        return ad.sqrt(L.Dot3(diff, diff))
    end
    L.length3 = L.length

   function L.SelectOnAll(pList,val,default)
        assert(#pList > 0, "SelectOnAll() requires at least one predicate")
        local result = L.Select(pList[#pList], val, default)
        for i=1,#pList-1 do
            local p = pList[#pList-i]
            result = L.Select(p, result, default)
        end
        return result
    end

    --TODO: Check transpose is correct
    function L.RodriguesSO3Exp(w, A, B)
        local R00, R01, R02
        local R10, R11, R12
        local R20, R21, R22
        do
            local wx2 = w(0) * w(0);
            local wy2 = w(1) * w(1);
            local wz2 = w(2) * w(2);
            R00 = 1.0 - B*(wy2 + wz2);
            R11 = 1.0 - B*(wx2 + wz2);
            R22 = 1.0 - B*(wx2 + wy2);
        end
        do
            local a = A*w(2)
            local b = B*(w(0) * w(1))
            R01 = b - a;
            R10 = b + a;
        end
        do
            local a = A*w(1)
            local b = B*(w(0) * w(2))
            R02 = b + a;
            R20 = b - a;
        end
        do
            local a = A*w(0)
            local b = B*(w(1) * w(2))
            R12 = b - a;
            R21 = b + a;
        end
        return ad.Vector(   R00, R01, R02,
                            R10, R11, R12,
                            R20, R21, R22)
    end

    function L.Cross3(a,b)
        return ad.Vector(a(1)*b(2) - a(2)*b(1), a(2)*b(0) - a(0)*b(2), a(0)*b(1) - a(1)*b(0))
    end

    function L.Matrix4(...)
        assert(select("#",...) == 16, "Provided "..tostring(select("#",...)).." elements to Matrix4 constructor, need 16")
        return ad.Vector(...)
    end

    function L.Vec4(...)
        assert(select("#",...) == 4, "Provided "..tostring(select("#",...)).." elements to Vec4 constructor, need 4")
        return ad.Vector(...)
    end

    function L.RotationMatrixAndTranslationToMat4(r, t)
        return ad.Vector(  r(0), r(1), r(2), t(0),
                    r(3), r(4), r(5), t(1),
                    r(6), r(7), r(8), t(2),
                     0.0,  0.0,  0.0, 1.0)
    end


    function L.CameraToDepth(fx, fy, cx, cy, pos)
        return ad.Vector(
            pos(0)*fx / pos(2) + cx,
            pos(1)*fy / pos(2) + cy)
    end

    function L.Max(a,b)
        return ad.select(ad.greater(a,b),a,b)
    end

    -- TODO: check transpose
    function L.Mat4Mul(a,b)
        return ad.Vector(
            a(0)*b(0) + a(1)*b(4) + a(2)*b(8) + a(3)*b(12),  
            a(0)*b(1) + a(1)*b(5) + a(2)*b(9) + a(3)*b(13),  
            a(0)*b(2) + a(1)*b(6) + a(2)*b(10) + a(3)*b(14), 
            a(0)*b(3) + a(1)*b(7) + a(2)*b(11) + a(3)*b(15),

            a(4)*b(0) + a(5)*b(4) + a(6)*b(8) + a(7)*b(12),  
            a(4)*b(1) + a(5)*b(5) + a(6)*b(9) + a(7)*b(13),  
            a(4)*b(2) + a(5)*b(6) + a(6)*b(10) + a(7)*b(14), 
            a(4)*b(3) + a(5)*b(7) + a(6)*b(11) + a(7)*b(15),

            a(8)*b(0) + a(9)*b(4) + a(10)*b(8) + a(11)*b(12),  
            a(8)*b(1) + a(9)*b(5) + a(10)*b(9) + a(11)*b(13),  
            a(8)*b(2) + a(9)*b(6) + a(10)*b(10) + a(11)*b(14), 
            a(8)*b(3) + a(9)*b(7) + a(10)*b(11) + a(11)*b(15),

            a(12)*b(0) + a(13)*b(4) + a(14)*b(8) + a(15)*b(12),  
            a(12)*b(1) + a(13)*b(5) + a(14)*b(9) + a(15)*b(13),  
            a(12)*b(2) + a(13)*b(6) + a(14)*b(10) + a(15)*b(14), 
            a(12)*b(3) + a(13)*b(7) + a(14)*b(11) + a(15)*b(15)
        )
    end


    function L.InverseMatrix4(entries)
        local inv = {}
        inv[0] = entries[5]  * entries[10] * entries[15] - 
            entries[5]  * entries[11] * entries[14] - 
            entries[9]  * entries[6]  * entries[15] + 
            entries[9]  * entries[7]  * entries[14] +
            entries[13] * entries[6]  * entries[11] - 
            entries[13] * entries[7]  * entries[10]

        inv[4] = -entries[4]  * entries[10] * entries[15] + 
            entries[4]  * entries[11] * entries[14] + 
            entries[8]  * entries[6]  * entries[15] - 
            entries[8]  * entries[7]  * entries[14] - 
            entries[12] * entries[6]  * entries[11] + 
            entries[12] * entries[7]  * entries[10]

        inv[8] = entries[4]  * entries[9] * entries[15] - 
            entries[4]  * entries[11] * entries[13] - 
            entries[8]  * entries[5] * entries[15] + 
            entries[8]  * entries[7] * entries[13] + 
            entries[12] * entries[5] * entries[11] - 
            entries[12] * entries[7] * entries[9]

        inv[12] = -entries[4]  * entries[9] * entries[14] + 
            entries[4]  * entries[10] * entries[13] +
            entries[8]  * entries[5] * entries[14] - 
            entries[8]  * entries[6] * entries[13] - 
            entries[12] * entries[5] * entries[10] + 
            entries[12] * entries[6] * entries[9]

        inv[1] = -entries[1]  * entries[10] * entries[15] + 
            entries[1]  * entries[11] * entries[14] + 
            entries[9]  * entries[2] * entries[15] - 
            entries[9]  * entries[3] * entries[14] - 
            entries[13] * entries[2] * entries[11] + 
            entries[13] * entries[3] * entries[10]

        inv[5] = entries[0]  * entries[10] * entries[15] - 
            entries[0]  * entries[11] * entries[14] - 
            entries[8]  * entries[2] * entries[15] + 
            entries[8]  * entries[3] * entries[14] + 
            entries[12] * entries[2] * entries[11] - 
            entries[12] * entries[3] * entries[10]

        inv[9] = -entries[0]  * entries[9] * entries[15] + 
            entries[0]  * entries[11] * entries[13] + 
            entries[8]  * entries[1] * entries[15] - 
            entries[8]  * entries[3] * entries[13] - 
            entries[12] * entries[1] * entries[11] + 
            entries[12] * entries[3] * entries[9]

        inv[13] = entries[0]  * entries[9] * entries[14] - 
            entries[0]  * entries[10] * entries[13] - 
            entries[8]  * entries[1] * entries[14] + 
            entries[8]  * entries[2] * entries[13] + 
            entries[12] * entries[1] * entries[10] - 
            entries[12] * entries[2] * entries[9]

        inv[2] = entries[1]  * entries[6] * entries[15] - 
            entries[1]  * entries[7] * entries[14] - 
            entries[5]  * entries[2] * entries[15] + 
            entries[5]  * entries[3] * entries[14] + 
            entries[13] * entries[2] * entries[7] - 
            entries[13] * entries[3] * entries[6]

        inv[6] = -entries[0]  * entries[6] * entries[15] + 
            entries[0]  * entries[7] * entries[14] + 
            entries[4]  * entries[2] * entries[15] - 
            entries[4]  * entries[3] * entries[14] - 
            entries[12] * entries[2] * entries[7] + 
            entries[12] * entries[3] * entries[6]

        inv[10] = entries[0]  * entries[5] * entries[15] - 
            entries[0]  * entries[7] * entries[13] - 
            entries[4]  * entries[1] * entries[15] + 
            entries[4]  * entries[3] * entries[13] + 
            entries[12] * entries[1] * entries[7] - 
            entries[12] * entries[3] * entries[5]

        inv[14] = -entries[0]  * entries[5] * entries[14] + 
            entries[0]  * entries[6] * entries[13] + 
            entries[4]  * entries[1] * entries[14] - 
            entries[4]  * entries[2] * entries[13] - 
            entries[12] * entries[1] * entries[6] + 
            entries[12] * entries[2] * entries[5]

        inv[3] = -entries[1] * entries[6] * entries[11] + 
            entries[1] * entries[7] * entries[10] + 
            entries[5] * entries[2] * entries[11] - 
            entries[5] * entries[3] * entries[10] - 
            entries[9] * entries[2] * entries[7] + 
            entries[9] * entries[3] * entries[6]

        inv[7] = entries[0] * entries[6] * entries[11] - 
            entries[0] * entries[7] * entries[10] - 
            entries[4] * entries[2] * entries[11] + 
            entries[4] * entries[3] * entries[10] + 
            entries[8] * entries[2] * entries[7] - 
            entries[8] * entries[3] * entries[6]

        inv[11] = -entries[0] * entries[5] * entries[11] + 
            entries[0] * entries[7] * entries[9] + 
            entries[4] * entries[1] * entries[11] - 
            entries[4] * entries[3] * entries[9] - 
            entries[8] * entries[1] * entries[7] + 
            entries[8] * entries[3] * entries[5]

        inv[15] = entries[0] * entries[5] * entries[10] - 
            entries[0] * entries[6] * entries[9] - 
            entries[4] * entries[1] * entries[10] + 
            entries[4] * entries[2] * entries[9] + 
            entries[8] * entries[1] * entries[6] - 
            entries[8] * entries[2] * entries[5]

        local matrixDet = entries[0] * inv[0] + entries[1] * inv[4] + entries[2] * inv[8] + entries[3] * inv[12];

        local d_r = 1.0 / matrixDet;
        local res = {}
        for i=1,16 do
            res[i] = inv[i-1]*d_r
        end
        return L.Matrix4(unpack(res))
    end

    function L.PoseToMatrix(rot,trans)
        local theta_sq = L.Dot3(rot, rot);
        local theta = L.Sqrt(theta_sq);

        local cr = L.Cross3(rot, trans);
        local smallAngle = ad.less(theta_sq, 1e-8)

        local ONE_SIXTH = (1.0/6.0)
        local ONE_TWENTIETH = (1.0/20.0)
        
        local A_s = 1.0 - ONE_SIXTH * theta_sq
        local B_s = 0.5
        local translation_s = trans + 0.5 * cr;

        local midAngle = ad.less(theta_sq, 1e-6)
        local C_m = ONE_SIXTH*(1.0 - ONE_TWENTIETH * theta_sq)
        local A_m = 1.0 - theta_sq * C_m
        local B_m = 0.5 - (0.25 * ONE_SIXTH * theta_sq)
        local inv_theta = 1.0 / theta
        local A_l = ad.sin(theta) * inv_theta
        local B_l = (1.0 - ad.cos(theta)) * (inv_theta * inv_theta)
        local C_l = (1.0 - A_l) * (inv_theta * inv_theta)
        local w_cross = L.Cross3(rot, cr)

        local translation_m = trans + B_m * cr + C_m * w_cross
        local translation_l = trans + B_l * cr + C_l * w_cross

        local translation = ad.select(smallAngle, translation_s, ad.select(midAngle, translation_m, translation_l))
        local A = ad.select(smallAngle, A_s, ad.select(midAngle, A_m, A_l))
        local B = ad.select(smallAngle, B_s, ad.select(midAngle, B_m, B_l))

        -- 3x3 rotation part:
        local rotationMatrix = L.RodriguesSO3Exp(rot, A, B)

        return L.RotationMatrixAndTranslationToMat4(rotationMatrix, translation)
    end

   function L.Mat3VMul(M,v)
        return ad.Vector(
            M(0)*v(0)+M(1)*v(1)+M(2)*v(2),
            M(3)*v(0)+M(4)*v(1)+M(5)*v(2),
            M(6)*v(0)+M(7)*v(1)+M(8)*v(2))
    end
    function L.Mat4VMul(M,v)
        return ad.Vector(
            M(0)*v(0)+  M(1)*v(1)+  M(2)*v(2)+  M(3)*v(3),
            M(4)*v(0)+  M(5)*v(1)+  M(6)*v(2)+  M(7)*v(3),
            M(8)*v(0)+  M(9)*v(1)+  M(10)*v(2)+ M(11)*v(3),
            M(12)*v(0)+ M(13)*v(1)+ M(14)*v(2)+ M(15)*v(3))
    end
    -- Assume w=1
    function L.Mat4V3Mul(M,v)
        return ad.Vector(
            M(0)*v(0)+  M(1)*v(1)+  M(2)*v(2)+  M(3),
            M(4)*v(0)+  M(5)*v(1)+  M(6)*v(2)+  M(7),
            M(8)*v(0)+  M(9)*v(1)+  M(10)*v(2)+ M(11))
    end

    function L.ShortenToVec3(v)
        return ad.Vector(v(0),v(1),v(2))
    end

    function L.Slice(im,s,e)
        return setmetatable({},{
            __call = function(self,ind)
                if s + 1 == e then return im(ind)(s) end
                local t = terralib.newlist()
                for i = s,e - 1 do
                    local val = im(ind)
                    local chan = val(i)
                    t:insert(chan)
                end
                return ad.Vector(unpack(t))
            end })
    end

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
        return L.Matrix3x3Mul(matrix,v)
    end
    function L.Rotate2D(angle, v)
	    local CosAlpha, SinAlpha = ad.cos(angle), ad.sin(angle)
        local matrix = ad.Vector(CosAlpha, -SinAlpha, SinAlpha, CosAlpha)
	    return ad.Vector(matrix(0)*v(0)+matrix(1)*v(1), matrix(2)*v(0)+matrix(3)*v(1))
    end
    L.Index = ad.Index
    L.SampledImage = ad.sampledimage

--
    function L.L_2_norm(v)
        -- TODO: check if scalar and just return
        return ad.sqrt(v:dot(v))
    end
    L.L_p_counter = 1
    function L.L_p(val, val_const, p, dims)
        local dist_const = L.L_2_norm(val_const)
        local eps = 0.0000001
        local C = ad.pow(dist_const+eps,(p-2))
        local sqrtC = ad.sqrt(C)
        local sqrtCImage = L.ComputedArray("L_p"..tostring(L.L_p_counter),dims,sqrtC)
        L.L_p_counter = L.L_p_counter + 1
        return sqrtCImage(0,0)*val
    end

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