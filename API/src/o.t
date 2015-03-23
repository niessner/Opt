opt = {} --anchor it in global namespace, otherwise it can be collected

local S = require("std")

local malloc = terralib.externfunction("malloc",uint64 -> &opaque)

local function newclass(name)
    local mt = { __name = name }
    mt.__index = mt
    function mt:is(obj)
        return getmetatable(obj) == self
    end
    function mt:__tostring()
        return "<"..name.." instance>"
    end
    function mt:new(obj)
        obj = obj or {}
        setmetatable(obj,self)
        return obj
    end
    return mt
end


struct opt.ImageBinding(S.Object) {
    data : &uint8
    stride : uint64
    elemsize : uint64
}
terra opt.ImageBinding:get(x : uint64, y : uint64) : &uint8
   return self.data + self.elemsize*(y*self.stride + x)
end

terra opt.ImageBind(data : &opaque, elemsize : uint64, stride : uint64) : &opt.ImageBinding
    var img = opt.ImageBinding.alloc()
    img.data,img.stride,img.elemsize = [&uint8](data),stride,elemsize
    return img
end
terra opt.ImageFree(img : &opt.ImageBinding)
    img:delete()
end

local Problem = newclass("Problem")

local ffi = require('ffi')

local problems = {}

function opt.ProblemDefineFromTable(tbl)
    local p = Problem:new { definition = tbl }
    problems[#problems+1] = p
    print("defining problem",#problems)
    return #problems
end
local function problemdefine(filename,result)
    local success,id = xpcall(function() 
        filename = ffi.string(filename)
        local tbl = assert(terralib.loadfile(filename))() 
        assert(type(tbl) == "table")
        return opt.ProblemDefineFromTable(tbl)
    end,function(err) return debug.traceback(err,2) end)
    if not success then error(id,0) end
    result[0] = id
end

struct opt.Problem {} -- actually &opt.Problems are just integers that point into the problems table

terra opt.ProblemDefine(filename : rawstring)
    var result : int
    problemdefine(filename,&result)
    return [&opt.Problem](result) --cast integer into pointer for API purposes
end 

local Dim = newclass("dimension")

function opt.Dim(name)
    return Dim:new { name = name }
end

opt.Unity = opt.Dim("unity") -- represent '1' as a dimension

local newimage = terralib.memoize(function(typ,W,H)
   local struct Image {
        impl : opt.ImageBinding
   }
   function Image.metamethods.__tostring()
      return string.format("Image(%s,%s,%s)",tostring(typ),W.name, H.name)
   end
   Image.metamethods.__apply = macro(function(self,x,y)
     return `@[&typ](self.impl:get(x,y))
   end)
   return Image
end)

function opt.Image(typ,W,H)
    if W == 1 then W = opt.Unity end
    if H == 1 then H = opt.Unity end
    assert(Dim:is(W) and Dim:is(H))
    assert(terralib.types.istype(typ))
    return newimage(typ,W,H)
end

struct opt.Plan(S.Object) {} 
terra opt.ProblemPlan(problem : &opt.Problem, kind : rawstring, dims : &uint64) : &opt.Plan
    -- Do allocations for problem here. 
    -- we need to figure out what form this would take.
    -- e.g., 'problem' itself might hold the pointer to the thing that allocates its plan.
    return opt.Plan.alloc()
end 

terra opt.PlanFree(plan : &opt.Plan)
    --...
    plan:delete()
end

terra opt.ProblemSolve(plan : &opt.Plan, images : &opt.ImageBinding)
end

return opt