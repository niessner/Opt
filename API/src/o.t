opt = {} --anchor it in global namespace, otherwise it can be collected

local S = require("std")

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
local Dim = newclass("dimension")


local ffi = require('ffi')

local problems = {}

-- this function should do anything it needs to compile an optimizer defined
-- using the functions in tbl, using the optimizer 'kind' (e.g. kind = gradientdecesnt)
-- it should generat the field planctor which is the terra function that 
-- allocates the plan
local C = terralib.includec("stdio.h")

local function compileproblem(tbl,kind)
    
    assert(kind == "gradientdescent")
    local dims = tbl.dims
    local dimindex = { [1] = 0 }
    for i,d in ipairs(dims) do
        assert(Dim:is(d))
        dimindex[d] = i -- index into DimList, 0th entry is always 1  
    end
        
    local costdim = tbl.cost.dim
    local gradtyp = tbl.gradient:gettype()
    local unknown = gradtyp.parameters[3] -- 3rd argument is the image that is the unknown we are mapping over
    
    local graddims = {unknown.metamethods.W,unknown.metamethods.H}
    
    local struct PlanData(S.Object) {
        plan : opt.Plan
        dims : int64[#dims+1]
    }
    
    local function emitcalltouserfunction(mapdims,actualdims,images,userfn)
        local typ = userfn:gettype()
        local di,dj = dimindex[mapdims[1]],dimindex[mapdims[2]]
        assert(di and dj)
        local imagewrappers = terralib.newlist()
        for i = 3,#typ.parameters do
            local imgtyp = typ.parameters[i]
            local W,H = imgtyp.metamethods.W,imgtyp.metamethods.H
            W,H = dimindex[W],dimindex[H]
            assert(W and H)
            imagewrappers:insert(`imgtyp { R = actualdims[W], C = actualdims[H], impl = @images[i - 3] })
        end
        return `userfn(actualdims[di],actualdims[dj],imagewrappers)
    end
    
    local terra impl(data_ : &opaque, images : &&opt.ImageBinding)
        var data = [&PlanData](data_)
        var dims = data.dims
        [ emitcalltouserfunction(costdim,dims,images,tbl.cost.fn) ]
        [ emitcalltouserfunction(graddims,dims,images,tbl.gradient) ]
    end
    local terra planctor(actualdims : &uint64) : &opt.Plan
        var pd = PlanData.alloc()
        pd.plan.data = pd
        pd.plan.impl = impl
        pd.dims[0] = 1
        for i = 0,[#dims] do
            pd.dims[i+1] = actualdims[i]
        end
        return &pd.plan
    end
    return Problem:new { planctor = planctor }
end

function opt.ProblemDefineFromTable(tbl,kind)
    local p = compileproblem(tbl,kind)
    -- store each problem in a table, and assign it an id
    problems[#problems+1] = p
    p.id = #problems
    return p
end
local function problemdefine(filename,kind,pt)
    local success,p = xpcall(function() 
        filename,kind = ffi.string(filename), ffi.string(kind)
        local tbl = assert(terralib.loadfile(filename))() 
        assert(type(tbl) == "table")
        return opt.ProblemDefineFromTable(tbl,kind)
    end,function(err) return debug.traceback(err,2) end)
    if not success then error(p,0) end
    pt.id,pt.planctor = p.id,p.planctor:getpointer()
end

struct opt.Plan(S.Object) {
    impl : {&opaque,&&opt.ImageBinding} -> {}
    data : &opaque
} 

struct opt.Problem(S.Object) {
    id : int
    planctor : &uint64 -> &opt.Plan
}

terra opt.ProblemDefine(filename : rawstring, kind : rawstring)
    var pt = opt.Problem.alloc()
    problemdefine(filename,kind,pt)
    return pt
end 

terra opt.ProblemDelete(p : &opt.Problem)
    -- TODO: deallocate any problem state here (remove from the Lua table,etc.)
    p:delete() 
end

function opt.Dim(name)
    return Dim:new { name = name }
end

local newimage = terralib.memoize(function(typ,W,H)
   local struct Image {
        impl : opt.ImageBinding
        R : uint64
        C : uint64
   }
   function Image.metamethods.__tostring()
      return string.format("Image(%s,%s,%s)",tostring(typ),W.name, H.name)
   end
   Image.metamethods.__apply = macro(function(self,x,y)
     return `@[&typ](self.impl:get(x,y))
   end)
   Image.metamethods.W,Image.metamethods.H = W,H
   return Image
end)

function opt.Image(typ,W,H)
    assert(W == 1 or Dim:is(W))
    assert(H == 1 or Dim:is(H))
    assert(terralib.types.istype(typ))
    return newimage(typ,W,H)
end

terra opt.ProblemPlan(problem : &opt.Problem, dims : &uint64) : &opt.Plan
    return problem.planctor(dims) -- this is just a wrapper around calling the plan constructor
end 

terra opt.PlanFree(plan : &opt.Plan)
    -- TODO: plan should also have a free implementation
    plan:delete()
end

terra opt.ProblemSolve(plan : &opt.Plan, images : &&opt.ImageBinding)
    return plan.impl(plan.data,images)
end

return opt