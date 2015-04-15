
--terralib.settypeerrordebugcallback( function(fn) fn:printpretty() end )

opt = {} --anchor it in global namespace, otherwise it can be collected
local S = require("std")

local util = require("util")
local solversCPU = require("solversCPU")
local solversGPU = require("solversGPU")

local C = util.C

-- constants
local verboseSolver = false

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

local vprintf = terralib.externfunction("cudart:vprintf", {&int8,&int8} -> int)

local function createbuffer(args)
    local Buf = terralib.types.newstruct()
    return quote
        var buf : Buf
        escape
            for i,e in ipairs(args) do
                local typ = e:gettype()
                local field = "_"..tonumber(i)
                typ = typ == float and double or typ
                table.insert(Buf.entries,{field,typ})
                emit quote
                   buf.[field] = e
                end
            end
        end
    in
        [&int8](&buf)
    end
end

printf = macro(function(fmt,...)
    local buf = createbuffer({...})
    return `vprintf(fmt,buf) 
end)

if verboseSolver then
	log = macro(function(fmt,...)
		local args = {...}
		return `C.printf(fmt, args)
	end)
else
	log = function(fmt,...)
	
	end
end

local GPUBlockDims = {{"blockIdx","ctaid"},
              {"gridDim","nctaid"},
              {"threadIdx","tid"},
              {"blockDim","ntid"}}
for i,d in ipairs(GPUBlockDims) do
    local a,b = unpack(d)
    local tbl = {}
    for i,v in ipairs {"x","y","z" } do
        local fn = cudalib["nvvm_read_ptx_sreg_"..b.."_"..v] 
        tbl[v] = `fn()
    end
    _G[a] = tbl
end

__syncthreads = cudalib.nvvm_barrier0

local Problem = newclass("Problem")
local Dim = newclass("dimension")


local ffi = require('ffi')

local problems = {}

-- this function should do anything it needs to compile an optimizer defined
-- using the functions in tbl, using the optimizer 'kind' (e.g. kind = gradientdecesnt)
-- it should generate the field makePlan which is the terra function that 
-- allocates the plan

local function compileProblem(tbl, kind)
	local vars = {
		dims = tbl.dims,
		dimIndex = { [1] = 0 },
		costDim = tbl.cost.dim,
		costType = tbl.cost.fn:gettype()
	}
	
	vars.unknownType = vars.costType.parameters[3] -- 3rd argument is the image that is the unknown we are mapping over
	vars.argumentTypes = terralib.newlist()
	vars.gradientDim = { vars.unknownType.metamethods.W, vars.unknownType.metamethods.H }
		
    for i, d in ipairs(vars.dims) do
        assert(Dim:is(d))
        vars.dimIndex[d] = i -- index into DimList, 0th entry is always 1
    end

	for i = 3,#vars.costType.parameters do
		vars.argumentTypes:insert(vars.costType.parameters[i])
	end
	
	vars.imagesAll = vars.argumentTypes:map(symbol)
	vars.unknownImage = vars.imagesAll[1]
	vars.dataImages = terralib.newlist()
	for i = 2,#vars.imagesAll do
		vars.dataImages:insert(vars.imagesAll[i])
	end
	
	vars.gradWIndex = vars.dimIndex[ vars.gradientDim[1] ]
	vars.gradHIndex = vars.dimIndex[ vars.gradientDim[2] ]
	
	vars.costWIndex = vars.dimIndex[ vars.costDim[1] ]
	vars.costHIndex = vars.dimIndex[ vars.costDim[2] ]

    if kind == "gradientDescentCPU" then
        return solversCPU.gradientDescentCPU(Problem, tbl, vars)
	elseif kind == "gradientDescentGPU" then
		return solversGPU.gradientDescentGPU(Problem, tbl, vars)
	elseif kind == "conjugateGradientCPU" then
		return solversCPU.conjugateGradientCPU(Problem, tbl, vars)
	elseif kind == "linearizedConjugateGradientCPU" then
		return solversCPU.linearizedConjugateGradientCPU(Problem, tbl, vars)
	elseif kind == "linearizedConjugateGradientGPU" then
		return solversGPU.linearizedConjugateGradientGPU(Problem, tbl, vars)
	elseif kind == "linearizedPreconditionedConjugateGradientCPU" then
		return solversCPU.linearizedPreconditionedConjugateGradientCPU(Problem, tbl, vars)
	end
    
end

function opt.ProblemDefineFromTable(tbl, kind, params)
    local p = compileProblem(tbl,kind,params)
    -- store each problem in a table, and assign it an id
    problems[#problems + 1] = p
    p.id = #problems
    return p
end

local function problemDefine(filename, kind, params, pt)
    pt.makePlan = nil
    local success,p = xpcall(function() 
        filename,kind = ffi.string(filename), ffi.string(kind)
        local tbl = assert(terralib.loadfile(filename))() 
        assert(type(tbl) == "table")
        local p = opt.ProblemDefineFromTable(tbl,kind,params)
		pt.id, pt.makePlan = p.id,p.makePlan:getpointer()
		return p
    end,function(err) print(debug.traceback(err,2)) end)
end

struct opt.GradientDescentPlanParams {
    nIterations : uint64
}

struct opt.ImageBinding(S.Object) {
    data : &uint8
    stride : uint64
    elemsize : uint64
}
terra opt.ImageBinding:get(x : uint64, y : uint64) : &uint8
   return self.data + y * self.stride + x * self.elemsize
end

terra opt.ImageBind(data : &opaque, elemsize : uint64, stride : uint64) : &opt.ImageBinding
    var img = opt.ImageBinding.alloc()
    img.data,img.stride,img.elemsize = [&uint8](data),stride,elemsize
    return img
end
terra opt.ImageFree(img : &opt.ImageBinding)
    img:delete()
end

struct opt.Plan(S.Object) {
    impl : {&opaque,&&opt.ImageBinding,&opaque} -> {}
    data : &opaque
} 

struct opt.Problem(S.Object) {
    id : int
    makePlan : &uint64 -> &opt.Plan
}

terra opt.ProblemDefine(filename : rawstring, kind : rawstring, params : &opaque)
    var pt = opt.Problem.alloc()
    problemDefine(filename,kind,params,pt)
	if pt.makePlan == nil then
		pt:delete()
		return nil
	end
    return pt
end 

terra opt.ProblemDelete(p : &opt.Problem)
    -- TODO: deallocate any problem state here (remove from the Lua table,etc.)
    p:delete() 
end

function opt.Dim(name)
    return Dim:new { name = name }
end

local newImage = terralib.memoize(function(typ, W, H)
	local struct Image {
		impl : opt.ImageBinding
		W : uint64
		H : uint64
	}
	function Image.metamethods.__tostring()
	  return string.format("Image(%s,%s,%s)",tostring(typ),W.name, H.name)
	end
	Image.metamethods.__apply = macro(function(self, x, y)
	 return `@[&typ](self.impl:get(x,y))
	end)
	terra Image:inbounds(x : int64, y : int64)
	    return x >= 0 and y >= 0 and x < self.W and y < self.H
	end
	--Image.methods.inbounds:disas()
	terra Image:get(x : int64, y : int64)
	    var v : typ = 0.f --TODO:only works for single precision things
	    var b = self:inbounds(x,y)
	    if b then
	        v = self(x,y)
	    end
	    return v,float(b)
	end
	terra Image:initCPU(actualW : int, actualH : int)
		self.W = actualW
		self.H = actualH
		self.impl.data = [&uint8](C.malloc(actualW * actualH * sizeof(typ)))
		self.impl.elemsize = sizeof(typ)
		self.impl.stride = actualW * sizeof(typ)
		
		for h = 0, self.H do
			for w = 0, self.W do
				self(w, h) = 0.0
			end
		end
	end
	terra Image:initGPU(actualW : int, actualH : int)
		self.W = actualW
		self.H = actualH
		var typeSize = sizeof(typ)
		var cudaError = C.cudaMalloc([&&opaque](&(self.impl.data)), actualW * actualH * typeSize)
		cudaError = C.cudaMemset([&opaque](self.impl.data), 0, actualW * actualH * typeSize)
		self.impl.elemsize = typeSize
		self.impl.stride = actualW * typeSize
	end
	terra Image:debugGPUPrint()
		
		--var cpuImage : type(self)
		--cpuImage:initCPU(self.W, self.H)
		
		--C.cudaMemcpy(dataGPU, cpuImage, sizeof(float) * dimX * dimY, cudaMemcpyHostToDevice)
	end
	Image.metamethods.typ,Image.metamethods.W,Image.metamethods.H = typ, W, H
	return Image
end)

function opt.Image(typ, W, H)
    assert(W == 1 or Dim:is(W))
    assert(H == 1 or Dim:is(H))
    assert(terralib.types.istype(typ))
    return newImage(typ, W, H)
end

terra opt.ProblemPlan(problem : &opt.Problem, dims : &uint64) : &opt.Plan
	return problem.makePlan(dims) -- this is just a wrapper around calling the plan constructor
end 

terra opt.PlanFree(plan : &opt.Plan)
    -- TODO: plan should also have a free implementation
    plan:delete()
end

terra opt.ProblemSolve(plan : &opt.Plan, images : &&opt.ImageBinding, params : &opaque)
	return plan.impl(plan.data, images, params)
end

ad = require("ad")


local ImageTable = newclass("ImageTable") -- a context that keeps a mapping from image accesses im(0,-1) to the ad variable object that represents the access

local ImageAccess = newclass("ImageAccess")

ImageAccess.get = terralib.memoize(function(self,im,field,x,y)
    return ImageAccess:new { image = im, field = field, x = x, y = y}
end)

function ImageAccess:__tostring()
    local xn,yn = tostring(self.x):gsub("-","m"),tostring(self.y):gsub("-","m")
    return ("%s_%s_%s_%s"):format(self.image.name,self.field,xn,yn)
end

function ImageTable:create()
    return self:new { accesstovarid_ = {}, 
                      varidtoaccess_ = terralib.newlist {},
                      imagetoaccesses = {} }
end

function ImageTable:accesstovarid(a) -- a is an ImageAccess object
    assert(ImageAccess:is(a))
    if not self.accesstovarid_[a] then
        self.varidtoaccess_:insert(a)
        self.accesstovarid_[a] = #self.varidtoaccess_
        assert(self.imagetoaccesses[a.image],"using an image not declared as part of the energy function "..a.image.name)
        self.imagetoaccesses[a.image]:insert(a)
    end
    return self.accesstovarid_[a]
end
function ImageTable:varidtoaccess(id)
    return assert(self.varidtoaccess_[assert(tonumber(id),"not a number?")])
end

function ImageTable:accesses(im) return assert(self.imagetoaccesses[im]) end
function ImageTable:addimage(im)
    assert(not self.imagetoaccesses[im])
    self.imagetoaccesses[im] = terralib.newlist()
end

local globalimagetable = ImageTable:create({"","_boundary"}) -- this is used when expressions are create
                                              -- when we scatter-to-gather we make a smaller table with the accesses needed
                                              -- for the specific problem 

local Image = newclass("Image")
-- Z: this will eventually be opt.Image, but that is currently used by our direct methods
-- so this is going in the ad table for now
function ad.Image(name,W,H)
    assert(W == 1 or Dim:is(W))
    assert(H == 1 or Dim:is(H))
    local im = Image:new { name = tostring(name), W = W, H = H }
    globalimagetable:addimage(im)
    return im
end

function Image:inbounds(x,y)
    x,y = assert(tonumber(x)),assert(tonumber(y))
    return ad.v[globalimagetable:accesstovarid(ImageAccess:get(self,"bounds",x,y))]
end
function Image:__call(x,y)
    x,y = assert(tonumber(x)),assert(tonumber(y))
    return ad.v[globalimagetable:accesstovarid(ImageAccess:get(self,"v",x,y))]
end

local function usedvariables(exp)
    local uses = terralib.newlist()
    local seen = {}
    local function visit(e)
        if seen[e] then return end
        seen[e] = true
        if ad.Var:is(e) then uses:insert(e) end
        for i,c in ipairs(e:children()) do visit(c) end
    end
    visit(exp)
    return uses
end

local function centerexp(fromtable,totable,names,access,exp)
    local x,y = -access.x,access.y
    local function shift(a)
        return ImageAccess:get(a.image,a.field,a.x + x, a.y + y)
    end
    local renames = {}
    for i,v in ipairs(usedvariables(exp)) do
        local sa = shift(fromtable:varidtoaccess(v:N()))
        local nv = totable:accesstovarid(sa)
        names[nv] = names[nv] or tostring(sa)
        renames[v:N()] = ad.v[nv]
    end
    return exp:rename(renames)
end 

local function createfunction(images,imagetable,exp,varnames,debug)
    local syms = images:map(function(im) return symbol(opt.Image(float,im.W,im.H),im.name) end)
    local i,j = symbol(int64,"i"), symbol(int64,"j")
    local stmts = terralib.newlist()
    local accesssyms = {}
    local vartosym = {}
    for imageidx,im in ipairs(images) do
        local accesses = imagetable:accesses(im)
        for _,a in ipairs(accesses) do
            local both = ImageAccess:get(a.image,"",a.x,a.y)
            if not accesssyms[both] then
                local va,ga = ImageAccess:get(a.image,"v",a.x,a.y),ImageAccess:get(a.image,"bounds",a.x,a.y)
                local r = { v = symbol(float,tostring(va)), bounds = symbol(float,tostring(ga)) }
                stmts:insert quote    
                    var [r.v],[r.bounds] = [syms[imageidx]]:get(i+[a.x],j+[a.y])
                    if i < 4 and j < 4 then
                        --C.printf("%s(%d + %d,%d + %d) = %f,%f\n",[a.image.name],i,[a.x],j,[a.y],[r.v],[r.bounds])
                    end
                end
                accesssyms[both] = r
            end
            vartosym[imagetable:accesstovarid(a)] = assert(accesssyms[both][a.field])
        end
    end
    local terrafn = ad.toterra({exp},nil,varnames)
    --terrafn:printpretty(true,false)
    local terra generatedfn([i] : int64, [j] : int64, [syms])
        [stmts]
        var r : float
        terrafn(&r,[vartosym])
        return r
    end
    --generatedfn:printpretty(false)
    --generatedfn:printpretty(true,false)
    --generatedfn:disas()
    return generatedfn
end
function ad.Cost(dims,images,costexp)
    images = terralib.islist(images) and images or terralib.newlist(images)
    assert(#images > 0)
    --TODO: check that Dims used in images are in dims list
    --TODO: check images are all images 
    costexp = assert(ad.toexp(costexp))
    --TODO: check all image uses in costexp are bound to images in list
    local unknown = images[1] -- assume for now that the first image is the unknown
    
    
    -- register the input images with the mappings from image -> uses
    local costtable = ImageTable:create()
    local gradtable = ImageTable:create()
    for i,im in ipairs(images) do
        costtable:addimage(im)
        gradtable:addimage(im)
    end
    
    
    -- collect the actually used variables and remap them to only local uses
    local unknownvars = terralib.newlist()
    local unknownvarnames = terralib.newlist()
    
    local costvarnames = terralib.newlist()
    
    local globaltocost = {}
    for i,c in ipairs(usedvariables(costexp)) do
        local a = globalimagetable:varidtoaccess(c:N())
        local cv = ad.v[costtable:accesstovarid(a)]
        globaltocost[c:N()] = cv 
        costvarnames:insert(tostring(a))
        if a.image == unknown and a.field == "v" then
            unknownvars:insert(cv)
            unknownvarnames:insert(tostring(a))
        end
    end
    
    costexp = costexp:rename(globaltocost)
    local gradient = costexp:gradient(unknownvars)
    
    print("cost expression")
    print(ad.tostrings({assert(costexp)}, costvarnames))
    print("grad expression")
    print(table.concat(unknownvarnames,", ").." = ",
          ad.tostrings(gradient, costvarnames))
    
    local gradientgathered = 0
    local gradnames = terralib.newlist()
    for i,u in ipairs(unknownvars) do
        local g = gradient[i]
        local a = costtable:varidtoaccess(u:N())
        gradientgathered = gradientgathered + centerexp(costtable,gradtable,gradnames,a,g)
    end
    
    print("grad gather")
    print(ad.tostrings({gradientgathered},gradnames))
    
    local costfn = createfunction(images,costtable,costexp,costvarnames,false)
    local gradfn = createfunction(images,gradtable,gradientgathered,gradnames,true)
    return { dims = dims, cost = { dim = dims, fn = costfn}, gradient = gradfn }
end
return opt