
--terralib.settypeerrordebugcallback( function(fn) fn:printpretty() end )

opt = {} --anchor it in global namespace, otherwise it can be collected
local S = require("std")

local util = require("util")
local solversCPU = require("solversCPU")
local solversGPU = require("solversGPU")

local C = util.C

-- constants
local verboseFixThis = false
local verboseSolver = false
local verboseAD = false

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
local dprint
if verboseFixThis then
	solverLog = macro(function(fmt,...)
		local args = {...}
		return `C.printf(fmt, args)
	end)
else
	solverLog = function(fmt,...)
	
	end
end

if verboseAD then
	log = macro(function(fmt,...)
		local args = {...}
		return `C.printf(fmt, args)
	end)
	dprint = print
else
	log = function(fmt,...)
	
	end
	dprint = function() end
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
		dimensions = tbl.dimensions,
		dimIndex = { [1] = 0 },
		costDim = tbl.cost.dimensions,
		costType = tbl.cost.boundary:gettype()
	}
	
	vars.unknownType = vars.costType.parameters[3] -- 3rd argument is the image that is the unknown we are mapping over
	vars.argumentTypes = terralib.newlist()
	vars.gradientDim = { vars.unknownType.metamethods.W, vars.unknownType.metamethods.H }
		
    for i, d in ipairs(vars.dimensions) do
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
	elseif kind == "lbfgsCPU" then
		return solversCPU.lbfgsCPU(Problem, tbl, vars)
	elseif kind == "vlbfgsCPU" then
		return solversCPU.vlbfgsCPU(Problem, tbl, vars)
	elseif kind == "vlbfgsGPU" then
		return solversGPU.vlbfgsGPU(Problem, tbl, vars)
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

terra opt.InBoundsCalc(x : int64, y : int64, W : int64, H : int64, sx : int64, sy : int64) : int
    var minx,maxx,miny,maxy = x - sx,x + sx,y - sy,y + sy
    return int(minx >= 0) and int(maxx < W) and int(miny >= 0) and int(maxy < H)
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
	    if opt.InBoundsCalc(x,y,self.W,self.H,0,0) ~= 0 then
	        v = self(x,y)
	    end
	    return v
	end
	terra Image:initCPU(actualW : int, actualH : int)
		self.W = actualW
		self.H = actualH
		self.impl.data = [&uint8](C.malloc(actualW * actualH * sizeof(typ)))
		self.impl.elemsize = sizeof(typ)
		self.impl.stride = actualW * sizeof(typ)
		
		for h = 0, self.H do
			for w = 0, self.W do
				self(w, h) = 0.0f
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

terra opt.ProblemPlan(problem : &opt.Problem, dimensions : &uint64) : &opt.Plan
	return problem.makePlan(dimensions) -- this is just a wrapper around calling the plan constructor
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
local BoundsAccess = newclass("BoundsAccess")

ImageAccess.get = terralib.memoize(function(self,im,field,x,y)
    return ImageAccess:new { image = im, field = field, x = x, y = y}
end)

function ImageAccess:__tostring()
    local xn,yn = tostring(self.x):gsub("-","m"),tostring(self.y):gsub("-","m")
    return ("%s_%s_%s_%s"):format(self.image.name,self.field,xn,yn)
end
function BoundsAccess:__tostring() return ("bounds_%d_%d_%d_%d"):format(self.x,self.y,self.sx,self.sy) end
BoundsAccess.get = terralib.memoize(function(self,x,y,sx,sy)
    return BoundsAccess:new { x = x, y = y, sx = sx, sy = sy }
end)

local Image = newclass("Image")
-- Z: this will eventually be opt.Image, but that is currently used by our direct methods
-- so this is going in the ad table for now
function ad.Image(name,W,H)
    assert(W == 1 or Dim:is(W))
    assert(H == 1 or Dim:is(H))
    return Image:new { name = tostring(name), W = W, H = H }
end

function Image:__call(x,y)
    x,y = assert(tonumber(x)),assert(tonumber(y))
    return ad.v[ImageAccess:get(self,"v",x,y)]
end
function opt.InBounds(sx,sy)
    return ad.v[BoundsAccess:get(0,0,sx,sy)]
end
function BoundsAccess:shift(x,y)
    return BoundsAccess:get(self.x+x,self.y+y,self.sx,self.sy)
end
function ImageAccess:shift(x,y)
    return ImageAccess:get(self.image,self.field,self.x + x, self.y + y)
end
local function centerexp(access,exp)
    local x,y = -access.x,-access.y
    local function rename(a)
        return ad.v[a:shift(x,y)]
    end
    return exp:rename(rename)
end 

local function removeboundaries(exp)
    local function nobounds(a)
        if BoundsAccess:is(a) then return ad.toexp(1)
        else return ad.v[a] end
    end
    return exp:rename(nobounds)
end
local function createfunction(images,exp,usebounds)
    if not usebounds then
        exp = removeboundaries(exp)
    end
    local imageindex = {}
    local imagesyms = terralib.newlist()
    for i,im in ipairs(images) do
        local s = symbol(opt.Image(float,im.W,im.H),im.name)
        imageindex[im] = s
        imagesyms:insert(s)
    end
    local stencil = {0,0}
    
    local unknownimage = imagesyms[1]
    local i,j = symbol(int64,"i"), symbol(int64,"j")
    local stmts = terralib.newlist()
    local accesssyms = {}
    local vartosym = {}
    local function emitvar(a)
        if not accesssyms[a] then
            local r 
            if ImageAccess:is(a) then
                local im = assert(imageindex[a.image],("cost function uses image %s not listed in parameters."):format(a.image))
                r = symbol(float,tostring(a))
                stmts:insert quote
                    var [r] = [ usebounds and (`im:get(i+[a.x],j+[a.y])) or (`im(i+[a.x],j+[a.y])) ]
                    --if i < 4 and j < 4 then
                    --    C.printf("%s(%d + %d,%d + %d) = %f,%f\n",[a.image.name],i,[a.x],j,[a.y],[r.v],[r.bounds])
                    --end
                end
                stencil[1] = math.max(stencil[1],math.abs(a.x))
                stencil[2] = math.max(stencil[2],math.abs(a.y))
            else --bounds calculation
                assert(usebounds) -- if we removed them, we shouldn't see any boundary accesses
                r = symbol(int,tostring(a))
                stmts:insert quote
                    var [r] = opt.InBoundsCalc(i+a.x,j+a.y,unknownimage.W,unknownimage.H,a.sx,a.sy)
                end
                stencil[1] = math.max(stencil[1],math.abs(a.x)+a.sx)
                stencil[2] = math.max(stencil[2],math.abs(a.y)+a.sy)
            end
            accesssyms[a] = r
        end
        return accesssyms[a]
    end
    local result = ad.toterra({exp},emitvar)
    local terra generatedfn([i] : int64, [j] : int64, [imagesyms])
        [stmts]
        return result
    end
    --generatedfn:printpretty(false,false)
    generatedfn:compile()
    --generatedfn:printpretty(true,false)
    if verboseSolver then
        generatedfn:disas()
    end
    return generatedfn,stencil
end
local function createfunctionset(images,exp)
    dprint("bound")
    local boundary,stencil = createfunction(images,exp,true)
    dprint("interior")
    local interior = createfunction(images,exp,false)
    return { boundary = boundary, stencil = stencil, interior = interior }
end

function ad.Cost(dimensions,images,costexp)
    images = terralib.islist(images) and images or terralib.newlist(images)
    assert(#images > 0)
    --TODO: check that Dims used in images are in dimensions list
    --TODO: check images are all images 
    costexp = assert(ad.toexp(costexp))
    --TODO: check all image uses in costexp are bound to images in list
    local unknown = images[1] -- assume for now that the first image is the unknown
    
    -- collect the actually used variables and remap them to only local uses
    local seenunknown = {}
    local unknownvars = terralib.newlist()
    
    costexp:rename(function(a)
        local v = ad.v[a]
        --assert(<image is in images list>)
        if ImageAccess:is(a) and a.image == unknown and a.field == "v" and not seenunknown[a] then
            unknownvars:insert(v)
            seenunknown[a] = true
        end
        return v
    end)
    
    local gradient = costexp:gradient(unknownvars)
    
    dprint("cost expression")
    dprint(ad.tostrings({assert(costexp)}))
    dprint("grad expression")
    local names = table.concat(unknownvars:map(function(v) return tostring(v:key()) end),", ")
    dprint(names.." = "..ad.tostrings(gradient))
    
    local gradientgathered = 0
    for i,u in ipairs(unknownvars) do
        gradientgathered = gradientgathered + centerexp(u:key(),gradient[i])
    end
    
    dprint("grad gather")
    dprint(ad.tostrings({gradientgathered}))
    
    dprint("cost")
    local cost = createfunctionset(images,costexp)
    cost.dimensions = dimensions
    dprint("grad")
    local gradient = createfunctionset(images,gradientgathered)
    local r = { dimensions = dimensions, cost = cost, gradient = gradient }
    if verboseSolver then
        terralib.tree.printraw(r)
    end
    return r
end
return opt