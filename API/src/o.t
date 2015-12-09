--terralib.settypeerrordebugcallback( function(fn) fn:printpretty() end )

opt = {} --anchor it in global namespace, otherwise it can be collected
local S = require("std")
local ffi = require("ffi")
local util = require("util")
local solversCPU = require("solversCPU")
local solversGPU = require("solversGPU")

local C = util.C

if false then
    local fileHandle = C.fopen("crap.txt", 'w')
    C._close(1)
    C._dup2(C._fileno(fileHandle), 1)
    C._close(2)
    C._dup2(C._fileno(fileHandle), 2)
end

-- constants
local verboseSolver = true
local verboseAD = true

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

local vprintfname = ffi.os == "Windows" and "vprintf" or "cudart:vprintf"
local vprintf = terralib.externfunction(vprintfname, {&int8,&int8} -> int)

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

if verboseSolver then
	logSolver = macro(function(fmt,...)
		local args = {...}
		return `C.printf(fmt, args)
	end)
else
	logSolver = macro(function(fmt,...)
		return 0
	end)
end

if verboseAD then
	logAD = macro(function(fmt,...)
		local args = {...}
		return `C.printf(fmt, args)
	end)
	dprint = print
else
	logAD = macro(function(fmt,...)
		return 0
	end)
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

local Dim = newclass("dimension")


local ffi = require('ffi')

local problems = {}

-- this function should do anything it needs to compile an optimizer defined
-- using the functions in tbl, using the optimizer 'kind' (e.g. kind = gradientdecesnt)
-- it should generate the field makePlan which is the terra function that 
-- allocates the plan

local function compilePlan(problemSpec, kind, params)
    assert(kind == "gaussNewtonGPU","expected solver kind to be gaussNewtonGPU")
    return solversGPU.gaussNewtonGPU(problemSpec)
end

struct opt.GradientDescentPlanParams {
    nIterations : uint32
}

struct opt.Plan(S.Object) {
    init : {&opaque,&&opaque,&int32,&&opaque,&&int32,&&int32,&&opaque,&&opaque} -> {}
    step : {&opaque,&&opaque,&int32,&&opaque,&&int32,&&int32,&&opaque,&&opaque} -> int
    data : &opaque
} 

struct opt.Problem {} -- just used as an opaque type, pointers are actually just the ID
local function problemDefine(filename, kind, params, pid)
    local problemmetadata = { filename = ffi.string(filename), kind = ffi.string(kind), params = params, id = #problems + 1 }
    problems[problemmetadata.id] = problemmetadata
    pid[0] = problemmetadata.id
end
-- define just stores meta-data right now. ProblemPlan does all compilation for now
terra opt.ProblemDefine(filename : rawstring, kind : rawstring, params : &opaque)
    var id : int
    problemDefine(filename, kind, params,&id)
    return [&opt.Problem](id)
end 
terra opt.ProblemDelete(p : &opt.Problem)
    var id = int64(p)
    --TODO: remove from problem table
end

local ProblemSpec = newclass("ProblemSpec")
opt.PSpec = ProblemSpec
local PROBLEM_STAGES  = { inputs = 0, functions = 1 }
function opt.ProblemSpec()
    local BlockedProblemParameters = terralib.types.newstruct("BlockedProblemParameters")
	local problemSpec = ProblemSpec:new { 
	                         shouldblock = opt.problemkind:match("Block") or false,
                             parameters = terralib.newlist(),-- listing of each parameter, {name = <string>, kind = <image|graph|edgevalue>, idx = <number>, type = <thetypeusedtostoreit> }
                             names = {}, -- name -> index in parameters list
                             ProblemParameters = terralib.types.newstruct("ProblemParameters"),
                             BlockedProblemParameters = BlockedProblemParameters,
							 functions = {},
							 maxStencil = 0,
							 stage = "inputs",
							 usepreconditioner = false,
                           }
	function BlockedProblemParameters.metamethods.__getentries(self)
		local entries = {}
		for i,p in ipairs(problemSpec.parameters) do
			if p.kind ~= "image" then
				entries[i] = {p.name,p.type}
			else
				entries[i] = {p.name,problemSpec:BlockedTypeForImageEntry(p)}
			end
		end
		return entries
	end
	return problemSpec
end

function ProblemSpec:UsePreconditioner(v)
	self:Stage "inputs"
	self.usepreconditioner = v
end
function ProblemSpec:Stage(name)
    assert(PROBLEM_STAGES[self.stage] <= PROBLEM_STAGES[name], "all inputs must be specified before functions are added")
    self.stage = name
end

function ProblemSpec:toname(name)
    name = assert(tostring(name))
    assert(not self.names[name],string.format("name %s already in use",name))
    self.names[name] = #self.parameters + 1
    return name
end

local newImage 

function ProblemSpec:MaxStencil()
    self:Stage "functions"
	return self.maxStencil
end

function ProblemSpec:Stencil(stencil) 
    self:Stage "inputs"
	self.maxStencil = math.max(stencil, self.maxStencil)
end


function ProblemSpec:BlockSize()
    self:Stage "functions"
	--TODO: compute based on problem
	--return opt.BLOCK_SIZE
	return 16
end

function ProblemSpec:BlockStride() 
    self:Stage "functions"
    return 2*self:MaxStencil() + self:BlockSize() 
end

function ProblemSpec:BlockedTypeForImage(W,H,typ)
    self:Stage "functions"
	local elemsize = terralib.sizeof(assert(typ))
	return newImage(typ, W, H, elemsize, elemsize*self:BlockStride())
end
function ProblemSpec:BlockedTypeForImageEntry(p)
    self:Stage "functions"
	local mm = p.type.metamethods
	return self:BlockedTypeForImage(mm.W,mm.H,mm.typ)
end

function ProblemSpec:newparameter(name,kind,idx,typ)
    self.parameters:insert { name = self:toname(name), kind = kind, idx = idx, type = typ }
	self.ProblemParameters.entries:insert { name, typ }
end

function ProblemSpec:ParameterType(blocked) 
	if blocked == nil then
		blocked = self.shouldblock
	end
	return blocked and self.BlockedProblemParameters or  self.ProblemParameters
end
function ProblemSpec:UnknownType(blocked)
    self:Stage "functions"
	return self:TypeOf("X",blocked) 
end
function ProblemSpec:UnknownArrayType(blocked)
	local ut = self:UnknownType(blocked)
	local mm = ut.metamethods
	local typ = mm.typ:isarray() and mm.typ or mm.typ[1]
	return newImage(typ, mm.W, mm.H, mm.elemsize, mm.stride)
end

function ProblemSpec:TypeOf(name,blocked)
    self:Stage "functions"
	if blocked == nil then
		blocked = self.shouldblock
	end 
	local p = self.parameters[assert(self.names[name],"unknown name: " .. name)] 
	return blocked and self:BlockedTypeForImageEntry(p) or p.type
end

function ProblemSpec:Function(name,unknownfunction, ...)
    self:Stage "functions"
    unknownfunction:gettype() -- check this typechecks
    local graphfunctions = terralib.newlist()
    for i = 1,select("#",...),2 do
        local graphname, implementation =  select(i,...)
        implementation:gettype()
        graphfunctions:insert { graph = graphname, implementation = implementation }
    end
    self.functions[name] = { name = name, unknownfunction = unknownfunction, graphfunctions = graphfunctions }
end

function ProblemSpec:Param(name,typ,idx)
    self:Stage "inputs"
    self:newparameter(name,"param",idx,typ)
end

function ProblemSpec:EvalExclude(...)
    local args = {...}
    if self.functions.exclude then
        return `bool(self.functions.exclude.boundary(args))
    else
        return `false
    end
end

local newDim = terralib.memoize(function(name,size,idx)
	return Dim:new { name = name, size = size, _index = idx }
end)

function opt.Dim(name, idx)
    idx = assert(tonumber(idx), "expected an index for this dimension")
    local size = tonumber(opt.dimensions[idx])
    return newDim(name,size,idx)
end

function opt.InternalDim(name, size)
	size = assert(tonumber(size), "expected a number for size")
    return Dim:new { name = name, size = size }
end

opt.InBoundsCalc = macro(function(x,y,W,H,sx,sy)
    return `x >= sx and x < W - sx and y >= sy and y < H - sy
end)
	
newImage = terralib.memoize(function(typ, W, H, elemsize, stride)
	local struct Image {
		data : &uint8
	}
	function Image.metamethods.__typename()
	  return string.format("Image(%s,%s,%s,%d,%d)",tostring(typ),W.name, H.name,elemsize,stride)
	end

	if ad.isterravectortype(typ) and typ.metamethods.type == float and (typ.metamethods.N == 4 or typ.metamethods.N == 2) then
	    -- emit code that will produce special CUDA vector load instructions
	    local storetype = vector(float,typ.metamethods.N)
	    terra Image.metamethods.__apply(self : &Image, x : int32, y : int32)
            var a = @[&storetype](self.data + y*stride + x*elemsize)
            return @[&typ](&a)
        end
        terra Image.metamethods.__update(self : &Image, x : int32, y : int32, v : typ)
            @[&storetype](self.data + y*stride + x*elemsize) = @[&storetype](&v)
        end
	else	
        terra Image.metamethods.__apply(self : &Image, x : int32, y : int32)
            return @[&typ](self.data + y*stride + x*elemsize)
        end
        terra Image.metamethods.__update(self : &Image, x : int32, y : int32, v : typ)
            @[&typ](self.data + y*stride + x*elemsize) = v
        end
    end
	terra Image:inbounds(x : int32, y : int32)
	    return x >= 0 and y >= 0 and x < W.size and y < H.size
	end
	if ad.isterravectortype(typ) then
	    terra Image:atomicAdd(x : int32, y : int32, v : typ)
	        escape
	            for i = 0,typ.metamethods.N - 1 do
	                emit quote
	                    util.atomicAdd( &(@[&typ](self.data + y*stride + x*elemsize))(i) ,v(i))
	                end
	            end
	        end
	    end
	else
	    terra Image:atomicAdd(x : int32, y : int32, v : typ)
	        util.atomicAdd([&typ](self.data + y*stride + x*elemsize),v)
	    end
	end
	Image.methods.get = macro(function(self,x,y,gx,gy)
		if not gx then
		    gx,gy = x,y
		end
		return quote
            var v : typ = 0.f
            if opt.InBoundsCalc(gx,gy,W.size,H.size,0,0) then
                v = self(x,y)
            end
        in v end
	end)
	--terra Image:get(x : int32, y : int32) : typ return self:get(x,y,x,y) end
	terra Image:H() return H.size end
	terra Image:W() return W.size end
	terra Image:elemsize() return elemsize end
	terra Image:stride() return stride end
	terra Image:initCPU()
		self.data = [&uint8](C.malloc(stride*H.size))
		for h = 0, H.size do
			for w = 0, W.size do
				self(w, h) = 0.0f
			end
		end
	end
	terra Image:initGPU()
		var cudaError = C.cudaMalloc([&&opaque](&(self.data)), stride*H.size)
		cudaError = C.cudaMemset([&opaque](self.data), 0, stride*H.size)
	end
	local mm = Image.metamethods
	mm.typ,mm.W,mm.H,mm.elemsize,mm.stride = typ,W,H,elemsize,stride
	return Image
end)


local unity = Dim:new { name = "1", size = 1 }
local function todim(d)
    return Dim:is(d) and d or d == 1 and unity
end

local function tovalidimagetype(typ)
    if not terralib.types.istype(typ) then return nil end
    if ad.isterravectortype(typ) then
        return typ, typ.metamethods.N
    elseif typ:isarithmetic() then
        return typ, 1
    end
end

function ProblemSpec:Image(name,typ,W,H,idx)
    self:Stage "inputs"
    typ = assert(tovalidimagetype(typ,"expected a number or an array of numbers"))

    local elemsize = assert(tonumber(opt.elemsizes[idx]))
    local stride = assert(tonumber(opt.strides[idx]))
    local r = newImage(typ, assert(todim(W)), assert(todim(H)), elemsize, stride)
    self:newparameter(name,"image",idx,r)
end


function ProblemSpec:InternalImage(typ,W,H,blocked)
    self:Stage "functions"
	if blocked == nil then
		blocked = self.shouldblock
	end
	if blocked then
		return self:BlockedTypeForImage(W,H,typ)
	else
		W,H = assert(todim(W)),assert(todim(H))
		assert(terralib.types.istype(typ))
		local elemsize = terralib.sizeof(typ)
		return newImage(typ,W,H,elemsize,elemsize*W.size)
	end
end

function ProblemSpec:Graph(name, idx, ...)
    self:Stage "inputs"
    local GraphType = terralib.types.newstruct(name)
    GraphType.entries:insert ( {"N",int32} )
    
    local mm = GraphType.metamethods
    mm.idx = idx -- the index into the graph size table
    mm.elements = terralib.newlist()
    for i = 1, select("#",...),4 do
        local name,W,H,idx = select(i,...) --TODO: we don't bother to track the dimensions of these things now
        assert(todim(W) and todim(H),"Expected dimensions")
        GraphType.entries:insert ( {name .. "_x", &int32} )
	GraphType.entries:insert ( {name .. "_y", &int32} )
        mm.elements:insert( { name = name, idx = assert(tonumber(idx),"expected a numeric index") } )
    end
    
    self:newparameter(name, "graph", idx, GraphType)
end

local allPlans = terralib.newlist()

errorPrint = rawget(_G,"errorPrint") or print

function opt.problemSpecFromFile(filename)
   local file, errorString = terralib.loadfile(filename)
   if not file then
      error(errorString, 0)
   end
   return file()
end

local function problemPlan(id, dimensions, elemsizes, strides, pplan)
    local success,p = xpcall(function() 
		local problemmetadata = assert(problems[id])
        opt.dimensions,opt.elemsizes,opt.strides = dimensions,elemsizes,strides
        opt.math = problemmetadata.kind:match("GPU") and util.gpuMath or util.cpuMath
        opt.problemkind = problemmetadata.kind
		
        local tbl = opt.problemSpecFromFile(problemmetadata.filename)
        assert(ProblemSpec:is(tbl))
		local result = compilePlan(tbl,problemmetadata.kind,problemmetadata.params)
		allPlans:insert(result)
		pplan[0] = result()
    end,function(err) errorPrint(debug.traceback(err,2)) end)
end

terra opt.ProblemPlan(problem : &opt.Problem, dimensions : &uint32, elemsizes : &uint32, strides : &uint32) : &opt.Plan
	var p : &opt.Plan = nil 
	problemPlan(int(int64(problem)),dimensions,elemsizes,strides,&p)
	return p
end 

terra opt.PlanFree(plan : &opt.Plan)
    -- TODO: plan should also have a free implementation
    plan:delete()
end

terra opt.ProblemInit(plan : &opt.Plan, images : &&opaque, graphsizes : &int32, edgevalues : &&opaque, xs : &&int32, ys : &&int32, params : &&opaque, solverparams : &&opaque) 
    return plan.init(plan.data, images, graphsizes, edgevalues, xs, ys, params, solverparams)
end
terra opt.ProblemStep(plan : &opt.Plan, images : &&opaque, graphsizes : &int32, edgevalues : &&opaque, xs : &&int32, ys : &&int32, params : &&opaque, solverparams : &&opaque) : int
    return plan.step(plan.data, images, graphsizes, edgevalues, xs, ys, params, solverparams)
end
terra opt.ProblemSolve(plan : &opt.Plan, images : &&opaque, graphsizes : &int32, edgevalues : &&opaque, xs : &&int32, ys : &&int32, params : &&opaque, solverparams : &&opaque)
   opt.ProblemInit(plan, images, graphsizes, edgevalues, xs, ys, params, solverparams)
   while opt.ProblemStep(plan, images, graphsizes, edgevalues, xs, ys, params, solverparams) ~= 0 do end
end


ad = require("ad")


local Index = ad.newclass("Index")
local Offset = Index:Variant("Offset") -- an local pixel relative index in an image
local GraphElement = Index:Variant("GraphElement") -- an index looked up from a graph node

function Offset:__tostring() return ("%d_%d"):format(self.x,self.y) end
Offset.get = terralib.memoize(function(self,x,y) assert(type(x) == "number" and type(y) == "number") return Offset:new { x = x, y = y } end)
function GraphElement:__tostring() return ("%s_%s"):format(tostring(self.graph), self.element) end
GraphElement.get = terralib.memoize(function(self,g,e) return GraphElement:new { graph = g, element = e } end)

local VarDef = ad.newclass("VarDef") -- meta-data attached to each ad variable about what actual value it is
local ImageAccess = VarDef:Variant("ImageAccess") -- access into one particular image
local BoundsAccess = VarDef:Variant("BoundsAccess") -- query about the bounds of an image
local IndexValue = VarDef:Variant("IndexValue") -- query of the numeric index
local ParamValue = VarDef:Variant("ParamValue") -- get one of the global parameter values

function ImageAccess:__tostring()
    print("HI")
    local r = ("%s_%s_%s"):format(self.image.name,tostring(self.index),self.channel)
    if self:shape() ~= ad.scalar then
        r = r .. ("_%s"):format(tostring(self:shape()))
    end
    return r
end
function BoundsAccess:__tostring() return ("bounds_%d_%d_%d_%d"):format(self.x,self.y,self.sx,self.sy) end
function IndexValue:__tostring() return ({[0] = "i","j","k"})[self.dim._index] end
function ParamValue:__tostring() return "param_"..self.name end

ImageAccess.get = terralib.memoize(function(self,im,shape,index,channel)
    assert(Index:is(index))
    return ImageAccess:new { image = im, index = index , channel = channel, _shape = shape}
end)

BoundsAccess.get = terralib.memoize(function(self,x,y,sx,sy)
    return BoundsAccess:new { x = x, y = y, sx = sx, sy = sy }
end)
IndexValue.get = terralib.memoize(function(self,dim,shift)
    return IndexValue:new { _shift = tonumber(shift) or 0, dim = assert(todim(dim),"expected a dimension object") } 
end)

function ImageAccess:shape() return self._shape end -- implementing AD's API for keys

function Dim:index() return ad.v[IndexValue:get(self)] end

local SumOfSquares = newclass("SumOfSquares")
function SumOfSquares:__toadexp()
    local sum = 0
    for i,t in ipairs(self.terms) do
        sum = sum + t*t
    end
    return sum
end
function ad.sumsquared(...)
    local exp = terralib.newlist {...}
    exp = exp:map(function(x) return assert(ad.toexp(x), "expected an ad expression") end)
    return SumOfSquares:new { terms = exp }
end
local ProblemSpecAD = newclass("ProblemSpecAD")

function ad.ProblemSpec()
    return ProblemSpecAD:new { P = opt.ProblemSpec(), nametoimage = {} }
end
function ProblemSpecAD:UsePreconditioner(v)
	self.P:UsePreconditioner(v)
end

local Image = newclass("Image")
-- Z: this will eventually be opt.Image, but that is currently used by our direct methods
-- so this is going in the ad table for now
-- small workaround: idx > 0 means it is part of ProblemSpec struct
-- idx < 0 means that it is the -idx-th argument to the function begin generated after the Parameters struct. (.e.g -1 is the first argument) 
function ProblemSpecAD:Image(name,typ,W,H,idx)
    if not terralib.types.istype(typ) then
        typ,W,H,idx = float,typ,W,H --shift arguments left
    end
    assert(W == 1 or Dim:is(W))
    assert(H == 1 or Dim:is(H))
    idx = assert(tonumber(idx))
    if idx >= 0 then
        self.P:Image(name,typ,W,H,idx)
    end
    local typ,N = tovalidimagetype(typ)
    local r = Image:new { name = tostring(name), W = W, H = H, idx = idx, N = N, type = typ }
    self.nametoimage[name] = r
    return r
end

local Graph = newclass("Graph")
function Graph:__tostring() return "< graph" .. self.name .. ">" end

function ProblemSpecAD:Graph(name,...)
    self.P:Graph(name,...)
    return Graph:new { name = tostring(name) }
end

function ProblemSpecAD:Param(name,typ,idx)
    self.P:Param(name,float,idx)
    return ad.v[ParamValue:new { name = name, type = typ }]
end

function Image:__call(x,y,c,extra)
    local shape = ad.scalar
    if false and y == nil and c == nil then
        shape = ad.Shape:fromkeys(x)
        x,y,c = y or 0,c or 0,extra
    end
    x,y,c = assert(tonumber(x)),assert(tonumber(y)),tonumber(c)
    assert(not c or c < self.N, "channel outside of range")
    if self.N == 1 or c then
        return ad.v[ImageAccess:get(self,shape,Offset:get(x,y),c or 0)]
    else
        local r = {}
        for i = 1,self.N do
            r[i] = ad.v[ImageAccess:get(self,shape,Offset:get(x,y),i-1)]
        end
        return ad.Vector(unpack(r))
    end
end
function opt.InBounds(x,y,sx,sy)
	assert(x and y and sx and sy, "InBounds Requires 4 values (x,y,stencil_x,stencil_y)")
    return ad.v[BoundsAccess:get(x,y,sx,sy)]
end
function BoundsAccess:shift(x,y)
    return BoundsAccess:get(self.x+x,self.y+y,self.sx,self.sy)
end
function BoundsAccess:type() return bool end --implementing AD's API for keys
function ImageAccess:shift(x,y)
    assert(Offset:is(self.index), "cannot shift graph accesses!")
    return ImageAccess:get(self.image,self:shape(),Offset:get(self.index.x + x, self.index.y + y),self.channel)
end
function IndexValue:shift(x,y)
    local v = {[0] = x,y}
    return IndexValue:get(self.dim,self._shift + assert(v[self.dim._index]))
end
function ParamValue:shift(x,y) return self end

local function shiftexp(exp,x,y)
    local function rename(a)
        return ad.v[a:shift(x,y)]
    end
    return exp:rename(rename)
end 

local function removeboundaries(exp)
    if ad.ExpVector:is(exp) or terralib.islist(exp) then return exp:map(removeboundaries) end
    local function nobounds(a)
        if BoundsAccess:is(a) and (a.x == 0 and a.y == 0 and a.sx == 0 and a.sy == 0) then return ad.toexp(1)
        else return ad.v[a] end
    end
    return exp:rename(nobounds)
end

-- code ir is a table { kind = "...", ... }    
local IRNode,nextirid = newclass("IRNode"),0
function IRNode:create(body)
    local ir = IRNode:new(body)
    ir.id,nextirid = nextirid,nextirid+1
    assert(body.type and terralib.types.istype(body.type),"missing type")
    if not ir.shape then
        ir.shape = ad.scalar
    end
    return ir
end

local Condition = newclass("Condition")

function Condition:create(members)
    local function cmp(a,b)
        if a.kind == "intrinsic" and b.kind ~= "intrinsic" then return true
        elseif a.kind ~= "intrinsic" and b.kind == "intrinsic" then return false
        else return a.id < b.id end
    end
    table.sort(members,cmp)
    return Condition:new { members = members }
end

function Condition:Intersect(rhs)
    local lhsmap = {}
    for i,m in ipairs(self.members) do
        lhsmap[m] = true
    end
    local r = terralib.newlist()
    for i,m in ipairs(rhs.members) do
        if lhsmap[m] then
            r:insert(m)
        end
    end
    return Condition:create(r)
end

function Condition:Union(rhs)
    local lhsmap = {}
    local r = terralib.newlist()
    for i,m in ipairs(self.members) do
        lhsmap[m] = true
        r:insert(m)
    end
    for i,m in ipairs(rhs.members) do
        if not lhsmap[m] then
            r:insert(m)
        end
    end
    return Condition:create(r)
end

local function createfunction(problemspec,name,exps,usebounds,W,H)
    exps = removeboundaries(exps)
    
    local imageload = terralib.memoize(function(image)
        return IRNode:create { kind = "vectorload", value = image, type = image.image.type, shape = image:shape() }
    end)
    
    local irmap
    
    local function tofloat(ir,exp)
        if ir.type ~= float then
            return `float(exp)
        else
            return exp
        end
    end
    local function createreduce(op,vardecl,n)
        local cond
        if op == "sum" and n.kind == "Apply" and n.op.name == "prod" then
            local conditions = terralib.newlist()
            local factors = terralib.newlist()
            for i,c in ipairs(n:children()) do
                if c:type() == bool then
                    conditions:insert(irmap(c))
                else
                    factors:insert(c)
                end
            end
            n = ad.prod(n.config.c,unpack(factors))
            cond = Condition:create(conditions)
        end
        return IRNode:create { kind = "reduce", op = op, children = terralib.newlist { vardecl, irmap(n) }, condition = cond, type = float, shape = vardecl.shape }
    end
    irmap = terralib.memoize(function(e)
        if ad.ExpVector:is(e) then
            return IRNode:create { kind = "vectorconstruct", children = e.data:map(irmap), type = ad.TerraVector(float,#e.data) }
        elseif "Var" == e.kind then
            local a = e:key()
            if "ImageAccess" == a.kind then
                if not a.image.type:isarithmetic() then
                    local loadvec = imageload(ImageAccess:get(a.image,a:shape(),a.index,0))
                    loadvec.count = (loadvec.count or 0) + 1
                    return IRNode:create { kind = "vectorextract", children = terralib.newlist { loadvec }, channel = a.channel, type = e:type(), shape = a:shape() }  
                else
                    return IRNode:create { kind = "load", value = a, type = e:type(), shape = a:shape() }
                end 
            else
                return IRNode:create { kind = "intrinsic", value = a, type = e:type() }
            end
        elseif "Const" == e.kind then
            return IRNode:create { kind = "const", value = e.v, type = e:type() }
        elseif "Apply" == e.kind then
            if (e.op.name == "sum") and #e:children() > 2 then
                local vardecl = IRNode:create { kind = "vardecl", constant = e.config.c, type = float, shape = e:shape() }
                local children = terralib.newlist { vardecl }
                local varuse = IRNode:create { kind = "varuse", children = children, type = float, shape = e:shape() }
                for i,c in ipairs(e:children()) do
                    children:insert(createreduce(e.op.name,vardecl,c))
                end
                return varuse
            end
            local children = e:children():map(irmap)
            local fn,gen = opt.math[e.op.name]
            if fn then
                function gen(args)
                    local nargs = terralib.newlist()
                    for i,a in ipairs(args) do
                        nargs[i] = tofloat(children[i],a)
                    end
                    return `fn(nargs) 
                end
            else
                function gen(args) return e.op:generate(e,args) end
            end
            return IRNode:create { kind = "apply", op = e.op.name, generator = gen, children = children, type = e:type(), shape = e:shape() }
        elseif "Reduce" == e.kind then
            local vardecl = IRNode:create { kind = "vardecl", constant = 0, type = e:type(), shape = e:shape() }
            local arg = e.args[1]
            local red = IRNode:create { kind = "reduce", op = "sum", children  = terralib.newlist { vardecl, irmap(arg) }, type = vardecl.type, shape = arg:shape() }
            local children = terralib.newlist { vardecl, red }
            local varuse = IRNode:create { kind = "varuse", children = children, type = vardecl.type, shape = e:shape() }
            return varuse
        end
    end)
    
    local irroots = exps:map(irmap)
    
    local function  linearizedorder(irroots)
        local visited = {}
        local linearized = terralib.newlist()
        local function visit(ir)
            if visited[ir] then return end
            visited[ir] = true
            if ir.children then
                for i,c in ipairs(ir.children) do visit(c) end
            end
            if ir.condition then
                for i,c in ipairs(ir.condition.members) do visit(c) end
            end
            linearized:insert(ir)
        end
        for i,r in ipairs(irroots) do
            visit(r)
        end
        return linearized
    end
    
    -- tighten the conditions under which ir nodes execute
    local linearized = linearizedorder(irroots)
    for i = #linearized,1,-1 do
        local ir = linearized[i]
        if not ir.condition then
            ir.condition = Condition:create {}
        end
        local function applyconditiontolist(condition,lst)
            for i,c in ipairs(lst) do
                if not c.condition then
                    c.condition = condition
                elseif c.kind == "reduce" then -- single use is this node, so the condition is the already established condition plus any that the variable use imposes
                    c.condition = c.condition:Union(condition)
                else
                    c.condition = c.condition:Intersect(condition)
                end
            end
        end
        if ir.children then applyconditiontolist(ir.condition,ir.children) end
        if ir.kind == "reduce" then applyconditiontolist(Condition:create {}, ir.condition.members) end
    end
    
    local function calculateusesanddeps(roots)
        local uses,deps = {},{}
        local function visit(parent,ir)
            if not deps[ir] then assert(not uses[ir])
                uses[ir],deps[ir] = terralib.newlist(),terralib.newlist()
                local function visitlist(lst)
                    for i,c in ipairs(lst) do
                        deps[ir]:insert(c)
                        visit(ir,c)
                    end
                end
                if ir.children then visitlist(ir.children) end
                if ir.condition then visitlist(ir.condition.members) end
            end
            if parent then
                uses[ir]:insert(parent)
            end
        end
        for i, r in ipairs(roots) do
            visit(nil,r)
        end
        return uses,deps
    end
    
    local uses,deps = calculateusesanddeps(irroots)
     
    local function prefixsize(a,b)
        for i = 1,math.huge do
            if a[i] ~= b[i] or a[i] == nil then return i - 1 end
        end
    end
    local function conditiondiff(current,next)
        local i = prefixsize(current.members,next.members)
        local uplevels,downlevels = #current.members - i, #next.members - i
        return uplevels,downlevels
    end
    local function conditioncost(current,next)
        local uplevels,downlevels = conditiondiff(current,next)
        return uplevels*1000 + downlevels
    end
    local function shapecost(current,next)
        return current ~= next and 1 or 0
    end
        
    local function schedulebackwards(roots,uses)
        
        local state = nil -- ir -> "ready" or ir -> "scheduled"
        local readylists = terralib.newlist()
        local currentcondition,currentshape = Condition:create {}, ad.scalar
        local function enter()
            state = setmetatable({}, {__index = state})
            readylists:insert(terralib.newlist())
        end
        enter() --initial root level for non-speculative moves
        
        for i,r in ipairs(roots) do
            state[r] = "ready"
            readylists[#readylists]:insert(r)
        end
        
        local function leave()
            readylists:remove()
            state = assert(getmetatable(state).__index,"pop!")    
        end
        
        local function registersreleased(ir)
            if ir.kind == "const" then return 0
            elseif ir.kind == "vectorload" then return ir.count
            elseif ir.kind == "vectorextract" then return 0
            elseif ir.kind == "varuse" then return 0
            elseif ir.kind == "vardecl" then return 1
            elseif ir.kind == "reduce" then return 0 
            else return 1 end
        end
        local function registersliveonuse(ir)
            if ir.kind == "const" then return 0
            elseif ir.kind == "vectorload" then return 0 
            elseif ir.kind == "vectorextract" then return 1
            elseif ir.kind == "varuse" then return 1
            elseif ir.kind == "reduce" then return 0
            elseif ir.kind == "vardecl" then return 0
            else return 1 end
        end
        local function netregisterswhenscheduled(ir)
            local n = -registersreleased(ir)
            local newlive = {}
            for i,c in ipairs(deps[ir]) do
                newlive[c] = true
            end
            for k,_ in pairs(newlive) do
                if not state[k] then
                    n = n + registersliveonuse(k)
                end
            end
            return n
        end
        local function checkandmarkready(ir)
            if state[ir] ~= "ready" then
                for i,u in ipairs(uses[ir]) do
                    if state[u] ~= "scheduled" then return end -- not ready
                end            
                readylists[#readylists]:insert(ir)
                state[ir] = "ready"
            end
        end
        local function markscheduled(ir)
            state[ir] = "scheduled"
            for i,c in ipairs(deps[ir]) do 
                if not state[c] then
                    state[c] = "used"
                end
                checkandmarkready(c)
            end
        end
        
        local function vardeclcost(ir)
            return ir.kind == "vardecl" and 0 or 1
        end

        local function costspeculate(depth,ir)
            local c = netregisterswhenscheduled(ir)
            if depth > 0 then
                local minr = math.huge
                enter() -- start speculation level
                markscheduled(ir)
                
                for _,rl in ipairs(readylists) do
                    for _,candidate in ipairs(rl) do
                        if state[candidate] == "ready" then -- might not be ready because an overlay already scheduled it and we don't track the deletions
                            minr = math.min(minr,costspeculate(depth-1,candidate))
                        end
                    end
                end
                
                leave()
                if minr ~= math.huge then
                    c = c*10 + minr
                end
            end
            return c
        end

        local function cost(idx,ir)
            local c =  { shapecost(currentshape,ir.shape), conditioncost(currentcondition,ir.condition), vardeclcost(ir), costspeculate(1,ir) }
            --print("cost",idx,unpack(c))
            return c
        end
        
        local function costless(n,a,b)
            for i,ac in ipairs(a) do
                local bc = b[i]
                if ac ~= bc then return ac < bc end
            end
            return false
        end
        local ready = readylists[1] -- the true ready list is the first one, the rest are the speculative lists
        local function choose()
            --print("---------------------")
            local best = cost(1,assert(ready[1]))
            local bestidx = 1
            for i = 2,#ready do
                local ci = cost(i,ready[i])
                if costless(i,ci,best) then
                    bestidx = i
                    best = ci
                end
            end
            --print("choose",bestidx)
            return table.remove(ready,bestidx)
        end
        
        local instructions = terralib.newlist()
        local regcounts = terralib.newlist()
        local currentregcount = 1
        while #ready > 0 do
            local ir = choose()
            instructions:insert(1,ir)
            regcounts:insert(1,currentregcount)
            currentregcount = currentregcount + netregisterswhenscheduled(ir)
            markscheduled(ir)
            currentcondition,currentshape = ir.condition,ir.shape
        end
        return instructions,regcounts
    end
    
    local instructions,regcounts = schedulebackwards(irroots,uses)
    
    local function printschedule(instructions,regcounts)
        print("schedule for ",name,"-----------")
        local emittedpos = {}
        local function formatchildren(children)
            local cs = terralib.newlist()
            for i,c in ipairs(children) do
                cs:insert("r"..tostring(emittedpos[c]))
            end
            return cs:concat(",")
        end
    
        local function formatinst(inst)
            local fs = terralib.newlist()
            fs:insert(inst.kind.." ")
            for k,v in pairs(inst) do
                if k ~= "kind" and k ~= "children" and type(v) ~= "function" and k ~= "id" and k ~= "condition" and k ~= "type" then
                    fs:insert(tostring(v))
                    fs:insert(" ")
                end
            end
            if inst.children then
                fs:insert("{")
                fs:insert(formatchildren(inst.children))
                fs:insert("}")
            end
            return fs:concat()
        end
        local function formatcondition(c)
            local fs = terralib.newlist()
            fs:insert("[")
            fs:insert(formatchildren(c.members))
            fs:insert("]")
            local r = fs:concat()
            return r .. (" "):rep(4*(1+#c.members) - #r)
        end
        for i,ir in ipairs(instructions) do
            emittedpos[ir] = i
            print(("[%d]%sr%d : %s%s = %s"):format(regcounts[i],formatcondition(ir.condition),i,tostring(ir.type),tostring(ir.shape),formatinst(ir)))
            if instructions[i+1] and conditioncost(ir.condition,instructions[i+1].condition) ~= 0 then
                print("---------------------")
            end
        end
        print("----------------------")
    end
    
    if verboseAD then
        printschedule(instructions,regcounts)
    end
    
    local P = symbol(problemspec.P:ParameterType(),"P")
    local i,j,gi,gj = symbol(int32,"i"), symbol(int32,"j"),symbol(int32,"gi"), symbol(int32,"gj")
    local mi,mj = symbol(int32,"mi"), symbol(int32,"mj")
    local indexes = {[0] = i,j }
    
    local statementstack = terralib.newlist { terralib.newlist() } 
    local statements = statementstack[1]
    local extraimages = terralib.newlist()
    local emit
    
    local function emitconditionchange(current,next)
        local u,d = conditiondiff(current,next)
        for i = 0,u - 1 do
            local c = current.members[#current.members - i]
            local ce = emit(c)
            local stmts = statementstack:remove()
            statementstack[#statementstack]:insert quote
                if ce then
                    [stmts]
                end
            end
        end
        for i = 1,d do
            statementstack:insert(terralib.newlist())
        end
        statements = statementstack[#statementstack]
    end
    local currentidx
    local function conditioncoversload(condition,x,y)
        if x == 0 and y == 0 then return true end
        for i,ir in ipairs(condition.members) do
            assert(ir.type == bool)
            if ir.kind == "intrinsic" and ir.value.kind == "BoundsAccess" then
                if x == ir.value.x and y == ir.value.y then
                    return true
                end
            end
        end
        return false
    end
    
    local function createexp(ir)
        local function imageref(image)
            if image.idx >= 0 then
                return `P.[image.name]
            else
                if not extraimages[-image.idx] then
                    local imtype = problemspec.P:InternalImage(image.type,image.W,image.H)
                    extraimages[-image.idx] = symbol(imtype,image.name)
                end
                return extraimages[-image.idx]
            end
        end
        if "const" == ir.kind then
            return `float(ir.value)
        elseif "intrinsic" == ir.kind then
            local a = ir.value
            if "BoundsAccess" == a.kind then--bounds calculation
                return `opt.InBoundsCalc(gi+a.x,gj+a.y,W.size,H.size,a.sx,a.sy)
            elseif "IndexValue" == a.kind then
                return `[ assert(indexes[a.dim._index]) ] + a._shift 
            else assert("ParamValue" == a.kind)
                return `float(P.[a.name])
            end
        elseif "load" == ir.kind then
            local a = ir.value
            local im = imageref(a.image)
            assert(Offset:is(a.index),"NYI - graphs")
            if conditioncoversload(ir.condition,a.index.x,a.index.y) then
               return `im(mi+[a.index.x],mj+[a.index.y])
            else
               return `im:get(mi+[a.index.x],mj+[a.index.y],mi+[a.index.x],mj+[a.index.y])
            end
        elseif "vectorload" == ir.kind then
            local a = ir.value
            local im = imageref(a.image)
            assert(Offset:is(a.index),"NYI - graphs")
            local s = symbol(("%s_%s"):format(a.image.name,tostring(a.index)))
            
            if conditioncoversload(ir.condition,a.index.x,a.index.y) then
                statements:insert(quote
                    var [s] : a.image.type = im(mi+[a.index.x],mj+[a.index.y])
                end)
            else 
                statements:insert(quote
                    var [s] : a.image.type = 0.f
                    if opt.InBoundsCalc(mi+[a.index.x],mj+[a.index.y],[W.size],[H.size],0,0) then
                        [s] = im(mi+[a.index.x],mj+[a.index.y])
                    end
                end)
            end
            return s
        elseif "vectorextract" == ir.kind then
            local v = emit(ir.children[1])
            return `v(ir.channel)
        elseif "vectorconstruct" == ir.kind then
            local exps = ir.children:map(emit)
            return `[ad.TerraVector(float,#exps)]{ array(exps) }
        elseif "apply" == ir.kind then
            local exps = ir.children:map(emit)
            return ir.generator(exps)
        elseif "vardecl" == ir.kind then
            return `float(ir.constant)
        elseif "varuse" == ir.kind then
            local children = ir.children:map(emit)
            return children[1] -- return the variable declaration, which is the first child
        elseif "reduce" == ir.kind then
            local children = ir.children:map(emit)
            local vd, exp = children[1], tofloat(ir.children[2],children[2])
            local op
            if ir.op == "sum" then
                op = quote [vd] = [vd] + [exp] end
            else
                op = quote [vd] = [vd] * [exp] end
            end
            statements:insert(op)
            return children[1]
        end
    end
    
    local emitted,emitteduse = {},{}
    
    function emit(ir)
        assert(ir)
        return assert(emitted[ir],"use before def")
    end

    local basecondition = Condition:create {}
    local currentcondition = basecondition
    local currentshape = ad.scalar
    
    local function emitshapechange(current,next)
        if current == next then return end
        emitconditionchange(currentcondition,basecondition) -- exit all conditions
        currentcondition = basecondition
        while not current:isprefixof(next) do
            local stmts = statementstack:remove()
            local a = current.keys[#current.keys]
            statementstack[#statementstack]:insert quote
                for adj in P.[a.name]:neighbors(mi,mj) do
                    var [mi],[mj] = adj.x,adj.y
                    [stmts]
                end
            end
            current = current:fromreduction()
       end
       for i = 1,#next.keys - #current.keys do
            statementstack:insert(terralib.newlist())
       end
       statements = statementstack[#statementstack]
    end
    
    local declarations = terralib.newlist()
    for i,ir in ipairs(instructions) do
        currentidx = i
        emitshapechange(currentshape,ir.shape)
        currentshape = ir.shape
        
        emitconditionchange(currentcondition,ir.condition)
        currentcondition = ir.condition
        
        if false then -- dynamically check dependencies are initialized before use, very slow, only use for debugging
            local ruse = symbol(bool,"ruse"..tostring(i))
            declarations:insert quote var [ruse] = false end
            statements:insert quote [ruse] = true end
            emitteduse[ir] = ruse
            for _,u in ipairs(deps[ir]) do
                if ir.kind ~= "varuse" or ir.children[1] == u then
                    local ruse = assert(emitteduse[u])
                    local str = ("%s r%s used %s which is not initialized\n"):format(name,tostring(i),tostring(ruse))
                    statements:insert quote
                        if not ruse then
                            printf(str)
                        end
                    end
                end
            end
        end
        
        local r
        if ir.kind == "const" or ir.kind == "varuse" or ir.kind == "reduce" then 
            r = assert(createexp(ir),"nil exp") 
        else
            r = symbol(ir.type,"r"..tostring(i))
            declarations:insert quote var [r] end
            local exp = assert(createexp(ir),"nil exp")
            statements:insert(quote
                [r] = exp
            end)
        end
        emitted[ir] = r
    end
    --emitconditionchange(currentcondition,basecondition)
    emitshapechange(currentshape,ad.scalar) -- also blanks condition
    assert(#statementstack == 1)
    
    local results = irroots:map(emit)
    local terra generatedfn([i], [j], [gi], [gj], [P], [extraimages])
        var [mi],[mj] = [i],[j]
        [declarations]
        [statements]
        return [results]
    end
    generatedfn:setname(name)
    if verboseAD then
        generatedfn:printpretty(false, false)
    end
    return generatedfn
end

local function stencilforexpression(exp)
    local stencil = 0
    if ad.ExpVector:is(exp) then 
        for i,e in ipairs(exp:expressions()) do
            stencil = math.max(stencil,stencilforexpression(e))
        end
        return stencil
    end
    exp:visit(function(a)
        if "ImageAccess" == a.kind then
            assert(Offset:is(a.index), "stencils not defined for graph image access")
            stencil = math.max(stencil,math.max(math.abs(a.index.x),math.abs(a.index.y))) 
        elseif "BoundsAccess" == a.kind then--bounds calculation
            stencil = math.max(stencil,math.max(math.abs(a.x)+a.sx,math.abs(a.y)+a.sy))
        end
        return ad.v[a]
    end)
    return stencil
end
function ProblemSpecAD:createfunctionset(name,...)
    local exps = terralib.newlist {...}
    local ut = self.P:UnknownType()
    local W,H = ut.metamethods.W,ut.metamethods.H
    
    dprint("function for: ",name)
    local boundary = createfunction(self,name,exps,true,W,H)
    self.P:Function(name,boundary)
end

local function unknowns(exp)
    local seenunknown = {}
    local unknownvars = terralib.newlist()
    exp:visit(function(a)
        local v = ad.v[a]
        if ImageAccess:is(a) and a.image.name == "X" and not seenunknown[a] then -- assume image X is unknown
            unknownvars:insert(v)
            seenunknown[a] = true
        end
        return v
    end)
    return unknownvars
end

local function unknownaccesses(exp)
	return unknowns(exp):map("key")
end
--given that the residual at (0,0) uses the variables in 'unknownsupport',
--what is the set of residuals will use variable X(0,0).
--this amounts to taking each variable in unknown support and asking which residual is it
--that makes that variable X(0,0)
local function residualsincludingX00(unknownsupport,channel)
    assert(channel)
    local r = terralib.newlist()
    for i,u in ipairs(unknownsupport) do
        assert(Offset:is(u.index),"unexpected graph access")
        if u.channel == channel then
            r:insert(Offset:get(-u.index.x,-u.index.y))
        end
    end
    return r
end
local function unknownsforresidual(r,unknownsupport)
    return unknownsupport:map("shift",r.x,r.y)
end

local function conformtounknown(exps,unknown)
    if ad.isterravectortype(unknown.type) then return ad.Vector(unpack(exps))
    else return exps[1] end
end


local function createzerolist(N)
    local r = terralib.newlist()
    for i = 1,N do
        r[i] = ad.toexp(0)
    end
    return r
end
    
local function lprintf(ident,fmt,...)
    if true then return end 
    local str = fmt:format(...)
    ident = (" "):rep(ident*4)
    str = ident..str:gsub('\n', "\n"..ident)
    return print(str) 
end

local function createjtj(Fs,unknown,P)
    local P_hat = createzerolist(unknown.N)
	for rn,F in ipairs(Fs) do
        lprintf(0,"\n\n\n\n\n##################################################")
        lprintf(0,"r%d = %s",rn,F)
        local unknownsupport = unknownaccesses(F)
        for channel = 0, unknown.N-1 do
            local x = unknown(0,0,channel)
            local residuals = residualsincludingX00(unknownsupport,channel)
            local columns = {}
            local nonzerounknowns = terralib.newlist()
        
            for _,r in ipairs(residuals) do
                local rexp = shiftexp(F,r.x,r.y)
                local drdx00 = rexp:d(x)
                lprintf(1,"instance:\ndr%d_%d%d/dx00[%d] = %s",rn,r.x,r.y,channel,tostring(drdx00))
                local unknowns = unknownsforresidual(r,unknownsupport)
                for _,u in ipairs(unknowns) do
                    local drdx_u = rexp:d(unknown(u.index.x,u.index.y,u.channel))
                    local exp = drdx00*drdx_u
                    lprintf(2,"term:\ndr%d_%d%d/dx%d%d[%d] = %s",rn,r.x,r.y,u.index.x,u.index.y,u.channel,tostring(drdx_u))
                    if not columns[u] then
                        columns[u] = 0
                        nonzerounknowns:insert(u)
                    end
                    columns[u] = columns[u] + exp
                end
            end
            for _,u in ipairs(nonzerounknowns) do
                P_hat[channel+1] = P_hat[channel+1] + P(u.index.x,u.index.y,u.channel) * columns[u]
            end
        end
    end
    for i,p in ipairs(P_hat) do
        P_hat[i] = 2.0 * p
    end
    dprint("JTJ[nopoly] = ", ad.tostrings(P_hat))
    P_hat = ad.polysimplify(P_hat)
    dprint("JTJ[poly] = ", ad.tostrings(P_hat))
    return conformtounknown(P_hat,unknown)
end

local function createjtf(problemSpec,Fs,unknown,P)
   local F_hat = createzerolist(unknown.N) --gradient
   local P_hat = createzerolist(unknown.N) --preconditioner
	
	for ridx,F in ipairs(Fs) do
	    lprintf(0,"-------------")
	    lprintf(1,"R[%d] = %s",ridx,tostring(F))
        local unknownsupport = unknownaccesses(F)
        for channel = 0, unknown.N-1 do
            local x = unknown(0,0,channel)
            local residuals = residualsincludingX00(unknownsupport,channel)
            
            
            local sum = 0
            for _,f in ipairs(residuals) do
                local F_x = shiftexp(F,f.x,f.y)
                local dfdx00 = F_x:d(x)		-- entry of J^T
                local dfdx00F = dfdx00*F_x	-- entry of \gradF == J^TF
                F_hat[channel+1] = F_hat[channel+1] + dfdx00F			-- summing it up to get \gradF
    
                local dfdx00Sq = dfdx00*dfdx00	-- entry of Diag(J^TJ)
                P_hat[channel+1] = P_hat[channel+1] + dfdx00Sq			-- summing the pre-conditioner up
                --sum = sum + dfdx00F
                lprintf(2,"dR[%d]_%d_%d/dx[%d] = %s",ridx,f.x,f.y,channel,tostring(dfdx00F))
            end
            
            
        end
    end
	for i = 1,unknown.N do
	    if not problemSpec.P.usepreconditioner then
		    P_hat[i] = ad.toexp(1.0)
	    else
		    P_hat[i] = 2.0*P_hat[i]
		    P_hat[i] = ad.select(ad.greater(P_hat[i],.0001), 1.0/P_hat[i], 1.0)
	        P_hat[i] = ad.polysimplify(P_hat[i])
	    end
	    F_hat[i] = ad.polysimplify(2.0*F_hat[i])
	end
	dprint("JTF =", ad.tostrings({F_hat[1], F_hat[2], F_hat[3]}))
    return conformtounknown(F_hat,unknown), conformtounknown(P_hat,unknown)

end

local lastTime = nil
function timeSinceLast(name)
    local currentTime = terralib.currenttimeinseconds()
    if (lastTime) then
        local deltaTime = currentTime-lastTime
        --print(name,": ", deltaTime,"s")
    end
    lastTime = currentTime
end

local function creategradient(unknown,costexp)
    local unknownvars = unknowns(costexp)
    local gradient = costexp:gradient(unknownvars)

    dprint("grad expression")
    local names = table.concat(unknownvars:map(function(v) return tostring(v:key()) end),", ")
    dprint(names.." = "..ad.tostrings(gradient))
    
    local gradientsgathered = createzerolist(unknown.N)
    for i,u in ipairs(unknownvars) do
        local a = u:key()
        local shift = shiftexp(gradient[i],-a.index.x,-a.index.y)
        gradientsgathered[a.channel+1] = gradientsgathered[a.channel+1] + shift
    end
    dprint("grad gather")
    dprint(ad.tostrings(gradientsgathered))
    return conformtounknown(gradientsgathered,unknown)
end

function ProblemSpecAD:Cost(costexp_,jtjexp)
    local costexp = assert(ad.toexp(costexp_))
    local unknown = assert(self.nametoimage.X, "unknown image X is not defined")
    
    
    dprint("cost expression")
    dprint(ad.tostrings({assert(costexp)}))
    
    local gradient = creategradient(unknown,costexp)
    
    self.P:Stencil(stencilforexpression(costexp))
    self.P:Stencil(stencilforexpression(gradient))
    
    if SumOfSquares:is(costexp_) then
        local P = self:Image("P",unknown.type,unknown.W,unknown.H,-1)
        local jtjorig = createjtj(costexp_.terms,unknown,P)
        if not jtjexp then
            jtjexp = jtjorig	-- includes the 2.0
            dprint("jtjexp")
            dprint(jtjexp)
        end
        self.P:Stencil(stencilforexpression(jtjexp))
        self:createfunctionset("applyJTJ",jtjexp)
		--gradient with pre-conditioning
        local gradient,preconditioner = createjtf(self,costexp_.terms,unknown,P)	--includes the 2.0
		self:createfunctionset("evalJTF",gradient,preconditioner)
		
		--print("Gradient: ", removeboundaries(gradient))
		--print("Preconditioner: ", removeboundaries(preconditioner))
    end
    
    self:createfunctionset("cost",costexp)
    self:createfunctionset("gradient",gradient)
    if self.excludeexp then
        self:createfunctionset("exclude",self.excludeexp)
    end
    
    if verboseAD then
        self.excludeexp = nil
        terralib.tree.printraw(self)
    end
    return self.P
end
function ProblemSpecAD:Exclude(exp)
    self.excludeexp = assert(ad.toexp(exp), "expected a AD expression")
end

opt.Vector = ad.TerraVector
for i = 2,4 do
    opt["float"..tostring(i)] = ad.TerraVector(float,i)
end

util.Dot = macro(function(a,b) 
    local at,bt = a:gettype(),b:gettype()
    if ad.isterravectortype(at) then
        return `a:dot(b)
    else
        return `a*b
    end
end)
opt.Dot = util.Dot

opt.newImage = newImage

return opt
