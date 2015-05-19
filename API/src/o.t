
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

local vprintf = terralib.externfunction("vprintf", {&int8,&int8} -> int)

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
	local vars = {
		costFunctionType = problemSpec.functions.cost.boundary:gettype()
	}

	if kind == "gradientDescentCPU" then
        return solversCPU.gradientDescentCPU(problemSpec, vars)
	elseif kind == "gradientDescentGPU" then
		return solversGPU.gradientDescentGPU(problemSpec, vars)
	elseif kind == "conjugateGradientCPU" then
		return solversCPU.conjugateGradientCPU(problemSpec, vars)
	elseif kind == "linearizedConjugateGradientCPU" then
		return solversCPU.linearizedConjugateGradientCPU(problemSpec, vars)
	elseif kind == "linearizedConjugateGradientGPU" then
		return solversGPU.linearizedConjugateGradientGPU(problemSpec, vars)
	elseif kind == "lbfgsCPU" then
		return solversCPU.lbfgsCPU(problemSpec, vars)
	elseif kind == "vlbfgsCPU" then
		return solversCPU.vlbfgsCPU(problemSpec, vars)
	elseif kind == "vlbfgsGPU" then
		return solversGPU.vlbfgsGPU(problemSpec, vars)
	elseif kind == "bidirectionalVLBFGSCPU" then
		return solversCPU.bidirectionalVLBFGSCPU(problemSpec, vars)
	elseif kind == "adaDeltaGPU" then
		return solversGPU.adaDeltaGPU(problemSpec, vars)
	elseif kind == "conjugateGradientGPU" then
		return solversGPU.conjugateGradientGPU(problemSpec, vars)
	elseif kind == "gaussNewtonGPU" then
		return solversGPU.gaussNewtonGPU(problemSpec, vars)
	elseif kind == "gaussNewtonBlockGPU" then
		return solversGPU.gaussNewtonBlockGPU(problemSpec, vars)
	end
	
	error("unknown kind: "..kind)
    
end

struct opt.GradientDescentPlanParams {
    nIterations : uint64
}

struct opt.Plan(S.Object) {
    impl : {&opaque,&&opaque,&&opaque,&&opaque} -> {}
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
function opt.ProblemSpec()
    local BlockedProblemParameters = terralib.types.newstruct("BlockedProblemParameters")
	local problemSpec = ProblemSpec:new { 
	                         shouldblock = opt.problemkind:match("Block") or false,
                             parameters = terralib.newlist(),-- listing of each parameter, {name = <string>, kind = <image|adjacency|edgevalue>, idx = <number>, type = <thetypeusedtostoreit>, obj = <theobject for adj> }
                             names = {}, -- name -> index in parameters list
                             ProblemParameters = terralib.types.newstruct("ProblemParameters"),
                             BlockedProblemParameters = BlockedProblemParameters,
							 functions = {},
							 maxStencil = 0
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

function ProblemSpec:toname(name)
    name = assert(tostring(name))
    assert(not self.names[name],string.format("name %s already in use",name))
    self.names[name] = #self.parameters + 1
    return name
end

local newImage 

function ProblemSpec:MaxStencil()
	return self.maxStencil
end

function ProblemSpec:Stencil(stencil) 
	self.maxStencil = math.max(stencil, self.maxStencil)
end


function ProblemSpec:BlockSize()
	--TODO: compute based on problem
	--return opt.BLOCK_SIZE
	return 16
end

function ProblemSpec:BlockStride() return 2*self:MaxStencil() + self:BlockSize() end

function ProblemSpec:BlockedTypeForImage(W,H,typ)
	local elemsize = terralib.sizeof(assert(typ))
	return newImage(typ, W, H, elemsize, elemsize*self:BlockStride())
end
function ProblemSpec:BlockedTypeForImageEntry(p)
	local mm = p.type.metamethods
	return self:BlockedTypeForImage(mm.W,mm.H,mm.typ)
end

function ProblemSpec:newparameter(name,kind,idx,typ,obj)
    self.parameters:insert { name = self:toname(name), kind = kind, idx = idx, type = typ, obj = obj }
	self.ProblemParameters.entries:insert { name, typ }
end

function ProblemSpec:ParameterType(blocked) 
	if blocked == nil then
		blocked = self.shouldblock
	end
	return blocked and self.BlockedProblemParameters or  self.ProblemParameters
end
function ProblemSpec:UnknownType(blocked) 
	return self:TypeOf("X",blocked) 
end
function ProblemSpec:TypeOf(name,blocked)
	if blocked == nil then
		blocked = self.shouldblock
	end 
	local p = self.parameters[assert(self.names[name],"unknown name: " .. name)] 
	return blocked and self:BlockedTypeForImageEntry(p) or p.type
end

function ProblemSpec:Function(name,dimensions,boundary,interior)
    interior = interior or boundary
    interior:gettype() -- check this typechecks
    self.functions[name] = { name = name, dimensions = dimensions, boundary = boundary, interior = interior }
end
function ProblemSpec:Param(name,typ,idx)
    self:newparameter(name,"param",idx,typ)
end

function opt.Dim(name, idx)
    idx = assert(tonumber(idx), "expected an index for this dimension")
    local size = tonumber(opt.dimensions[idx])
    return Dim:new { name = name, size = size, _index = idx }
end

function opt.InternalDim(name, size)
	size = assert(tonumber(size), "expected a number for size")
    return Dim:new { name = name, size = size }
end

terra opt.InBoundsCalc(x : int64, y : int64, W : int64, H : int64, sx : int64, sy : int64) : int
    var minx,maxx,miny,maxy = x - sx,x + sx,y - sy,y + sy
    return int(minx >= 0) and int(maxx < W) and int(miny >= 0) and int(maxy < H)
end 

newImage = terralib.memoize(function(typ, W, H, elemsize, stride)
	local struct Image {
		data : &uint8
	}
	function Image.metamethods.__typename()
	  return string.format("Image(%s,%s,%s,%d,%d)",tostring(typ),W.name, H.name,elemsize,stride)
	end
	Image.metamethods.__apply = macro(function(self, x, y)
	 return `@[&typ](self.data + y*stride + x*elemsize)
	end)
	terra Image:inbounds(x : int64, y : int64)
	    return x >= 0 and y >= 0 and x < W.size and y < H.size
	end
	terra Image:get(x : int64, y : int64, gx : int64, gy : int64) : typ
	    var v : typ = 0.f --TODO:only works for single precision things
	    if opt.InBoundsCalc(gx,gy,W.size,H.size,0,0) ~= 0 then
	        v = self(x,y)
	    end
	    return v
	end
	terra Image:get(x : int64, y : int64) : typ return self:get(x,y,x,y) end
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

function ProblemSpec:Image(name,typ,W,H,idx)
    assert(terralib.types.istype(typ))
    local elemsize = assert(tonumber(opt.elemsizes[idx]))
    local stride = assert(tonumber(opt.strides[idx]))
    local r = newImage(typ, assert(todim(W)), assert(todim(H)), elemsize, stride)
    self:newparameter(name,"image",idx,r,nil)
end


function ProblemSpec:InternalImage(typ,W,H,blocked)
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


local newAdjacency = terralib.memoize(function(w0,h0,w1,h1)
    local struct Adj {
        rowpointer : &int64 --size: w0*h0+1
        x : &int64 --size is number of total edges in graph
        y : &int64
    }
    local struct AdjEntry {
        x : int64
        y : int64
    }
    function Adj.metamethods.__typename()
	  return string.format("Adj( {%s,%s}, {%s,%s} )",w0.name, h0.name,w1.name,h1.name)
	end
    local mm = Adj.metamethods
    terra Adj:count(i : int64, j : int64)
        var idx = j*w0.size + i
        return self.rowpointer[idx+1] - self.rowpointer[idx]
    end
    terra Adj:W0() return w0.size end
    terra Adj:H0() return h0.size end
    terra Adj:W1() return w1.size end
    terra Adj:H1() return h1.size end
    local struct AdjIter {
        adj : &Adj
        idx : int64
    }
    terra Adj:neighbors(i : int64, j : int64)
        return AdjIter { self, j*self:W0() + i }
    end
    AdjIter.metamethods.__for = function(syms,iter,body)
        return syms, quote
            var it = iter
            for i = it.adj.rowpointer[it.idx], it.adj.rowpointer[it.idx+1] do
                var [syms[1]] : AdjEntry = AdjEntry { it.adj.x[i], it.adj.y[i] }
                body
            end
        end
    end
    mm.fromDim = {w0,h0}
    mm.toDim = {w1,h1}
    mm.entry = AdjEntry
    return Adj
end)


function ProblemSpec:Adjacency(name,fromDim,toDim,idx)
    local w0,h0,w1,h1 = assert(todim(fromDim[1])),assert(todim(fromDim[2])),assert(todim(toDim[1])),assert(todim(toDim[2]))
    local Adj = newAdjacency(w0,h0,w1,h1)
    local obj = terralib.new(Adj,{assert(opt.rowindexes[idx]),assert(opt.xs[idx]),assert(opt.ys[idx])})
    self:newparameter(name,"adjacency",idx,Adj,obj)
end

local newEdgeValues = terralib.memoize(function(typ,adj)
     assert(terralib.types.istype(typ))
     local struct EdgeValues {
        data : &typ
     }
	 EdgeValues.metamethods.type = typ
     terra EdgeValues:get(a : adj.metamethods.entry) : &typ
        return self.data + a.y*[adj.metamethods.toDim[1].size] + a.x
     end
     EdgeValues.metamethods.__apply = macro(function(self, a)
	    return `@self:get(a)
	 end)
	 return EdgeValues
end)

function ProblemSpec:EdgeValues(name,typ,adjName, idx)
    local param = self.parameters[assert(self.names[adjName],"unknown adjacency")]
    assert(param.kind == "adjacency", "expected the name of an adjacency")
    local ev = newEdgeValues(typ, param.type)
    self:newparameter(name,"edgevalues",idx,ev,nil)
end

local allPlans = terralib.newlist()

errorPrint = rawget(_G,"errorPrint") or print

local function problemPlan(id, dimensions, elemsizes, strides, rowindexes, xs, ys, pplan)
    local success,p = xpcall(function() 
		local problemmetadata = assert(problems[id])
        opt.dimensions,opt.elemsizes,opt.strides = dimensions,elemsizes,strides
        opt.rowindexes,opt.xs,opt.ys = rowindexes,xs,ys
        opt.math = problemmetadata.kind:match("GPU") and util.gpuMath or util.cpuMath
		opt.problemkind = problemmetadata.kind
		
        local file, errorString = terralib.loadfile(problemmetadata.filename)
        if not file then
            error(errorString, 0)
        end
        local tbl = file()
        assert(ProblemSpec:is(tbl))
		local result = compilePlan(tbl,problemmetadata.kind,problemmetadata.params)
		allPlans:insert(result)
		pplan[0] = result()
    end,function(err) errorPrint(debug.traceback(err,2)) end)
end

terra opt.ProblemPlan(problem : &opt.Problem, dimensions : &uint64, elemsizes : &uint64, strides : &uint64, rowindexes : &&int64, xs : &&int64, ys : &&int64) : &opt.Plan
	var p : &opt.Plan = nil 
	problemPlan(int(int64(problem)),dimensions,elemsizes,strides,rowindexes,xs,ys,&p)
	return p
end 

terra opt.PlanFree(plan : &opt.Plan)
    -- TODO: plan should also have a free implementation
    plan:delete()
end

terra opt.ProblemSolve(plan : &opt.Plan, images : &&opaque, edgevalues : &&opaque, params : &&opaque)
	return plan.impl(plan.data, images, edgevalues, params)
end

ad = require("ad")

local VarDef = ad.newclass("VarDef") -- meta-data attached to each ad variable about what actual value it is
local ImageAccess = VarDef:Variant("ImageAccess") -- access into one particular image
local BoundsAccess = VarDef:Variant("BoundsAccess") -- query about the bounds of an image
local IndexValue = VarDef:Variant("IndexValue") -- query of the numeric index
local ParamValue = VarDef:Variant("ParamValue") -- get one of the global parameter values

function ImageAccess:__tostring()
    local xn,yn = tostring(self.x):gsub("-","m"),tostring(self.y):gsub("-","m")
    return ("%s_%s_%s_%s"):format(self.image.name,self.field,xn,yn)
end
function BoundsAccess:__tostring() return ("bounds_%d_%d_%d_%d"):format(self.x,self.y,self.sx,self.sy) end
function IndexValue:__tostring() return ({[0] = "i","j","k"})[self.dim._index] end
function ParamValue:__tostring() return "param_"..self.name end

ImageAccess.get = terralib.memoize(function(self,im,field,x,y)
    return ImageAccess:new { image = im, field = field, x = x, y = y}
end)

BoundsAccess.get = terralib.memoize(function(self,x,y,sx,sy)
    return BoundsAccess:new { x = x, y = y, sx = sx, sy = sy }
end)
IndexValue.get = terralib.memoize(function(self,dim,shift)
    return IndexValue:new { _shift = tonumber(shift) or 0, dim = assert(todim(dim),"expected a dimension object") } 
end)

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
    return ProblemSpecAD:new { P = opt.ProblemSpec() }
end


local Image = newclass("Image")
-- Z: this will eventually be opt.Image, but that is currently used by our direct methods
-- so this is going in the ad table for now
-- small workaround: idx > 0 means it is part of ProblemSpec struct
-- idx < 0 means that it is the -idx-th argument to the function begin generated after the ProblemSpec struct. (.e.g -1 is the first argument) 
function ProblemSpecAD:Image(name,W,H,idx)
    assert(W == 1 or Dim:is(W))
    assert(H == 1 or Dim:is(H))
    idx = assert(tonumber(idx))
    if idx >= 0 then
        self.P:Image(name,float,W,H,idx)
    end
    return Image:new { name = tostring(name), W = W, H = H, idx = idx }
end

function ProblemSpecAD:Param(name,typ,idx)
    self.P:Param(name,float,idx)
    return ad.v[ParamValue:new { name = name, type = typ }]
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
    local function nobounds(a)
        if BoundsAccess:is(a) then return ad.toexp(1)
        else return ad.v[a] end
    end
    return exp:rename(nobounds)
end
local function createfunction(problemspec,exp,usebounds,W,H)
    if not usebounds then
        exp = removeboundaries(exp)
    end
    
    local P = symbol(problemspec.P:ParameterType(),"P")
    local extraimages = terralib.newlist()
    local imagetosym = {} 
    local imageinits = terralib.newlist()
    local stmts = terralib.newlist()
    
    local stencil = {0,0}
    
    local i,j,gi,gj = symbol(int64,"i"), symbol(int64,"j"),symbol(int64,"gi"), symbol(int64,"gj")
    local indexes = {[0] = i,j }
    local accesssyms = {}
    local vartosym = {}
    local function emitvar(a)
        if not accesssyms[a] then
            local r 
            if "ImageAccess" == a.kind then
                local im = imagetosym[a.image] 
                if not im then
                    if a.image.idx >= 0 then
                        im = symbol(problemspec.P:TypeOf(a.image.name),a.image.name)
                        imageinits:insert(quote var [im] = P.[a.image.name] end)
                    else
                        local imtype = problemspec.P:InternalImage(float,a.image.W,a.image.H)
                        im = symbol(imtype,a.image.name)
                        extraimages[-a.image.idx] = im
                    end
                    imagetosym[a.image] = im
                end
                r = symbol(float,tostring(a))
                stmts:insert quote
                    var [r] = [ usebounds and (`im:get(i+[a.x],j+[a.y],gi+[a.x],gj+[a.y])) or (`im(i+[a.x],j+[a.y])) ]
                end
                stencil[1] = math.max(stencil[1],math.abs(a.x))
                stencil[2] = math.max(stencil[2],math.abs(a.y))
            elseif "BoundsAccess" == a.kind then--bounds calculation
                assert(usebounds) -- if we removed them, we shouldn't see any boundary accesses
                r = symbol(int,tostring(a))
                stmts:insert quote
                    var [r] = opt.InBoundsCalc(gi+a.x,gj+a.y,W.size,H.size,a.sx,a.sy)
                end
                stencil[1] = math.max(stencil[1],math.abs(a.x)+a.sx)
                stencil[2] = math.max(stencil[2],math.abs(a.y)+a.sy)
            elseif "IndexValue" == a.kind then
                r = `[ assert(indexes[a.dim._index]) ] + a._shift 
            else assert("ParamValue" == a.kind)
                r = `float(P.[a.name])
            end
            accesssyms[a] = r
        end
        return accesssyms[a]
    end
    -- NOTE: math is set globally for the particular plan being compiled to either util.gpuMath or util.cpuMath
    local function generatormap(op) 
        local fn = opt.math[op.name]
        if fn then
            return function(...) 
                local args = {...} 
                return `fn(args)
            end
        end 
    end
    local result = ad.toterra({exp},emitvar,generatormap)
    local terra generatedfn([i], [j], [gi], [gj], [P], [extraimages])
        [imageinits]
        [stmts]
        return result
    end
    if verboseAD then
        generatedfn:printpretty(false, false)
    end
    return generatedfn,stencil
end
local function createfunctionset(problemspec,name,exp)
    local ut = problemspec.P:UnknownType()
    local W,H = ut.metamethods.W,ut.metamethods.H
    
    dprint("function set for: ",name)
    dprint("bound")
    local boundary,stencil = createfunction(problemspec,exp,true,W,H)
    dprint("interior")
    local interior = createfunction(problemspec,exp,false,W,H)
    
    problemspec.P:Function(name,{W,H},boundary,interior)
end
local function unknowns(exp)
    local seenunknown = {}
    local unknownvars = terralib.newlist()
    exp:rename(function(a)
        local v = ad.v[a]
        if ImageAccess:is(a) and a.image.idx == 0 and a.field == "v" and not seenunknown[a] then -- assume image 0 is unknown
            unknownvars:insert(v)
            seenunknown[a] = true
        end
        return v
    end)
    return unknownvars
end
local function imagesusedinexpression(exp)
    local N = 0
    local idxtoimage = terralib.newlist{}
    exp:rename(function(a)
        if ImageAccess:is(a) then
            N = math.max(N,a.image.idx+1)
            assert(idxtoimage[a.image.idx+1] == nil or idxtoimage[a.image.idx+1] == a.image, "image for index " ..tostring(a.image.idx).. " defined twice?")
            idxtoimage[a.image.idx+1] = a.image
        end
        return ad.v[a]
    end)
    for i = 1,N do
        assert(idxtoimage[i],"undefined image at index "..tostring(i-1))
    end
    return idxtoimage
end

local getshift = terralib.memoize(function(x,y) return {x = x, y = y} end)

local function shiftswithoverlappingstencil(unknownvars)
    local shifttooverlap = {}
    local shifts = terralib.newlist()
    for i,a_ in ipairs(unknownvars) do
        local a = a_:key()
        for j,b_ in ipairs(unknownvars) do
            local b = b_:key()
            local s = getshift(a.x - b.x, a.y - b.y) -- at what shift from a to b does a's (a.x,a.y) overlap with b's (b.x,b.y)
            if not shifttooverlap[s] then
                shifttooverlap[s] = terralib.newlist()
                shifts:insert(s)
            end
            shifttooverlap[s]:insert({left = i, right = j})
        end
    end
    return shifts,shifttooverlap
end

local function createjtj(Fs,unknown,P)
    local P_hat = 0
	local x = unknown(0,0)
    for _,F in ipairs(Fs) do
        local P_F = 0
        local unknownvars = unknowns(F)
        local J = terralib.newlist()
		local dfdxshifts = F:gradient(unknownvars)
		local dfshiftsdx = terralib.newlist()
		for _,u in ipairs(unknownvars) do --Loop over the symmetries
			local a = u:key()
			local F_shift = shiftexp(F,a.x,a.y)
			dfshiftsdx:insert(F_shift:d(x))
		end
		local shifts,shifttooverlap = shiftswithoverlappingstencil(unknownvars)
        for _,shift in pairs(shifts) do
            local overlaps = shifttooverlap[shift]
            local sum = 0
            for i,o in ipairs(overlaps) do
                local alpha = dfdxshifts[o.right]
				local beta = dfshiftsdx[o.left]
                sum = sum + alpha*beta
            end
            P_F = P_F + P(shift.x,shift.y) * sum  
        end
        P_hat = P_hat + P_F
    end
    return P_hat
end

function ProblemSpecAD:Cost(costexp_)
    local costexp = assert(ad.toexp(costexp_))
    local images = imagesusedinexpression(costexp)
    local unknown = images[1]
    
    local unknownvars = unknowns(costexp)
    local gradient = costexp:gradient(unknownvars)
    
    dprint("cost expression")
    dprint(ad.tostrings({assert(costexp)}))
    dprint("grad expression")
    local names = table.concat(unknownvars:map(function(v) return tostring(v:key()) end),", ")
    dprint(names.." = "..ad.tostrings(gradient))
    
    local gradientgathered = 0
    for i,u in ipairs(unknownvars) do
        local a = u:key()
        gradientgathered = gradientgathered + shiftexp(gradient[i],-a.x,-a.y)
    end
    
    dprint("grad gather")
    dprint(ad.tostrings({gradientgathered}))
    
    createfunctionset(self,"cost",costexp)
    createfunctionset(self,"gradient",gradientgathered)
    
    if SumOfSquares:is(costexp_) then
        local P = self:Image("P",unknown.W,unknown.H,-1)
        local jtjexp = 2.0*createjtj(costexp_.terms,unknown,P)
        createfunctionset(self,"applyJTJ",jtjexp)
    end
    if verboseAD then
        terralib.tree.printraw(self)
    end
    return self.P
end

return opt