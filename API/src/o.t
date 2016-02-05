--terralib.settypeerrordebugcallback( function(fn) fn:printpretty() end )

opt = {} --anchor it in global namespace, otherwise it can be collected
local S = require("std")
local ffi = require("ffi")
local util = require("util")
local solversCPU = require("solversCPU")
local solversGPU = require("solversGPU")

local C = util.C

local use_bindless_texture = true
local use_pitched_memory = true
local use_split_sums = true
local use_condition_scheduling = true
local use_register_minimization = true
local use_conditionalization = true

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

local function compilePlan(problemSpec, kind)
    assert(kind == "gaussNewtonGPU","expected solver kind to be gaussNewtonGPU")
    return solversGPU.gaussNewtonGPU(problemSpec)
end

struct opt.Plan(S.Object) {
    init : {&opaque,&&opaque,&&opaque} -> {} -- plan.data,params,solverparams
    step : {&opaque,&&opaque,&&opaque} -> int
    data : &opaque
} 

struct opt.Problem {} -- just used as an opaque type, pointers are actually just the ID
local function problemDefine(filename, kind, pid)
    local problemmetadata = { filename = ffi.string(filename), kind = ffi.string(kind), id = #problems + 1 }
    problems[problemmetadata.id] = problemmetadata
    pid[0] = problemmetadata.id
end
-- define just stores meta-data right now. ProblemPlan does all compilation for now
terra opt.ProblemDefine(filename : rawstring, kind : rawstring)
    var id : int
    problemDefine(filename, kind, &id)
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
    local problemSpec = ProblemSpec:new { 
	                         parameters = terralib.newlist(),-- listing of each parameter, {name = <string>, kind = <image|graph>, idx = <number>, type = <thetypeusedtostoreit> }
                             names = {}, -- name -> index in parameters list
                             ProblemParameters = terralib.types.newstruct("ProblemParameters"),
                             functions = {},
							 maxStencil = 0,
							 stage = "inputs",
							 usepreconditioner = false,
                           }
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

function ProblemSpec:newparameter(name,kind,idx,typ)
    self.parameters:insert { name = self:toname(name), kind = kind, idx = idx, type = typ }
	self.ProblemParameters.entries:insert { name, typ }
end

function ProblemSpec:ParameterType()
    return self.ProblemParameters
end

function ProblemSpec:UnknownType()
    self:Stage "functions"
	return self:TypeOf("X") 
end

function ProblemSpec:TypeOf(name)
    self:Stage "functions"
	local p = self.parameters[assert(self.names[name],"unknown name: " .. name)] 
	return p.type
end

function ProblemSpec:Function(name,unknownfunction, ...)
    self:Stage "functions"
    unknownfunction:gettype() -- check this typechecks
    local graphfunctions = terralib.newlist()
    for i = 1,select("#",...),2 do
        local graphname, implementation =  select(i,...)
        implementation:gettype()
        graphfunctions:insert { graphname = graphname, implementation = implementation }
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
        return `bool(self.functions.exclude.unknownfunction(args))
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

opt.InBoundsCalc = macro(function(x,y,W,H,sx,sy)
    return `x >= sx and x < W - sx and y >= sy and y < H - sy
end)
	
newImage = terralib.memoize(function(typ, W, H)
    local elemsize = terralib.sizeof(assert(typ))
    local stride = elemsize*W.size
    -- NOTE: only use textures for float, vector(2, float), vector(4, float)
    local use_texture = use_bindless_texture
    if not (util.isvectortype(typ) and typ.metamethods.type == float and
                (typ.metamethods.N == 4 or typ.metamethods.N == 2)
           ) and typ ~= float
    then
        use_texture = false
    end

    local cd = macro(function(apicall) return quote
        var r = apicall
        if r ~= 0 then  
            C.printf("Cuda reported error %d: %s\n",r, C.cudaGetErrorString(r))
            return r
        end
    end end)

    local nvec = util.isvectortype(typ) and typ.metamethods.N or 1
    local wrapBindlessTexture, tex_read
    if use_texture then
        -- check alignment
        if stride % 128 ~= 0 then
            print(string.format("***** Bad Stride %d (elemsize %d) *****", stride, elemsize))
            use_pitched_memory = false
        end

        local name = "Image("..tostring(typ)..','..W.size..','..H.size..
                          ','..elemsize..','..stride..')'
        terra wrapBindlessTexture(data : &uint8) : C.cudaTextureObject_t

            -- Description of Texture Resource
            var res_desc : C.cudaResourceDesc
                C.memset(&res_desc, 0, sizeof(C.cudaResourceDesc))

            -- The following fields are in the same place for both
            -- the linear and pitch2D variant layouts
            res_desc.res.linear.devPtr      = data
            -- encode the fact we have vectors here...
            res_desc.res.linear.desc.f      = C.cudaChannelFormatKindFloat
            res_desc.res.linear.desc.x      = 32  -- bits per channel
            if nvec > 1 then
                res_desc.res.linear.desc.y  = 32
            end
            if nvec == 4 then
                res_desc.res.linear.desc.z  = 32
                res_desc.res.linear.desc.w  = 32
            end

            -- fill out differing data for linear vs pitch2D variants
            escape if use_pitched_memory then emit quote
                res_desc.resType                  = C.cudaResourceTypePitch2D
                res_desc.res.pitch2D.width        = W.size
                res_desc.res.pitch2D.height       = H.size
                res_desc.res.pitch2D.pitchInBytes = stride

                if [uint64](data) % 128 ~= 0 then
                    C.printf(["*****\n*****\n"..
                              " Bad Texture Start Alignment %lu"..
                              "\n*****\n*****\n"], [uint64](data))
                end
            end else emit quote
                res_desc.resType                = C.cudaResourceTypeLinear
                res_desc.res.linear.sizeInBytes = stride * H.size
            end end end

            var tex_desc : C.cudaTextureDesc
                C.memset(&tex_desc, 0, sizeof(C.cudaTextureDesc))
            -- out of bounds accesses are set to 0:
            --tex_desc.addressMode[0] = C.cudaAddressModeBorder
            --tex_desc.addressMode[1] = C.cudaAddressModeBorder
            ---- just read the entry:
            --tex_desc.filterMode     = C.cudaFilterModePoint
            tex_desc.readMode       = C.cudaReadModeElementType

            -- cudaTextureObject_t should be uint64
            var tex : C.cudaTextureObject_t = 0;
            cd(C.cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nil))

            return tex
        end

        -- texture base address is assumed to be aligned to a 16-byte boundary
        if use_pitched_memory then
            terra tex_read(tex : C.cudaTextureObject_t, x : int32, y : int32)
                var read = terralib.asm([tuple(float,float,float,float)],
                    "tex.2d.v4.f32.s32  {$0,$1,$2,$3}, [$4,{$5,$6}];",
                    "=f,=f,=f,=f,l,r,r",false, tex, x,y)
                return escape if nvec == 1 then
                    emit `read._0
                else
                    emit `@[&typ](&read)
                end end
            end
        else
            terra tex_read(tex : C.cudaTextureObject_t, x : int32, y : int32)
                var idx  = y * [stride/elemsize] + x
                var read = terralib.asm([tuple(float,float,float,float)],
                    "tex.1d.v4.f32.s32  {$0,$1,$2,$3}, [$4,{$5}];",
                    "=f,=f,=f,=f,l,r",false, tex, idx)
                return escape if nvec == 1 then
                    emit `read._0
                else
                    emit `@[&typ](&read)
                end end
            end
        end
    end

	local struct Image {
		data : &uint8;
        tex  : C.cudaTextureObject_t;
	}
	function Image.metamethods.__typename()
	  return string.format("Image(%s,%s,%s,%d,%d)",tostring(typ),W.name, H.name,elemsize,stride)
	end

    terra Image:debugprint()
        C.printf(['  %p %lu '..tostring(typ)..'\n'], self.data, self.tex)
    end

	if util.isvectortype(typ) and typ.metamethods.type == float and (typ.metamethods.N == 4 or typ.metamethods.N == 2) then
	    -- emit code that will produce special CUDA vector load instructions
	    local storetype = vector(float,typ.metamethods.N)
        if use_texture then
          terra Image.metamethods.__apply(self : &Image, x : int32, y : int32)
            return tex_read(self.tex, x, y)
          end
        else
          terra Image.metamethods.__apply(self : &Image, x : int32, y : int32)
            var a = @[&storetype](self.data + y*stride + x*elemsize)
            return @[&typ](&a)
          end
        end
        terra Image.metamethods.__update(self : &Image, x : int32, y : int32, v : typ)
            @[&storetype](self.data + y*stride + x*elemsize) = @[&storetype](&v)
        end
	else
        if use_texture then
          terra Image.metamethods.__apply(self : &Image, x : int32, y : int32)
            return tex_read(self.tex, x, y)
          end
        else
          terra Image.metamethods.__apply(self : &Image, x : int32, y : int32)
              return @[&typ](self.data + y*stride + x*elemsize)
          end
        end
        terra Image.metamethods.__update(self : &Image, x : int32, y : int32, v : typ)
            @[&typ](self.data + y*stride + x*elemsize) = v
        end
    end
	terra Image:inbounds(x : int32, y : int32)
	    return x >= 0 and y >= 0 and x < W.size and y < H.size
	end
	if util.isvectortype(typ) then
	    terra Image:atomicAdd(x : int32, y : int32, v : typ)
	        escape
	            for i = 0,typ.metamethods.N - 1 do
	                emit quote
	                    util.atomicAdd( &(@[&typ](self.data + y*stride + x*elemsize))(i) ,v(i))
	                end
	            end
	        end
	    end
	    terra Image:atomicAddChannel(x : int32, y : int32, c : int32, v : typ.metamethods.type)
	        var addr = &(@[&typ](self.data + y*stride + x*elemsize))(c)
	        util.atomicAdd(addr,v)
	    end
	else
	    terra Image:atomicAdd(x : int32, y : int32, v : typ)
	        util.atomicAdd([&typ](self.data + y*stride + x*elemsize),v)
	    end
	    terra Image:atomicAddChannel(x : int32, y : int32, c : int32, v : typ) 
	        var addr = [&typ](self.data + y*stride + x*elemsize)
	        util.atomicAdd(addr,v)
	    end
	end
    if use_texture and use_pitched_memory then
        Image.methods.get = macro(function(self,x,y,gx,gy)
            -- ignore gx, gy cause assuming no patch solver
            return `self(x,y)
        end)
    else
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
    end
    local terra lerp(v0 : typ, v1 : typ, t : float)
        return (1.f - t)*v0 + t*v1
    end
    terra Image:sample(x : float, y : float)
        var x0 : int, x1 : int = opt.math.floor(x),opt.math.ceil(x)
        var y0 : int, y1 : int = opt.math.floor(y),opt.math.ceil(y)
        var xn,yn = x - x0,y - y0
        var u = lerp(self:get(x0,y0),self:get(x1,y0),xn)
        var b = lerp(self:get(x0,y1),self:get(x1,y1),xn)
        return lerp(u,b,yn)
    end
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
        var data : &uint8
        C.cudaMalloc([&&opaque](&data), stride*H.size)
        C.cudaMemset([&opaque](data), 0, stride*H.size)
        self:initFromGPUptr(data)
    end
    terra Image:initFromGPUptr( ptr : &uint8 )
        self.data = nil
        self:setGPUptr(ptr)
    end
    if use_texture then
        terra Image:setGPUptr(ptr : &uint8)
            if self.data ~= ptr then
                if self.data ~= nil then
                    cd(C.cudaDestroyTextureObject(self.tex))
                end
                self.tex = wrapBindlessTexture(ptr)
            end
            self.data = ptr
        end
    else
        terra Image:setGPUptr(ptr : &uint8) self.data = ptr end
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
    if util.isvectortype(typ) then
        return typ, typ.metamethods.N
    elseif typ:isarithmetic() then
        return typ, 1
    end
end

function ProblemSpec:Image(name,typ,W,H,idx)
    self:Stage "inputs"
    typ = assert(tovalidimagetype(typ,"expected a number or an array of numbers"))
    local r = self:ImageType(typ,W,H) 
    self:newparameter(name,"image",idx,r)
end

function ProblemSpec:ImageType(typ,W,H)
    W,H = assert(todim(W)),assert(todim(H))
    assert(terralib.types.istype(typ))
    return newImage(typ,W,H)
end

function ProblemSpec:Graph(name, idx, ...)
    self:Stage "inputs"
    local GraphType = terralib.types.newstruct(name)
    GraphType.entries:insert ( {"N",int32} )
    
    local mm = GraphType.metamethods
    mm.idx = idx -- the index into the graph size table
    mm.elements = terralib.newlist()
    for i = 1, select("#",...),5 do
        local name,W,H,xidx,yidx = select(i,...) --TODO: we don't bother to track the dimensions of these things now
        assert(todim(W) and todim(H),"Expected dimensions")
        GraphType.entries:insert ( {name .. "_x", &int32} )
        GraphType.entries:insert ( {name .. "_y", &int32} )
        mm.elements:insert( { name = name, xidx = assert(tonumber(xidx)), yidx = assert(tonumber(yidx)) } )
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

local function problemPlan(id, dimensions, pplan)
    local success,p = xpcall(function() 
		local problemmetadata = assert(problems[id])
        opt.dimensions = dimensions
        opt.math = problemmetadata.kind:match("GPU") and util.gpuMath or util.cpuMath
        opt.problemkind = problemmetadata.kind
		
        local tbl = opt.problemSpecFromFile(problemmetadata.filename)
        assert(ProblemSpec:is(tbl))
		local result = compilePlan(tbl,problemmetadata.kind)
		allPlans:insert(result)
		pplan[0] = result()
    end,function(err) errorPrint(debug.traceback(err,2)) end)
end

terra opt.ProblemPlan(problem : &opt.Problem, dimensions : &uint32) : &opt.Plan
	var p : &opt.Plan = nil 
	problemPlan(int(int64(problem)),dimensions,&p)
	return p
end 

terra opt.PlanFree(plan : &opt.Plan)
    -- TODO: plan should also have a free implementation
    plan:delete()
end

terra opt.ProblemInit(plan : &opt.Plan, params : &&opaque, solverparams : &&opaque) 
    return plan.init(plan.data, params, solverparams)
end
terra opt.ProblemStep(plan : &opt.Plan, params : &&opaque, solverparams : &&opaque) : int
    return plan.step(plan.data, params, solverparams)
end
terra opt.ProblemSolve(plan : &opt.Plan, params : &&opaque, solverparams : &&opaque)
   opt.ProblemInit(plan, params, solverparams)
   while opt.ProblemStep(plan, params, solverparams) ~= 0 do end
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
local emptygradient = {}
function ImageAccess:gradient()
    if self.image.gradientimages then
        assert(Offset:is(self.index),"NYI - support for graphs")
        local gt = {}
        for u,im in pairs(self.image.gradientimages) do
            local exp = self.image.gradientexpressions[u]
            local k = u:shift(self.index.x,self.index.y)
            local v = ad.Const:is(exp) and exp or im(self.index.x,self.index.y)
            gt[k] = v
        end
        return gt
    end
    return emptygradient
 end

function Dim:index() return ad.v[IndexValue:get(self)] end

local SumOfSquares = newclass("SumOfSquares")
function ad.sumsquared(...)
    local exp = terralib.newlist {}
    for i = 1, select("#",...) do
        local e = select(i,...)
        if ad.ExpVector:is(e) then
            for i,t in ipairs(e:expressions()) do
                t = assert(ad.toexp(t), "expected an ad expression")
                exp:insert(t)
            end
        else
            exp:insert((assert(ad.toexp(e), "expected an ad expression")))
        end
    end
    return SumOfSquares:new { terms = exp }
end
local ProblemSpecAD = newclass("ProblemSpecAD")

function ad.ProblemSpec()
    return ProblemSpecAD:new { P = opt.ProblemSpec(), nametoimage = {}, precomputed = terralib.newlist() }
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
    assert(type(idx) == "number" or idx == "alloc", "expected an index number") -- alloc indicates that the solver should allocate the image as an intermediate
    if idx == "alloc" or idx >= 0 then
        self.P:Image(name,typ,W,H,idx)
    end
    local typ,N = tovalidimagetype(typ)
    local r = Image:new { name = tostring(name), W = W, H = H, idx = idx, N = N, type = typ }
    self.nametoimage[name] = r
    return r
end

function Image:__tostring() return self.name end

local ImageVector = newclass("ImageVector") -- wrapper for many images in a vector, just implements the __call methodf for Images Image:

function ProblemSpecAD:ComputedImage(name,W,H,exp)
    if ad.ExpVector:is(exp) then
        local imgs = terralib.newlist()
        for i,e in ipairs(exp:expressions()) do
            imgs:insert(self:ComputedImage(name.."_"..tostring(i-1),W,H,e))
        end
        return ImageVector:new { images = imgs }
    end
    exp = assert(ad.toexp(exp),"expected a math expression")
    local unknowns = terralib.newlist()
    local seen = {}
    exp:visit(function(a)
        if ImageAccess:is(a) and a.image.name == "X" then
            assert(Offset:is(a.index),"NYI - support for graphs")
            if not seen[a] then
                seen[a] = true
                unknowns:insert(a)
            end
        end
    end)
    local im = self:Image(name,float,W,H,"alloc")
    local gradients = exp:gradient(unknowns:map(function(x) return ad.v[x] end))
    local gradientexpressions = {}
    local gradientimages = {}
    for i,u in ipairs(unknowns) do
        gradientexpressions[u] = gradients[i]
        gradientimages[u] = self:Image(name.."_d_"..tostring(u),float,W,H,"alloc")
    end
    
    im.expression = exp
    im.gradientexpressions = gradientexpressions
    im.gradientimages = gradientimages
    self.precomputed:insert(im)
    return im
end

local Graph = newclass("Graph")
function Graph:__tostring() return self.name end

function ProblemSpecAD:Graph(name,idx,...)
    self.P:Graph(name,idx,...)
    local g = Graph:new { name = tostring(name) }
    for i = 1, select("#",...),5 do
        local name,W,H,xidx,yidx = select(i,...)
        g[name] = GraphElement:get(g,name)
    end
    return g
end

function ProblemSpecAD:Param(name,typ,idx)
    self.P:Param(name,float,idx)
    return ad.v[ParamValue:new { name = name, type = typ }]
end

function Image:__call(x,y,c)
    local index
    if GraphElement:is(x) then
        index = x
        assert(not c,"extra argument?")
        c = y
    else
        index = Offset:get(assert(tonumber(x)),assert(tonumber(y)))
    end
    c = tonumber(c)
    assert(not c or c < self.N, "channel outside of range")
    if self.N == 1 or c then
        return ad.v[ImageAccess:get(self,ad.scalar,index,c or 0)]
    else
        local r = {}
        for i = 1,self.N do
            r[i] = ad.v[ImageAccess:get(self,ad.scalar,index,i-1)]
        end
        return ad.Vector(unpack(r))
    end
end
function ImageVector:__call(x,y,c)
    if c then
        assert(c < #self.images, "channel outside of range")
        return self.images[c+1](x,y)
    end
    local result = self.images:map(function(im) return im(x,y) end)
    return ad.Vector(unpack(result))
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

local function createfunction(problemspec,name,usebounds,W,H,ndims,results,scatters)
    results = removeboundaries(results)
    
    local imageload = terralib.memoize(function(image)
        return IRNode:create { kind = "vectorload", value = image, type = image.image.type, shape = image:shape(), count = 0 }
    end)
    local imagesample = terralib.memoize(function(image, shape, x, y)
        return IRNode:create { kind = "sampleimage", image = image, type = image.type, shape = shape, count = 0, children = terralib.newlist {x,y} }
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
            return IRNode:create { kind = "vectorconstruct", children = e.data:map(irmap), type = util.Vector(float,#e.data) }
        elseif "Var" == e.kind then
            local a = e:key()
            if "ImageAccess" == a.kind then
                if not a.image.type:isarithmetic() then
                    local loadvec = imageload(ImageAccess:get(a.image,a:shape(),a.index,0))
                    loadvec.count = loadvec.count + 1
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
            if use_split_sums and (e.op.name == "sum") and #e:children() > 2 then
                local vardecl = IRNode:create { kind = "vardecl", constant = e.config.c, type = float, shape = e:shape() }
                local children = terralib.newlist { vardecl }
                local varuse = IRNode:create { kind = "varuse", children = children, type = float, shape = e:shape() }
                for i,c in ipairs(e:children()) do
                    children:insert(createreduce(e.op.name,vardecl,c))
                end
                return varuse
            end
            local children = e:children():map(irmap)
            if e.op.name:match("^sampleimage") then
                local sm = imagesample(e.op.imagebeingsampled,e:shape(),children[1],children[2])
                sm.count = sm.count + 1
                if not util.isvectortype(sm.image.type) then
                    return sm
                end
                return IRNode:create { kind = "vectorextract", children = terralib.newlist { sm }, channel = e.config.c, type = e:type(), shape = e:shape() }
            end
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
    
    local irroots = results:map(irmap)
    for i,s in ipairs(scatters) do
        irroots:insert(irmap(s.expression))
    end
    
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
        if use_conditionalization then
            if ir.children then applyconditiontolist(ir.condition,ir.children) end
        end
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
            elseif ir.kind == "vectorload" or ir.kind == "sampleimage" then return ir.count
            elseif ir.kind == "vectorextract" then return 0
            elseif ir.kind == "varuse" then return 0
            elseif ir.kind == "vardecl" then return 1
            elseif ir.kind == "reduce" then return 0 
            else return 1 end
        end
        local function registersliveonuse(ir)
            if ir.kind == "const" then return 0
            elseif ir.kind == "vectorload" then return 0
            elseif ir.kind == "sampleimage" then return util.isvectortype(ir.type) and 0 or 1
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
            local c =  { shapecost(currentshape,ir.shape) }
            if use_condition_scheduling then
                table.insert(c, conditioncost(currentcondition,ir.condition))
            end
            if use_register_minimization then
                table.insert(c, vardeclcost(ir))
                table.insert(c, costspeculate(1,ir))
            end
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
                local bx,by,sx,sy = ir.value.x,ir.value.y,ir.value.sx,ir.value.sy
                local minx,maxx,miny,maxy = bx - sx, bx + sx,by - sy, by + sy
                if minx <= x and x <= maxx and miny <= y and y <= maxy then
                    return true
                end
            end
        end
        return false
    end
    
    local function imageref(image)
        if image.idx == "alloc" or image.idx >= 0 then
            return `P.[image.name]
        else
            if not extraimages[-image.idx] then
                local imtype = problemspec.P:ImageType(image.type,image.W,image.H)
                extraimages[-image.idx] = symbol(imtype,image.name)
            end
            return extraimages[-image.idx]
        end
    end
    local function graphref(ge)
        return {`P.[ge.graph.name].[ge.element.."_x"][i],`P.[ge.graph.name].[ge.element.."_y"][i]}
    end
    local function createexp(ir)        
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
            if Offset:is(a.index) then
                if conditioncoversload(ir.condition,a.index.x,a.index.y) then
                   return `im(mi+[a.index.x],mj+[a.index.y])
                else
                   return `im:get(mi+[a.index.x],mj+[a.index.y],mi+[a.index.x],mj+[a.index.y])
                end
            else
                local gr = graphref(a.index)
                return `im(gr)
            end
        elseif "vectorload" == ir.kind then
            local a = ir.value
            local im = imageref(a.image)
            local s = symbol(("%s_%s"):format(a.image.name,tostring(a.index)))
            if Offset:is(a.index) then
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
            else
                local gr = graphref(a.index)
                statements:insert(quote
                    var [s] : a.image.type = im(gr)
                end)
            end
            return s
        elseif "vectorextract" == ir.kind then
            local v = emit(ir.children[1])
            return `v(ir.channel)
        elseif "vectorconstruct" == ir.kind then
            local exps = ir.children:map(emit)
            return `[util.Vector(float,#exps)]{ array(exps) }
        elseif "sampleimage" == ir.kind then
            local im = imageref(ir.image)
            local exps = ir.children:map(emit)
            return `im:sample(exps)
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
                --[[if [name == "precompute"] and mi == 10 and mj == 40 then
                    printf("%d,%d: k_%s: %s = %f\n",mi,mj,name,[tostring(r)],float(r))
                end]]
            end)
        end
        emitted[ir] = r
    end
    --emitconditionchange(currentcondition,basecondition)
    emitshapechange(currentshape,ad.scalar) -- also blanks condition
    assert(#statementstack == 1)
    
    local expressions = irroots:map(emit)
    local resultexpressions,scatterexpressions = {unpack(expressions,1,#results)},{unpack(expressions,#results+1)}
    
    local dimarguments
    if ndims == 2 then
        dimarguments = {i,j,gi,gj}
    else
        dimarguments = {i}
    end
        
    local scatterstatements = terralib.newlist()
    local function xypair(index)
        if Offset:is(index) then return {`mi + index.x, `mj + index.y }
        else return graphref(index) end
    end 
    for i,s in ipairs(scatters) do
        local image,exp = imageref(s.image),scatterexpressions[i]
        local xy = xypair(s.index)
        local stmt
        if s.kind == "add" then
            stmt = s.channel and (`image:atomicAddChannel(xy, s.channel, exp)) or (`image:atomicAdd(xy, exp))
        else
            assert(s.kind == "set" and s.channel == 0)
            stmt = quote 
                image(xy) = exp
            end -- NYI - multi-channel images
        end
        scatterstatements:insert(stmt)
    end

    local terra generatedfn([dimarguments], [P], [extraimages])
        escape
            if ndims == 1 then
                emit quote
                  var[j],[gi],[gj] = 0,0,0
                end
            end
        end
        var [mi],[mj] = [i],[j]
        [declarations]
        [statements]
        [scatterstatements]
        return [resultexpressions]
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
    end)
    return stencil
end
local noscatters = terralib.newlist()
function ProblemSpecAD:createfunctionset(name,results,graphfunctions)
    results,graphfunctions = terralib.newlist(results), terralib.newlist(graphfunctions)
    local ut = self.P:UnknownType()
    local W,H = ut.metamethods.W,ut.metamethods.H
    dprint("function for: ",name)
    local centered = createfunction(self,name,true,W,H,2,results,noscatters)
    local args = terralib.newlist()
    for i,g in ipairs(graphfunctions) do
        local gf = createfunction(self,("%s_graph%d"):format(name,i),true,W,H,1,g.results,g.scatters)
        args:insert(g.graph.name)
        args:insert(gf)
    end
    self.P:Function(name,centered,unpack(args))
end

-- the result is a data structure
-- { unknown = {<entry>}, graphs = { <graphobj> -> <entry> } }, where each entry is:
-- { expression = <ad-exp>, unknownaccess = [<ImagAccess>] }

local function classifyresiduals(Rs)
    local result = { unknown = terralib.newlist {}, graphs = {} }
    local function unknownaccesses(exp)
        local classification
        local seenunknown = {}
        local unknownaccesses = terralib.newlist()
        local function addunknown(u)
            if not seenunknown[u] then
                unknownaccesses:insert(u)
                seenunknown[u] = true
            end
        end
        exp:visit(function(a)
            if ImageAccess:is(a) then -- assume image X is unknown
                if a.image.name == "X"then
                    addunknown(a)
                elseif a.image.gradientimages then
                    for u,_ in pairs(a.image.gradientimages) do
                        assert(Offset:is(a.index),"NYI - precomputed with graphs")
                        addunknown(u:shift(a.index.x,a.index.y))
                    end
                end
                local aclass = Offset:is(a.index) and "centered" or a.index.graph
                assert(nil == classification or aclass == classification, "residual contains image reads from multiple domains")
                classification = aclass
            end
        end)
        local entry = { expression = exp, unknownaccesses = unknownaccesses }
        if not classification then
            classification = "centered"
        end
        --assert(classification, "residual does not contain an unknown term?")
        if classification == "centered" then
            result.unknown:insert(entry)
        else
            local list = result.graphs[classification] or terralib.newlist()
            list:insert(entry)
            result.graphs[classification] = list
        end
    end
    for i,r in ipairs(Rs) do
        unknownaccesses(r)
    end
    return result
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
    if util.isvectortype(unknown.type) then return ad.Vector(unpack(exps))
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

local function createjtjcentered(residuals,unknown,P)
    local P_hat_c = {}
    local conditions = terralib.newlist()
    for rn,residual in ipairs(residuals.unknown) do
        local F,unknownsupport = residual.expression,residual.unknownaccesses
        lprintf(0,"\n\n\n\n\n##################################################")
        lprintf(0,"r%d = %s",rn,F)
        for channel = 0, unknown.N-1 do
            local x = unknown(0,0,channel)
            local residuals = residualsincludingX00(unknownsupport,channel)
            local columns = {}
            local nonzerounknowns = terralib.newlist()
        
            for _,r in ipairs(residuals) do
                local rexp = shiftexp(F,r.x,r.y)
                local condition,drdx00 = ad.splitcondition(rexp:d(x))
                lprintf(1,"instance:\ndr%d_%d%d/dx00[%d] = %s",rn,r.x,r.y,channel,tostring(drdx00))
                local unknowns = unknownsforresidual(r,unknownsupport)
                for _,u in ipairs(unknowns) do
                    local condition2, drdx_u = ad.splitcondition(rexp:d(unknown(u.index.x,u.index.y,u.channel)))
                    local exp = drdx00*drdx_u
                    lprintf(2,"term:\ndr%d_%d%d/dx%d%d[%d] = %s",rn,r.x,r.y,u.index.x,u.index.y,u.channel,tostring(drdx_u))
                    if not columns[u] then
                        columns[u] = 0
                        nonzerounknowns:insert(u)
                    end
                    local conditionmerged = condition*condition2
                    if not P_hat_c[conditionmerged] then
                        conditions:insert(conditionmerged)
                        P_hat_c[conditionmerged] = createzerolist(unknown.N)
                    end
                    P_hat_c[conditionmerged][channel+1] = P_hat_c[conditionmerged][channel+1] + P(u.index.x,u.index.y,u.channel)*exp
                end
            end
        end
    end
    local P_hat = createzerolist(unknown.N)
    for _,c in ipairs(conditions) do
        for i = 1,unknown.N do
            P_hat[i] = P_hat[i] + c*P_hat_c[c][i]
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

local function NewGraphFunctionSpec(graph, results, scatters)
    return { graph = graph, results = results, scatters = scatters }
end
local function NewScatter(im,idx,channel,exp, kind) 
    return { image = im, index = idx, channel = channel, expression = exp, kind = kind or "add" }
end

local function createjtjgraph(residuals,P,Ap_X)
    local jtjgraph = terralib.newlist()
    for graph,terms in pairs(residuals.graphs) do
        local result = ad.toexp(0)
        local scatters = terralib.newlist() 
        local scattermap = {}
        local function addscatter(u,exp)
            local s = scattermap[u]
            if not s then
                s =  NewScatter(Ap_X,u.index,u.channel,ad.toexp(0))
                scattermap[u] = s
                scatters:insert(s)
            end
            s.expression = s.expression + exp
        end
        for i,term in ipairs(terms) do
            local F,unknownsupport = term.expression,term.unknownaccesses
            local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
            local partials = F:gradient(unknownvars)
            local Jp = ad.toexp(0)
            for i,partial in ipairs(partials) do
                local u = unknownsupport[i]
                assert(GraphElement:is(u.index))
                Jp = Jp + partial*P(u.index,u.channel)
            end
            for i,partial in ipairs(partials) do
                local u = unknownsupport[i]
                local jtjp = 2*Jp*partial
                result = result + P(u.index,u.channel)*jtjp
                addscatter(u,jtjp)
            end
        end
        jtjgraph:insert(NewGraphFunctionSpec(graph,terralib.newlist { result },scatters)) 
    end
    return jtjgraph
end
local function createjtj(residuals,unknown,P,Ap_X)
    return createjtjcentered(residuals,unknown,P),createjtjgraph(residuals,P,Ap_X)
end

local function createjtfcentered(problemSpec,residuals,unknown)
   local F_hat = createzerolist(unknown.N) --gradient
   local P_hat = createzerolist(unknown.N) --preconditioner
    
    for ridx,residual in ipairs(residuals.unknown) do
        local F, unknownsupport = residual.expression,residual.unknownaccesses
        lprintf(0,"-------------")
        lprintf(1,"R[%d] = %s",ridx,tostring(F))
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
		    P_hat[i] = ad.polysimplify(P_hat[i])
	    end
	    F_hat[i] = ad.polysimplify(2.0*F_hat[i])
	end
	dprint("JTF =", ad.tostrings({F_hat[1], F_hat[2], F_hat[3]}))
    return terralib.newlist{conformtounknown(F_hat,unknown), conformtounknown(P_hat,unknown) }
end

local function createjtfgraph(residuals,R,Pre)
    local jtjgraph = terralib.newlist()
    for graph,terms in pairs(residuals.graphs) do
        local scatters = terralib.newlist() 
        local scattermap = { [R] = {}, [Pre] = {}}
        local function addscatter(im,u,exp)
            local s = scattermap[im][u]
            if not s then
                s =  NewScatter(im,u.index,u.channel,ad.toexp(0))
                scattermap[im][u] = s
                scatters:insert(s)
            end
            s.expression = s.expression + exp
        end
        for i,term in ipairs(terms) do
            local F,unknownsupport = term.expression,term.unknownaccesses
            local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
            local partials = F:gradient(unknownvars)
            local Jp = ad.toexp(0)
            for i,partial in ipairs(partials) do
                local u = unknownsupport[i]
                assert(GraphElement:is(u.index))
                addscatter(R,u,-2.0*partial*F)
                addscatter(Pre,u,2.0*partial*partial)
            end
        end
        jtjgraph:insert(NewGraphFunctionSpec(graph,terralib.newlist {},scatters))
    end
    return jtjgraph
end
local function createjtf(problemSpec,residuals,unknown,R,Pre)
    return createjtfcentered(problemSpec,residuals,unknown), createjtfgraph(residuals,R,Pre)
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

local function createcost(residuals)
    local function sumsquared(terms)
        terms = terms:map("expression")
        local sum = ad.toexp(0)
        for i,t in ipairs(terms) do
            sum = sum + t*t
        end
        return sum
    end
    local graphwork = terralib.newlist()
    for graph,terms in pairs(residuals.graphs) do
        graphwork:insert( NewGraphFunctionSpec(graph,terralib.newlist { sumsquared(terms) }, terralib.newlist()) )
    end
    return sumsquared(residuals.unknown), graphwork
end

function createprecomputed(self,name,precomputedimages)
    local scatters = terralib.newlist()
    for i,im in ipairs(precomputedimages) do
        local expression = ad.polysimplify(im.expression)
        scatters:insert(NewScatter(im, Offset:get(0,0), 0, im.expression, "set"))
        for u,gim in pairs(im.gradientimages) do
            local gradientexpression = im.gradientexpressions[u]
            gradientexpression = ad.polysimplify(gradientexpression)
            if not ad.Const:is(gradientexpression) then
                scatters:insert(NewScatter(gim, Offset:get(0,0), 0, gradientexpression, "set"))
            end
        end
    end
    
    local ut = self.P:UnknownType()
    local W,H = ut.metamethods.W,ut.metamethods.H    
    return createfunction(self,name,true,W,H,2,terralib.newlist(),scatters)
end

function ProblemSpecAD:Cost(costexp)
    local unknown = assert(self.nametoimage.X, "unknown image X is not defined")
    
    assert(SumOfSquares:is(costexp),"expected a sum of squares object")
    
    local residuals = classifyresiduals(costexp.terms)
    
    local centeredcost,graphcost = createcost(residuals)
    
    self.P:Stencil(stencilforexpression(centeredcost))
    
    -- Not updated for graphs yet:    
    local P = self:Image("P",unknown.type,unknown.W,unknown.H,-1)
    local Ap_X = self:Image("Ap_X",unknown.type,unknown.W,unknown.H,-2)
    local jtjexp,jtjgraph = createjtj(residuals,unknown,P,Ap_X)
    self.P:Stencil(stencilforexpression(jtjexp))
    
    local R = self:Image("R",unknown.type,unknown.W,unknown.H,-1)
    local Pre = self:Image("Pre",unknown.type,unknown.W,unknown.H,-2)
    local jtfcentered,jtfgraph = createjtf(self,residuals,unknown,R,Pre) --includes the 2.0
    self.P:Stencil(stencilforexpression(jtfcentered[1]))
    
    self:createfunctionset("cost",terralib.newlist{centeredcost},graphcost)
    self:createfunctionset("applyJTJ",terralib.newlist{jtjexp},jtjgraph)
    --gradient with pre-conditioning
    self:createfunctionset("evalJTF",jtfcentered,jtfgraph)
    
    
    local precomputed = createprecomputed(self,"precompute",self.precomputed)
    self.P:Function("precompute",precomputed)
    
    if self.excludeexp then
        self:createfunctionset("exclude",terralib.newlist{self.excludeexp},{})
    end
    
    return self.P
end

function ProblemSpecAD:Exclude(exp)
    self.excludeexp = assert(ad.toexp(exp), "expected a AD expression")
end

local SampledImage = ad.newclass("SampledImage")
function SampledImage:__call(x,y,c)
    if c or self.op.imagebeingsampled.N == 1 then
        assert(not c or c < self.op.imagebeingsampled.N, "index out of bounds")
        return self.op(c or 0,x,y)
    else
        local r = {}
        for i = 0,self.op.imagebeingsampled.N - 1 do
            r[i+1] = self.op(i,x,y)
        end
        return ad.Vector(unpack(r))
    end
end
local function tosampledimage(im)
    if Image:is(im) then
        return ad.sampledimage(im)
    end
    return SampledImage:is(im) and im or nil
end
function ad.sampledimage(image,imagedx,imagedy)
    if imagedx then
        imagedx = assert(tosampledimage(imagedx), "expected an image or a sampled image as a derivative")
        imagedy = assert(tosampledimage(imagedy), "expected an image or a sampled image as a derivative")
    end
    local op = ad.newop("sampleimage_"..image.name)
    op.imagebeingsampled = image --not the best place to store this but other ways are more cumbersome
    op.config = { "c" }
    function op:generate(exp,args) error("sample image is not implemented directly") end
    function op:getpartials(exp)
        assert(imagedx and imagedy, "image derivatives are not defined for this image and cannot be used in autodiff")
        local x,y = unpack(exp:children())
        return terralib.newlist { imagedx(x,y,exp.config.c), imagedy(x,y,exp.config.c) }
    end
    return SampledImage:new { op = op }
end

for i = 2,12 do
    opt["float"..tostring(i)] = util.Vector(float,i)
end

opt.Dot = util.Dot

return opt
