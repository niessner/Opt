--terralib.settypeerrordebugcallback( function(fn) fn:printpretty() end )
opt = {} --anchor it in global namespace, otherwise it can be collected
local S = require("std")
local A = require("asdl")
local ffi = require("ffi")
local util = require("util")
ad = require("ad")
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

local List = terralib.newlist
A:Extern("imageindex",function(x) return type(x) == "number" or x == "alloc" end)
A:Extern("ExpLike",function(x) return ad.Exp:is(x) or ad.ExpVector:is(x) end)
A:Define [[
Dim = (string name, number size, number? _index) unique
IndexSpace = (Dim* dims) unique
Index = Offset(number* data) unique
      | GraphElement(any graph, string element) unique
ImageType = (IndexSpace ispace, TerraType scalartype, number channelcount) unique
Image = (string name, ImageType type, imageindex idx, boolean scalar, boolean isunknown)
ImageVector = (Image* images)
ProblemParam = ImageParam(ImageType imagetype, boolean isunknown)
             | ScalarParam(TerraType type)
             | GraphParam(TerraType type)
             attributes (string name, any idx)
VarDef =  ImageAccess(Image image,  Shape _shape, Index index, number channel) unique
       | BoundsAccess(Offset offset, number expand) unique
       | IndexValue(number dim, number shift_) unique
       | ParamValue(string name,TerraType type) unique
Graph = (string name)
GraphFunctionSpec = (Graph graph, ExpLike* results, Scatter* scatters)
Scatter = (Image image,Index index, number channel, Exp expression, string kind)
Condition = (IRNode* members)
IRNode = vectorload(ImageAccess value, number count)
       | sampleimage(Image image, number count, IRNode* children)
       | reduce(string op, IRNode* children)
       | vectorconstruct(IRNode* children)
       | vectorextract(IRNode* children, number channel)
       | load(ImageAccess value)
       | intrinsic(VarDef value)
       | const(number value)
       | vardecl(number constant)
       | varuse(IRNode* children)
       | apply(string op, function generator, IRNode * children, number? const)
         attributes (TerraType type, Shape shape, Condition? condition)
ProblemSpec = ()
ProblemSpecAD = ()
SampledImage = (table op)
GradientImage = (ImageAccess unknown, Exp expression, Image image)
UnknownType = (ImageParam* images)
ProblemFunctionType = CenteredFunction(IndexSpace ispace)
                    | GraphFunction(string graphname)
ProblemFunctions = (ProblemFunctionType typ, table functionmap)
]]
local Dim,IndexSpace,Index,Offset,GraphElement,ImageType,Image,ImageVector,ProblemParam,ImageParam,ScalarParam,GraphParam,VarDef,ImageAccess,BoundsAccess,IndexValue,ParamValue,Graph,GraphFunctionSpec,Scatter,Condition,IRNode,ProblemSpec,ProblemSpecAD,SampledImage, GradientImage,UnknownType = 
      A.Dim,A.IndexSpace,A.Index,A.Offset,A.GraphElement,A.ImageType,A.Image,A.ImageVector,A.ProblemParam,A.ImageParam,A.ScalarParam,A.GraphParam,A.VarDef,A.ImageAccess,A.BoundsAccess,A.IndexValue,A.ParamValue,A.Graph,A.GraphFunctionSpec,A.Scatter,A.Condition,A.IRNode,A.ProblemSpec,A.ProblemSpecAD,A.SampledImage,A.GradientImage,A.UnknownType

opt.PSpec = ProblemSpec
local PROBLEM_STAGES  = { inputs = 0, functions = 1 }
function opt.ProblemSpec()
    local ps = ProblemSpec()
    ps.parameters = terralib.newlist() -- ProblemParam*
    ps.names = {} -- name -> index in parameters list
    ps.functions = List() -- ProblemFunctions*
	ps.maxStencil = 0
	ps.stage = "inputs"
	ps.usepreconditioner = false
	return ps
end

function ProblemSpec:UsePreconditioner(v)
	self:Stage "inputs"
	self.usepreconditioner = v
end
function ProblemSpec:Stage(name)
    assert(PROBLEM_STAGES[self.stage] <= PROBLEM_STAGES[name], "all inputs must be specified before functions are added")
    self.stage = name
end

function ProblemSpec:registername(name)
    assert(not self.names[name],string.format("name %s already in use",name))
    self.names[name] = #self.parameters + 1
end

function ProblemParam:terratype() return self.type end
function ImageParam:terratype() return self.imagetype:terratype() end


function ProblemSpec:MaxStencil()
    self:Stage "functions"
	return self.maxStencil
end

function ProblemSpec:Stencil(stencil) 
    self:Stage "inputs"
	self.maxStencil = math.max(stencil, self.maxStencil)
end

function ProblemSpec:newparameter(p)
    assert(ProblemParam:is(p))
    self:registername(p.name)
    self.parameters:insert(p)
end

function ProblemSpec:ParameterType()
    self:Stage "functions"
    if not self.ProblemParameters then
        self.ProblemParameters = terralib.types.newstruct("ProblemParameters")
        self.ProblemParameters.entries:insert { "X" , self:UnknownType():terratype() }
        for i,p in ipairs(self.parameters) do
            local n,t = p.name,p:terratype()
            if not p.isunknown then self.ProblemParameters.entries:insert { n, t } end
        end
    end
    return self.ProblemParameters
end

function ProblemSpec:UnknownType()
    self:Stage "functions"
    if not self.Unknown then
        local images = List()
        for _,p in ipairs(self.parameters) do
            if p.isunknown then images:insert(p) end
        end
        self.Unknown = UnknownType(images)
    end
    return self.Unknown
end

function A.CenteredFunction:__tostring() return tostring(self.ispace) end
function A.GraphFunction:__tostring() return tostring(self.graphname) end

function ProblemSpec:Functions(ft, functions)
    self:Stage "functions"
    for k,v in pairs(functions) do
        v:gettype() -- check they typecheck now
    end
    if type(ft) == "string" then
        local idx = assert(self.names[ft],"graph not defined")
        assert(self.parameters[idx].kind == "GraphParam","expected a valid graph name")
        ft = A.GraphFunction(ft)
        self.usesgraphs = true
    else
        ft = A.CenteredFunction(ft)
    end
    if not functions.exclude then
        functions.exclude = macro(function() return `false end)
    end
    self.functions:insert(A.ProblemFunctions(ft, functions))
end
function ProblemSpec:UsesGraphs() return self.usesgraphs or false end

function ProblemSpec:Param(name,typ,idx)
    self:Stage "inputs"
    self:newparameter(ScalarParam(typ,name,idx))
end

function Dim:__tostring() return "Dim("..self.name..")" end

function opt.Dim(name, idx)
    idx = assert(tonumber(idx), "expected an index for this dimension")
    local size = tonumber(opt.dimensions[idx])
    return Dim(name,size,idx)
end

function IndexSpace:cardinality()
    local c = 1
    for i,d in ipairs(self.dims) do
        c = c * d.size
    end
    return c
end
function IndexSpace:init()
    self._string = self.dims:map(function(x) return x.name end):concat("_")
end
function IndexSpace:__tostring() return self._string end

function IndexSpace:ZeroOffset()
    if self._zerooffset then return self._zerooffset end
    local zeros = terralib.newlist()
    for i = 1,#self.dims do
        zeros:insert(0)
    end
    self._zerooffset = Offset(zeros)
    return self._zerooffset
end

function IndexSpace:indextype()
    if self._terratype then return self._terratype end
    local dims = self.dims
    assert(#dims > 0, "index space must have at least 1 dimension")
    local struct Index {}
    self._terratype = Index
    
    local params = terralib.newlist()
    local fieldnames = terralib.newlist()
    for i = 1,#dims do
        params:insert(symbol(int,"d"..tostring(i-1)))
        local n = "d"..tostring(i-1)
        fieldnames:insert(n)
        Index.entries:insert { n, int }
    end

    terra Index.metamethods.__apply(self : &Index, [params])
        var rhs : Index
        escape
            for i = 1,#dims do
                emit quote  
                    rhs.[fieldnames[i]] = self.[fieldnames[i]] + [params[i]]
                end 
            end
        end
        return rhs
    end
    local function genoffset(self)
        local s = 1
        local offset = `self.d0
        for i = 2,#dims do
            s = s * dims[i-1].size
            offset = `s*self.[fieldnames[i]] + offset
        end
        return offset
    end
    terra Index:tooffset()
        return [genoffset(self)]
    end
    local function genbounds(self,e)
        local valid
        for i = 1, #dims do
            local n = fieldnames[i]
            local v = `self.[n] >= e and self.[n] < [dims[i].size] - e
            if valid then
                valid = `valid and v
            else
                valid = v
            end
        end
        return valid
    end
    terra Index:InBounds() return [ genbounds(self,0) ] end
    terra Index:InBoundsExpanded(e : int) return [ genbounds(self,e) ] end
    
    if #dims <= 3 then
        local dimnames = "xyz"
        terra Index:initFromCUDAParams() : bool
            escape
                local lhs,rhs = terralib.newlist(),terralib.newlist()
                local valid = `true
                for i = 1,#dims do
                    local name = dimnames:sub(i,i)
                    local l = `self.[fieldnames[i]]
                    local r = `blockDim.[name] * blockIdx.[name] + threadIdx.[name]
                    lhs:insert(l)
                    rhs:insert(r)
                    valid = `valid and l < [dims[i].size]
                end
                emit quote
                    [lhs] = [rhs]
                    return valid
                end
            end  
        end
    end
    return Index
end

function ImageType:usestexture() -- texture, 2D texture
    local c = self.channelcount
    if use_bindless_texture and self.scalartype == float and 
       (c == 1 or c == 2 or c == 4) then
       if use_pitched_memory and #self.ispace.dims == 2 then
            local floatstride = self.ispace.dims[1].size*c
            local m = floatstride % 32
            if m ~= 0 then
                print(string.format("***** falling back to linear texture (width in floats %d %% 32 == %d)", floatstride , m))
            end
            return true,m == 0
       end
       return true, false 
    end
    return false, false 
end

local cd = macro(function(apicall) return quote
    var r = apicall
    if r ~= 0 then  
        C.printf("Cuda reported error %d: %s\n",r, C.cudaGetErrorString(r))
        return r
    end
end end)

local terra wrapBindlessTexture(data : &uint8, channelcount : int, width : int, height : int) : C.cudaTextureObject_t
    var res_desc : C.cudaResourceDesc
    C.memset(&res_desc, 0, sizeof(C.cudaResourceDesc))

    res_desc.res.linear.devPtr = data
    res_desc.res.linear.desc.f = C.cudaChannelFormatKindFloat
    res_desc.res.linear.desc.x = 32  -- bits per channel
    if  channelcount > 1 then
        res_desc.res.linear.desc.y = 32
    end
    if channelcount == 4 then
        res_desc.res.linear.desc.z = 32
        res_desc.res.linear.desc.w = 32
    end

    if height ~= 0 then
        res_desc.resType = C.cudaResourceTypePitch2D
        res_desc.res.pitch2D.width = width
        res_desc.res.pitch2D.height = height
        res_desc.res.pitch2D.pitchInBytes = sizeof(float)*width*channelcount
    else
        res_desc.resType = C.cudaResourceTypeLinear
        res_desc.res.linear.sizeInBytes = sizeof(float)*width*channelcount
    end

    var tex_desc : C.cudaTextureDesc
    C.memset(&tex_desc, 0, sizeof(C.cudaTextureDesc))
    tex_desc.readMode       = C.cudaReadModeElementType

    var tex : C.cudaTextureObject_t = 0;
    cd(C.cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nil))

    return tex
end

function ImageType:ElementType() return util.Vector(self.scalartype,self.channelcount) end
function ImageType:LoadAsVector() return self.channelcount == 2 or self.channelcount == 4 end
function ImageType:terratype()
    if self._terratype then return self._terratype end
    local scalartype = self.scalartype
    local vectortype = self:ElementType()
    local struct Image {
        data : &vectortype
        tex  : C.cudaTextureObject_t;
    }
    self._terratype = Image
    local channelcount = self.channelcount
    local textured,pitched = self:usestexture()
    local Index = self.ispace:indextype()
    function Image.metamethods.__typename()
	  return string.format("Image(%s,%s,%d)",tostring(self.scalartype),tostring(self.ispace),channelcount)
	end

    local VT = &vector(scalartype,channelcount)    
    -- reads
    if pitched then
        terra Image.metamethods.__apply(self : &Image, idx : Index) : vectortype
            var read = terralib.asm([tuple(float,float,float,float)],
                "tex.2d.v4.f32.s32  {$0,$1,$2,$3}, [$4,{$5,$6}];",
                "=f,=f,=f,=f,l,r,r",false, self.tex, idx.d0,idx.d1)
            return @[&vectortype](&read)
        end
    elseif textured then         
        terra Image.metamethods.__apply(self : &Image, idx : Index) : vectortype
            var read = terralib.asm([tuple(float,float,float,float)],
                "tex.1d.v4.f32.s32  {$0,$1,$2,$3}, [$4,{$5}];",
                "=f,=f,=f,=f,l,r",false, self.tex,idx:tooffset())
            return @[&vectortype](&read)
        end
    elseif self:LoadAsVector() then
        terra Image.metamethods.__apply(self : &Image, idx : Index) : vectortype
            var a = VT(self.data)[idx:tooffset()]
            return @[&vectortype](&a)
        end
    else    
        terra Image.metamethods.__apply(self : &Image, idx : Index) : vectortype
            return self.data[idx:tooffset()]
        end
    end
    -- writes
    if self:LoadAsVector() then
        terra Image.metamethods.__update(self : &Image, idx : Index, v : vectortype)
            VT(self.data)[idx:tooffset()] = @VT(&v)
        end
    else
        terra Image.metamethods.__update(self : &Image, idx : Index, v : vectortype)
            self.data[idx:tooffset()] = v
        end
    end
    
	terra Image:atomicAddChannel(idx : Index, c : int32, v : scalartype)
	    var addr : &scalartype = &self.data[idx:tooffset()].data[c]
	    util.atomicAdd(addr,v)
	end
	terra Image:atomicAdd(idx : Index, v : vectortype) -- only for hand written stuff
	    for i = 0,channelcount do
	        self:atomicAddChannel(idx,i,v(i))
	    end
	end
	
    terra Image:get(idx : Index)
        var v : vectortype = 0.f
        if idx:InBounds() then
            v = self(idx)
        end
        return v
    end
    -- lerps for 2D images only
    if 2 == #self.ispace.dims then
        local terra lerp(v0 : vectortype, v1 : vectortype, t : float)
            return (1.f - t)*v0 + t*v1
        end
        terra Image:sample(x : float, y : float)
            var x0 : int, x1 : int = opt.math.floor(x),opt.math.ceil(x)
            var y0 : int, y1 : int = opt.math.floor(y),opt.math.ceil(y)
            var xn,yn = x - x0,y - y0
            var u = lerp(self:get( Index {x0,y0} ),self:get( Index {x1,y0} ),xn)
            var b = lerp(self:get( Index {x0,y1} ),self:get( Index {x1,y1} ),xn)
            return lerp(u,b,yn)
        end
    end
    local cardinality = self.ispace:cardinality()
    terra Image:totalbytes() return sizeof(vectortype)*cardinality end
	terra Image:initCPU()
		self.data = [&vectortype](C.malloc(self:totalbytes()))
		C.memset(self.data,0,self:totalbytes())
	end
	terra Image:initGPU()
        var data : &uint8
        C.cudaMalloc([&&opaque](&data), self:totalbytes())
        C.cudaMemset([&opaque](data), 0, self:totalbytes())
        self:initFromGPUptr(data)
    end
    terra Image:initFromGPUptr( ptr : &uint8 )
        self.data = nil
        self:setGPUptr(ptr)
    end
    if textured then
        local W,H = cardinality,0
        if pitched then
            W,H = self.ispace.dims[1].size,self.ispace.dims[2].size
        end
        terra Image:setGPUptr(ptr : &uint8)
            if [&uint8](self.data) ~= ptr then
                if self.data ~= nil then
                    cd(C.cudaDestroyTextureObject(self.tex))
                end
                self.tex = wrapBindlessTexture(ptr, channelcount, W, H)
            end
            self.data = [&vectortype](ptr)
        end
    else
        terra Image:setGPUptr(ptr : &uint8) self.data = [&vectortype](ptr) end
    end
    return Image
end

function UnknownType:init()
    self.ispaces = List()
    self.ispacetoimages,self.ispacesizes = {},{} -- IndexSpace -> ImageParam
    for i,ip in ipairs(self.images) do
        assert(ip.imagetype.scalartype == float, "unknowns must be floating point numbers")
        local ispace = ip.imagetype.ispace
        if not self.ispacetoimages[ispace] then
            self.ispacetoimages[ispace],self.ispacesizes[ispace] = List(),0
            self.ispaces:insert(ispace)
        end
        self.ispacetoimages[ispace]:insert(ip)
        self.ispacesizes[ispace] = self.ispacesizes[ispace] + ip.imagetype.channelcount 
    end
end
function UnknownType:IndexSpaces()
    return self.ispaces
end
function UnknownType:VectorTypeForIndexSpace(ispace)
    return util.Vector(float,assert(self.ispacesizes[ispace],"unused ispace"))
end

function UnknownType:terratype()
    if self._terratype then return self._terratype end
    self._terratype = terralib.types.newstruct("UnknownType")
    local T = self._terratype
    local images = self.images
    for i,ip in ipairs(images) do
        T.entries:insert { ip.name, ip.imagetype:terratype() }
    end
    terra T:initGPU()
        escape
            for i,ip in ipairs(images) do
                emit quote self.[ip.name]:initGPU() end
            end
        end
    end
    for _,ispace in ipairs(self:IndexSpaces()) do
        local Index = ispace:indextype()
        local ispaceimages = self.ispacetoimages[ispace]
        local VT = self:VectorTypeForIndexSpace(ispace)
        terra T.metamethods.__apply(self : &T, idx : Index) : VT
            var r : VT
            escape
                local off = 0
                for _,im in ipairs(ispaceimages) do
                    emit quote
                        var d = self.[im.name](idx)
                        for i = 0,im.imagetype.channelcount do
                            r.data[off+i] = d.data[i]
                        end
                    end
                    off = off + im.imagetype.channelcount
                end
            end
            return r
        end
        terra T.metamethods.__update(self : &T, idx : Index, v : VT)
            escape
                local off = 0
                for _,im in ipairs(ispaceimages) do
                    emit quote
                        var d : im.imagetype:ElementType()
                        for i = 0,im.imagetype.channelcount do
                            d.data[i] = v.data[off+i]
                        end
                        self.[im.name](idx) = d
                    end
                    off = off + im.imagetype.channelcount
                end
            end
        end
        self._terratype.metamethods.__apply:printpretty()
        self._terratype.metamethods.__update:printpretty()
    end
    
    return self._terratype
end

local unity = Dim("1",1)
local function todim(d)
    return Dim:is(d) and d or d == 1 and unity
end

local function tovalidimagetype(typ)
    if not terralib.types.istype(typ) then return nil end
    if util.isvectortype(typ) then
        return typ.metamethods.type, typ.metamethods.N
    elseif typ:isarithmetic() then
        return typ, 1
    end
end

function ProblemSpec:ImageType(typ,ispace)
    local scalartype,channelcount = tovalidimagetype(typ,"expected a number or an array of numbers")
    assert(scalartype,"expected a number or an array of numbers")
    return ImageType(ispace,scalartype,channelcount) 
end

local function toispace(ispace)
    if not IndexSpace:is(ispace) then -- for handwritten API
        assert(#ispace > 0, "expected at least one dimension")
        ispace = IndexSpace(List(ispace)) 
    end
    return ispace
end


function ProblemSpec:Image(name,typ,ispace,idx)
    self:Stage "inputs"
    self:newparameter(ImageParam(self:ImageType(typ,toispace(ispace)),name == "X",name,idx))
end


function ProblemSpec:Graph(name, idx, ...)
    self:Stage "inputs"
    local GraphType = terralib.types.newstruct(name)
    GraphType.entries:insert ( {"N",int32} )
    
    local mm = GraphType.metamethods
    mm.idx = idx -- the index into the graph size table
    mm.elements = terralib.newlist()
    for i = 1, select("#",...),3 do
        local name,dims,didx = select(i,...) --TODO: we don't bother to track the dimensions of these things now
        local ispace = toispace(dims)
        local Index = ispace:indextype()
        GraphType.entries:insert {name, &Index}
        mm.elements:insert( { name = name, type = Index, idx = assert(tonumber(didx))} )
    end
    self:newparameter(GraphParam(GraphType,name,idx))
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
		local b = terralib.currenttimeinseconds()
        local tbl = opt.problemSpecFromFile(problemmetadata.filename)
        assert(ProblemSpec:is(tbl))
		local result = compilePlan(tbl,problemmetadata.kind)
		local e = terralib.currenttimeinseconds()
		print("compile time: ",e - b)
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

function Offset:__tostring() return string.format("(%s)",self.data:map(tostring):concat(",")) end
function GraphElement:__tostring() return ("%s_%s"):format(tostring(self.graph), self.element) end

function VarDef:asvar() return ad.v[self] end

function ImageAccess:__tostring()
    local r = ("%s_%s_%s"):format(self.image.name,tostring(self.index),self.channel)
    if self:shape() ~= ad.scalar then
        r = r .. ("_%s"):format(tostring(self:shape()))
    end
    return r
end
function BoundsAccess:__tostring() return ("bounds_%s_%d"):format(tostring(self.offset),self.expand) end
function IndexValue:__tostring() return ({[0] = "i","j","k"})[self.dim] end
function ParamValue:__tostring() return "param_"..self.name end

function ImageAccess:shape() return self._shape end -- implementing AD's API for keys

local emptygradient = {}
function ImageAccess:gradient()
    if self.image.gradientimages then
        assert(Offset:is(self.index),"NYI - support for graphs")
        local gt = {}
        for i,im in ipairs(self.image.gradientimages) do
            local k = im.unknown:shift(self.index)
            local v = ad.Const:is(im.expression) and im.expression or im.image(self.index)
            gt[k] = v
        end
        return gt
    end
    return emptygradient
 end
 
function ad.Index(d) return IndexValue(d,0):asvar() end
 

function ad.ProblemSpec()
    local ps = ProblemSpecAD()
    ps.P,ps.nametoimage,ps.precomputed = opt.ProblemSpec(), {}, List{}
    return ps
end
function ProblemSpecAD:UsePreconditioner(v)
	self.P:UsePreconditioner(v)
end

-- Z: this will eventually be opt.Image, but that is currently used by our direct methods
-- so this is going in the ad table for now
-- small workaround: idx > 0 means it is part of ProblemSpec struct
-- idx < 0 means that it is the -idx-th argument to the function begin generated after the Parameters struct. (.e.g -1 is the first argument) 
function ProblemSpecAD:Image(name,typ,dims,idx)
    if not terralib.types.istype(typ) then
        typ,ispace,idx = float,typ,ispace --shift arguments left
    end
    local ispace = toispace(dims)
    assert( (type(idx) == "number" and idx >= 0) or idx == "alloc", "expected an index number") -- alloc indicates that the solver should allocate the image as an intermediate
    self.P:Image(name,typ,ispace,idx)
    local r = Image(name,self.P:ImageType(typ,ispace),idx,not util.isvectortype(typ),name == "X")
    self.nametoimage[name] = r
    return r
end
function ProblemSpecAD:ImageArgument(name,imagetype,argpos)
    return Image(name,imagetype,-argpos,false,false)
end

function Image:__tostring() return self.name end

function ProblemSpecAD:ComputedImage(name,dims,exp)
    if ad.ExpVector:is(exp) then
        local imgs = terralib.newlist()
        for i,e in ipairs(exp:expressions()) do
            imgs:insert(self:ComputedImage(name.."_"..tostring(i-1),dims,e))
        end
        return ImageVector(imgs)
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
    local im = self:Image(name,float,dims,"alloc")
    local gradients = exp:gradient(unknowns:map(function(x) return ad.v[x] end))
    im.gradientimages = terralib.newlist()
    for i,g in ipairs(gradients) do
        local u = unknowns[i]
        local gim = self:Image(name.."_d_"..tostring(u),float,dims,"alloc")
        im.gradientimages:insert(GradientImage(u,g,gim))
    end
    im.expression = exp
    self.precomputed:insert(im)
    return im
end

function Graph:__tostring() return self.name end
function ProblemSpecAD:Graph(name,idx,...)
    self.P:Graph(name,idx,...)
    local g = Graph(name)
    for i = 1, select("#",...),3 do
        local name,dims,didx = select(i,...)
        g[name] = GraphElement(g,name)
    end
    return g
end

function ProblemSpecAD:Param(name,typ,idx)
    self.P:Param(name,typ,idx)
    return ParamValue(name,typ):asvar()
end

function Image:DimCount() return #self.type.ispace.dims end
function Image:__call(first,...)
    local index,c
    if GraphElement:is(first) or Offset:is(first) then
        index = first
        c = ...
    else
        local o = terralib.newlist { (assert(first,"no arguments?")) }
        for i = 1,self:DimCount() - 1 do
            o:insert((select(i,...)))
        end
        index = Offset(o)
        c = select(self:DimCount(), ...)
    end
    c = tonumber(c)
    assert(not c or c < self.type.channelcount, "channel outside of range")
    if self.scalar or c then
        return ImageAccess(self,ad.scalar,index,c or 0):asvar()
    else
        local r = {}
        for i = 1,self.type.channelcount do
            r[i] = ImageAccess(self,ad.scalar,index,i-1):asvar()
        end
        return ad.Vector(unpack(r))
    end
end
 -- wrapper for many images in a vector, just implements the __call methodf for Images Image:
function ImageVector:__call(...)
    local args = {...}
    local channelindex = self.images[1]:DimCount() + 1
    if #args == channelindex then
        local c = args[channelindex]
        assert(c < #self.images, "channel outside of range")
        return self.images[c+1](unpack(args,1,channelindex-1))
    end
    local result = self.images:map(function(im) return im(unpack(args)) end)
    return ad.Vector(unpack(result))
end

function opt.InBounds(...)
	return BoundsAccess(Offset(List{...}),0):asvar()
end
function opt.InBoundsExpanded(...)
    local args = {...}
    return BoundsAccess(Offset(List{unpack(args,1,#args-1)}),args[#args]):asvar()
end
function BoundsAccess:type() return bool end --implementing AD's API for keys


function VarDef:shift(o) return self end
function BoundsAccess:shift(o)
    return BoundsAccess(self.offset:shift(o),self.expand)
end
function ImageAccess:shift(o)
    assert(Offset:is(self.index), "cannot shift graph accesses!")
    return ImageAccess(self.image,self:shape(),self.index:shift(o),self.channel)
end
function IndexValue:shift(o)
    return IndexValue(self.dim,self.shift_ + assert(o.data[self.dim+1],"dim of index not in shift"))
end

local function shiftexp(exp,o)
    local function rename(a)
        return ad.v[a:shift(o)]
    end
    return exp:rename(rename)
end 

function Offset:IsZero()
    for i,o in ipairs(self.data) do
        if o ~= 0 then return false end
    end
    return true
end
function Offset:MaxValue()
    local m = 0
    for i, o in ipairs(self.data) do
        if o > m then m = o end
    end
    return m
end
function Offset:Invert()
    local r = terralib.newlist()
    for i,o in ipairs(self.data) do
        r[i] = -o
    end
    return Offset(r)
end
function Offset:shift(o)
    assert(Offset:is(o) and #o.data == #self.data)
    local ns = terralib.newlist()
    for i = 1,#self.data do
        ns[i] = self.data[i] + o.data[i]
    end
    return Offset(ns)
end
local function removeboundaries(exp)
    if ad.ExpVector:is(exp) or terralib.islist(exp) then return exp:map(removeboundaries) end
    local function nobounds(a)
        if BoundsAccess:is(a) and a.offset:IsZero() and a.expand == 0 then return ad.toexp(1)
        else return ad.v[a] end
    end
    return exp:rename(nobounds)
end

local nextirid = 0
function IRNode:init()
    self.id,nextirid = nextirid,nextirid+1
end
function Condition:create(members)
    local function cmp(a,b)
        if a.kind == "intrinsic" and b.kind ~= "intrinsic" then return true
        elseif a.kind ~= "intrinsic" and b.kind == "intrinsic" then return false
        else return a.id < b.id end
    end
    table.sort(members,cmp)
    return Condition(members)
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

local function createfunction(problemspec,name,Index,results,scatters)
    results = removeboundaries(results)
    
    local imageload = terralib.memoize(function(imageaccess)
        return A.vectorload(imageaccess,0,imageaccess.image.type:ElementType(),imageaccess:shape())
    end)
    local imagesample = terralib.memoize(function(image, shape, x, y)
        return A.sampleimage(image,0,List{x,y},image.scalar and image.type.scalartype or image.type:ElementType(),shape)
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
            n = ad.prod(n.const,unpack(factors))
            cond = Condition:create(conditions)
        end
        return A.reduce(op,List{vardecl,irmap(n)},float,vardecl.shape,cond)
    end
    irmap = terralib.memoize(function(e)
        if ad.ExpVector:is(e) then
            return A.vectorconstruct(e.data:map(irmap),util.Vector(float,#e.data),ad.scalar)
        elseif "Var" == e.kind then
            local a = e:key()
            if "ImageAccess" == a.kind then
                if not a.image.scalar then
                    local loadvec = imageload(ImageAccess(a.image,a:shape(),a.index,0))
                    loadvec.count = loadvec.count + 1
                    return A.vectorextract(List {loadvec}, a.channel, e:type(), a:shape())
                else
                    return A.load(a,e:type(),a:shape()) 
                end 
            else
                return A.intrinsic(a,e:type(),ad.scalar)
            end
        elseif "Const" == e.kind then
            return A.const(e.v,e:type(),ad.scalar)
        elseif "Apply" == e.kind then
            if use_split_sums and (e.op.name == "sum") and #e:children() > 2 then
                local vardecl = A.vardecl(e.const,float,e:shape())
                local children = List { vardecl }
                local varuse = A.varuse(children,float,e:shape())
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
                return A.vectorextract(List {sm}, e.const, e:type(), e:shape()) 
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
            return A.apply(e.op.name,gen,children,e.const,e:type(),e:shape()) 
        elseif "Reduce" == e.kind then
            local vardecl = A.vardecl(0,e:type(),e:shape()) 
            local arg = e.args[1]
            local red = A.reduce("sum",List { vardecl, irmap(arg) }, vardecl.type, arg:shape()) 
            local children = List { vardecl, red }
            local varuse = A.varuse(children,vardecl.type,e:shape())
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
            ir.condition = Condition:create(List{})
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
        if ir.kind == "reduce" then applyconditiontolist(Condition:create(List{}), ir.condition.members) end
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
        local currentcondition,currentshape = Condition:create(List{}), ad.scalar
        local function enter()
            state = setmetatable({}, {__index = state})
            readylists:insert(terralib.newlist())
        end
        enter() --initial root level for non-speculative moves
        
        for i,r in ipairs(roots) do
            if not state[r] then -- roots may appear in list more than once
                state[r] = "ready"
                readylists[#readylists]:insert(r)
            end
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
    
    local function printschedule(W,instructions,regcounts)
        W:write(string.format("schedule for %s -----------\n",name))
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
            W:write(("[%d]%sr%d : %s%s = %s\n"):format(regcounts[i],formatcondition(ir.condition),i,tostring(ir.type),tostring(ir.shape),formatinst(ir)))
            if instructions[i+1] and conditioncost(ir.condition,instructions[i+1].condition) ~= 0 then
                W:write("---------------------\n")
            end
        end
        W:write("----------------------\n")
    end
    
    if verboseAD or true then
        local W = io.open("log.txt","a")
        printschedule(W,instructions,regcounts)
        W:close()
    end
    
    local P = symbol(problemspec.P:ParameterType(),"P")
    local idx = symbol(Index,"idx")
    local midx = symbol(Index,"midx")
    
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
    local function boundcoversload(ba,off)
        --print("Bound Covers? ",ba,off)
        assert(#off.data == #ba.offset.data)
        for i = 1,#off.data do
            local o,b = off.data[i],ba.offset.data[i]
            if o < b - ba.expand or o > b + ba.expand then
                --print("no")
                return false
            end
        end
        --print("yes")
        return true
    end
    local function conditioncoversload(condition,off)
        if off:IsZero() then return true end
        for i,ir in ipairs(condition.members) do
            assert(ir.type == bool)
            if ir.kind == "intrinsic" and ir.value.kind == "BoundsAccess" and boundcoversload(ir.value,off) then
                return true
            end
        end
        return false
    end
    local function imageref(image)
        if image.idx == "alloc" or image.idx >= 0 then
            if image.isunknown then return `P.X.[image.name]
            else return `P.[image.name] end
        else
            if not extraimages[-image.idx] then
                extraimages[-image.idx] = symbol(image.type:terratype(),image.name)
            end
            return extraimages[-image.idx]
        end
    end
    local function graphref(ge)
        return `P.[ge.graph.name].[ge.element][idx]
    end
    local function createexp(ir)        
        if "const" == ir.kind then
            return `float(ir.value)
        elseif "intrinsic" == ir.kind then
            local a = ir.value
            if "BoundsAccess" == a.kind then--bounds calculation
                return `midx([a.offset.data]):InBoundsExpanded(a.expand)
            elseif "IndexValue" == a.kind then
                local n = "d"..tostring(a.dim)
                return `idx.[n] + a.shift_ 
            else assert("ParamValue" == a.kind)
                return `float(P.[a.name])
            end
        elseif "load" == ir.kind then
            local a = ir.value
            local im = imageref(a.image)
            if Offset:is(a.index) then
                if conditioncoversload(ir.condition,a.index) then
                   return `im(midx(a.index.data))(0) 
                else
                   return `im:get(midx(a.index.data))(0)
                end
            else
                local gr = graphref(a.index)
                return `im(gr)(0)
            end
        elseif "vectorload" == ir.kind then
            local a = ir.value
            local im = imageref(a.image)
            local s = symbol(("%s_%s"):format(a.image.name,tostring(a.index)))
            if Offset:is(a.index) then
                if conditioncoversload(ir.condition,a.index) then
                    statements:insert(quote
                        var [s] : a.image.type:ElementType() = im(midx(a.index.data))
                    end)
                else 
                    statements:insert(quote
                        var [s] : a.image.type:ElementType() = 0.f
                        if midx(a.index.data):InBounds() then
                            [s] = im(midx(a.index.data))
                        end
                    end)
                end
            else
                local gr = graphref(a.index)
                statements:insert(quote
                    var [s] : a.image.type:ElementType() = im(gr)
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
            local r = `im:sample(exps)
            if ir.image.scalar then
                r = `r(0)
            end
            return r
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

    local basecondition = Condition:create(List{})
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
                error("NYI - shapeiteration")
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
    
    emitshapechange(currentshape,ad.scalar) -- also blanks condition
    assert(#statementstack == 1)
    
    local expressions = irroots:map(emit)
    local resultexpressions,scatterexpressions = {unpack(expressions,1,#results)},{unpack(expressions,#results+1)}
        
    local scatterstatements = terralib.newlist()
    local function toidx(index)
        if Offset:is(index) then return `midx(index.data)
        else return graphref(index) end
    end
    for i,s in ipairs(scatters) do
        local image,exp = imageref(s.image),scatterexpressions[i]
        local index = toidx(s.index)
        local stmt
        if s.kind == "add" then
            assert(s.channel, "no channel on scatter?")
            stmt = `image:atomicAddChannel(index, s.channel, exp)
        else
            assert(s.kind == "set" and s.channel == 0, "set only works for single channel images")
            stmt = quote 
                image(index) = exp
            end
        end
        scatterstatements:insert(stmt)
    end

    local terra generatedfn([idx], [P], [extraimages])
        var [midx] = idx
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
            stencil = math.max(stencil,a.index:MaxValue()) 
        elseif "BoundsAccess" == a.kind then--bounds calculation
            stencil = math.max(stencil,a.offset:MaxValue()+a.expand)
        end
    end)
    return stencil
end
local noscatters = terralib.newlist()
function ProblemSpecAD:createfunctionset(name,results,graphfunctions)
    results,graphfunctions = terralib.newlist(results), terralib.newlist(graphfunctions)
    local ut = self.P:UnknownType().images[1].imagetype
    dprint("function for: ",name)
    local centered = createfunction(self,name,ut.ispace:indextype(),results,noscatters)
    local args = terralib.newlist()
    for i,g in ipairs(graphfunctions) do
        local gf = createfunction(self,("%s_graph%d"):format(name,i),int,g.results,g.scatters)
        args:insert(g.graph.name)
        args:insert(gf)
    end
    self.P:Function(name,centered,unpack(args))
end

-- the result is a data structure
-- { unknown = {<entry>}, graphs = { <graphobj> -> <entry> } }, where each entry is:
-- { expression = <ad-exp>, unknownaccess = [<ImagAccess>] }

local function classifyresiduals(uispace, Rs)
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
                    for i,im in ipairs(a.image.gradientimages) do
                        assert(Offset:is(a.index),"NYI - precomputed with graphs")
                        addunknown(im.unknown:shift(a.index))
                    end
                end
                local aclass = Offset:is(a.index) and a.image.type.ispace or a.index.graph
                assert(nil == classification or aclass == classification, "residual contains image reads from multiple domains")
                classification = aclass
            end
        end)
        local entry = { expression = exp, unknownaccesses = unknownaccesses }
        if not classification then
            classification = uispace
        end
        if IndexSpace:is(classification) then
            assert(classification == uispace,"energy must be defined over graphs or the index space of the unknown")
            exp:visit(function(a)
                if BoundsAccess:is(a) and #a.offset.data ~= #classification.dims then
                    error(string.format("%s does not match index space %s",a,classification))
                end
            end)
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
            r:insert(u.index:Invert())
        end
    end
    return r
end
local function unknownsforresidual(r,unknownsupport)
    return unknownsupport:map("shift",r)
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
        for channel = 0, unknown.type.channelcount-1 do
            local x = unknown(unknown.type.ispace:ZeroOffset(),channel)
            local residuals = residualsincludingX00(unknownsupport,channel)
            local columns = {}
            local nonzerounknowns = terralib.newlist()
        
            for _,r in ipairs(residuals) do
                local rexp = shiftexp(F,r)
                local condition,drdx00 = ad.splitcondition(rexp:d(x))
                lprintf(1,"instance:\ndr%d_%s/dx00[%d] = %s",rn,tostring(r),channel,tostring(drdx00))
                local unknowns = unknownsforresidual(r,unknownsupport)
                for _,u in ipairs(unknowns) do
                    local condition2, drdx_u = ad.splitcondition(rexp:d(unknown(u.index,u.channel)))
                    local exp = drdx00*drdx_u
                    lprintf(2,"term:\ndr%d_%s/dx%s[%d] = %s",rn,tostring(r),tostring(u.index),u.channel,tostring(drdx_u))
                    if not columns[u] then
                        columns[u] = 0
                        nonzerounknowns:insert(u)
                    end
                    local conditionmerged = condition*condition2
                    if not P_hat_c[conditionmerged] then
                        conditions:insert(conditionmerged)
                        P_hat_c[conditionmerged] = createzerolist(unknown.type.channelcount)
                    end
                    P_hat_c[conditionmerged][channel+1] = P_hat_c[conditionmerged][channel+1] + P(u.index,u.channel)*exp
                end
            end
        end
    end
    local P_hat = createzerolist(unknown.type.channelcount)
    for _,c in ipairs(conditions) do
        for i = 1,unknown.type.channelcount do
            P_hat[i] = P_hat[i] + c*P_hat_c[c][i]
        end
    end
    for i,p in ipairs(P_hat) do
        P_hat[i] = 2.0 * p
    end
    dprint("JTJ[nopoly] = ", ad.tostrings(P_hat))
    P_hat = ad.polysimplify(P_hat)
    dprint("JTJ[poly] = ", ad.tostrings(P_hat))
    return ad.Vector(unpack(P_hat))
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
                s =  Scatter(Ap_X,u.index,u.channel,ad.toexp(0),"add")
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
        jtjgraph:insert(GraphFunctionSpec(graph,List { result },scatters)) 
    end
    return jtjgraph
end
local function createjtj(residuals,unknown,P,Ap_X)
    return createjtjcentered(residuals,unknown,P),createjtjgraph(residuals,P,Ap_X)
end

local function createjtfcentered(problemSpec,residuals,unknown)
   local F_hat = createzerolist(unknown.type.channelcount) --gradient
   local P_hat = createzerolist(unknown.type.channelcount) --preconditioner
    
    for ridx,residual in ipairs(residuals.unknown) do
        local F, unknownsupport = residual.expression,residual.unknownaccesses
        lprintf(0,"-------------")
        lprintf(1,"R[%d] = %s",ridx,tostring(F))
        for channel = 0, unknown.type.channelcount-1 do
            local x = unknown(unknown.type.ispace:ZeroOffset(),channel)
            local residuals = residualsincludingX00(unknownsupport,channel)
            
            local sum = 0
            for _,f in ipairs(residuals) do
                local F_x = shiftexp(F,f)
                local dfdx00 = F_x:d(x)		-- entry of J^T
                local dfdx00F = dfdx00*F_x	-- entry of \gradF == J^TF
                F_hat[channel+1] = F_hat[channel+1] + dfdx00F			-- summing it up to get \gradF
    
                local dfdx00Sq = dfdx00*dfdx00	-- entry of Diag(J^TJ)
                P_hat[channel+1] = P_hat[channel+1] + dfdx00Sq			-- summing the pre-conditioner up
                --sum = sum + dfdx00F
                lprintf(2,"dR[%d]_%s/dx[%d] = %s",ridx,tostring(f),channel,tostring(dfdx00F))
            end
            
            
        end
    end
	for i = 1,unknown.type.channelcount do
	    if not problemSpec.P.usepreconditioner then
		    P_hat[i] = ad.toexp(1.0)
	    else
		    P_hat[i] = 2.0*P_hat[i]
		    P_hat[i] = ad.polysimplify(P_hat[i])
	    end
	    F_hat[i] = ad.polysimplify(2.0*F_hat[i])
	end
	dprint("JTF =", ad.tostrings({F_hat[1], F_hat[2], F_hat[3]}))
    return terralib.newlist{ ad.Vector(unpack(F_hat)), ad.Vector(unpack(P_hat)) }
end

local function createjtfgraph(residuals,R,Pre)
    local jtjgraph = terralib.newlist()
    for graph,terms in pairs(residuals.graphs) do
        local scatters = terralib.newlist() 
        local scattermap = { [R] = {}, [Pre] = {}}
        local function addscatter(im,u,exp)
            local s = scattermap[im][u]
            if not s then
                s =  Scatter(im,u.index,u.channel,ad.toexp(0),"add")
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
        jtjgraph:insert(GraphFunctionSpec(graph,List {},scatters))
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
        print(string.format("%s: %f sec\n",name,deltaTime))
    end
    lastTime = currentTime
end

local function creategradient(unknown,costexp)
    local unknownvars = unknowns(costexp)
    local gradient = costexp:gradient(unknownvars)

    dprint("grad expression")
    local names = table.concat(unknownvars:map(function(v) return tostring(v:key()) end),", ")
    dprint(names.." = "..ad.tostrings(gradient))
    
    local gradientsgathered = createzerolist(unknown.type.channelcount)
    for i,u in ipairs(unknownvars) do
        local a = u:key()
        local shift = shiftexp(gradient[i],a.index:Invert())
        gradientsgathered[a.channel+1] = gradientsgathered[a.channel+1] + shift
    end
    dprint("grad gather")
    dprint(ad.tostrings(gradientsgathered))
    return ad.Vector(gradientsgathered)
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
        graphwork:insert(GraphFunctionSpec(graph,List { sumsquared(terms) }, List()))
    end
    return sumsquared(residuals.unknown), graphwork
end

function createprecomputed(self,name,precomputedimages)
    local scatters = terralib.newlist()
    local ut = self.P:UnknownType().images[1].imagetype
    
    local zoff = ut.ispace:ZeroOffset()
    
    for i,im in ipairs(precomputedimages) do
        local expression = ad.polysimplify(im.expression)
        scatters:insert(Scatter(im, zoff, 0, im.expression, "set"))
        for _,gim in ipairs(im.gradientimages) do
            local gradientexpression = ad.polysimplify(gim.expression)
            if not ad.Const:is(gradientexpression) then
                scatters:insert(Scatter(gim.image, zoff, 0, gradientexpression, "set"))
            end
        end
    end
    
    return createfunction(self,name,ut.ispace:indextype(),terralib.newlist(),scatters)
end
local function extractresidualterms(...)
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
    return exp
end
function ProblemSpecAD:Cost(...)
    local terms = extractresidualterms(...)
    local unknown = assert(self.nametoimage.X, "unknown image X is not defined")

    local residuals = classifyresiduals(unknown.type.ispace,terms)
    
    local centeredcost,graphcost = createcost(residuals)
    
    self.P:Stencil(stencilforexpression(centeredcost))
    
    -- Not updated for graphs yet:    
    local P = self:ImageArgument("P",unknown.type,1)
    local Ap_X = self:ImageArgument("Ap_X",unknown.type,2)
    local jtjexp,jtjgraph = createjtj(residuals,unknown,P,Ap_X)
    self.P:Stencil(stencilforexpression(jtjexp))
    
    local R = self:ImageArgument("R",unknown.type,1)
    local Pre = self:ImageArgument("Pre",unknown.type,2)
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

function SampledImage:__call(x,y,c)
    if c or self.op.imagebeingsampled.type.channelcount == 1 then
        assert(not c or c < self.op.imagebeingsampled.type.channelcount, "index out of bounds")
        return self.op(c or 0,x,y)
    else
        local r = {}
        for i = 0,self.op.imagebeingsampled.type.channelcount - 1 do
            r[i+1] = self.op(i,x,y)
        end
        return ad.Vector(unpack(r))
    end
end
local function tosampledimage(im)
    if Image:is(im) then
        assert(im:DimCount() == 2, "sampled images must be 2D")
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
    op.hasconst = true
    function op:generate(exp,args) error("sample image is not implemented directly") end
    function op:getpartials(exp)
        assert(imagedx and imagedy, "image derivatives are not defined for this image and cannot be used in autodiff")
        local x,y = unpack(exp:children())
        return terralib.newlist { imagedx(x,y,exp.const), imagedy(x,y,exp.const) }
    end
    return SampledImage(op)
end

for i = 2,12 do
    opt["float"..tostring(i)] = util.Vector(float,i)
end

opt.Dot = util.Dot
opt.toispace = toispace
return opt
