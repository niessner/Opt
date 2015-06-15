local ad = {}
local C = terralib.includec("math.h")

local function newclass(name)
    local mt = { __name = name, variants = terralib.newlist() }
    mt.__index = mt
    function mt:is(obj)
        local omt = getmetatable(obj)
        while omt ~= nil do
            if self == omt then return true end
            omt = omt.__parent
        end
        return false
    end
    function mt:new(obj)
        obj = obj or {}
        setmetatable(obj,self)
        return obj
    end
    function mt:Variant(name)
        local vt = { kind = name, __parent = self }
        for k,v in pairs(self) do
            vt[k] = v
        end
        self.variants:insert(vt)
        vt.__index = vt
        return vt
    end
    setmetatable(mt, { __newindex = function(self,idx,v)
        rawset(self,idx,v)
        for i,j in ipairs(self.variants) do
            rawset(j,idx,v)
        end
    end })
 
    return mt
end

local Op = newclass("Op") -- a primitive operator like + or sin

local Exp = newclass("Exp") -- an expression involving primitives
local Var = Exp:Variant("Var") -- a variable
local Apply = Exp:Variant("Apply") -- an application
local Const = Exp:Variant("Const") -- a constant, C

local Shape = newclass("Shape") -- the shape of an expression Scalar, Vector, Matrix, etc.
                                -- eventually this may incorporate vectors derived from traversing graph edges, etc.
local Scalar = Shape:Variant("Scalar")
local Array = Shape:Variant("Array") 

local ScalarShape = Scalar:new {} --single representing the scalar shape
function Scalar:__tostring() return "float" end
function Array:__tostring() return ("%s[%d]"):format(tostring(self.element), self.N) end

function Scalar:order()
    return 0
end
function Array:order()
    return self.order_
end

local function toshape(s)
    if Shape:is(s) then return s
    else return nil end
end

local ArrayShape = terralib.memoize(function(shape,N)
    local element = assert(toshape(shape),"expected a shape")
    return Array:new { element = element, N = assert(tonumber(N),"expected a number"), order_ = element:order() + 1 }
end)

local empty = terralib.newlist {}
function Exp:children() return empty end
function Apply:children() return self.args end 

function Exp:shape() return self.shape_ end
function Apply:argumentshapes() return self.argumentshapes_ end

local function newapply(op,config,args)
    assert(not op.nparams or #args == op.nparams)
    local shape,argumentshapes = op:shapefromarguments(config,args:map("shape"))
    return Apply:new { op = op, config = config, args = args, argumentshapes_ = argumentshapes, shape_ = shape} 
end

local getconst = terralib.memoize(function(shape,n) return Const:new { shape_ = shape, v = n } end)
local function toexp(n)
    if n then 
        if Exp:is(n) then return n
        elseif type(n) == "number" then return getconst(ScalarShape,n)
        elseif type(n) == "table" then
            local mt = getmetatable(n)
            if mt and type(mt.__toadexp) == "function" then
                return toexp(mt.__toadexp(n))
            end
        end
    end
    return nil
end

local zero,one,negone = toexp(0),toexp(1),toexp(-1)
local function allconst(args)
    for i,a in ipairs(args) do
        if not Const:is(a) then return false end
    end
    return true
end

function Var:key() return self.key_ end

local function shouldcommute(a,b)
    if Const:is(a) then return true end
    if b:prec() < a:prec() then return true end
    return false
end

local commutes = { add = true, mul = true }
local assoc = { add = true, mul = true }
local function factors(e)
    if Apply:is(e) and e.op.name == "mul" then
        return e.args[1],e.args[2]
    end
    return e,one
end
local function simplifymore(op,config,args)
    local x,y,z = unpack(args)
    
    if #args == 2 and Apply:is(x) and Apply:is(y) and x.op.name == "select" and y.op.name == "select" and x.args[1] == y.args[1] then
        return ad.select(x.args[1],op(x.args[2],y.args[2]),op(x.args[3],y.args[3]))
    end
    if op.name == "select" and Apply:is(y) and y.op.name == "select" and y.args[3] == args[3] then
        return ad.select(ad.and_(x,y.args[1]),y.args[2],args[3]) 
    end
    if assoc[op.name] and Apply:is(x) and x.op.name == op.name
           and Const:is(x.args[2]) then -- (e + c) + ? -> e + (? + c)
        --print("assoc1")
        return op(x.args[1],op(x.args[2],y)) -- e0 + (e1 + c) ->  (e0 + e1) + c
    elseif assoc[op.name] and Apply:is(y) and y.op.name == op.name
           and Const:is(y.args[2]) then
        --print("assoc2")
        return op(op(x,y.args[1]),y.args[2])
    elseif commutes[op.name] and shouldcommute(x,y) then 
        --print("commute")
        return op(y,x) -- constants will be on rhs
    end
    
    if op.name == "mul" then
        if Apply:is(x) and x.op.name == "select" then
            if x.args[2] == one or x.args[2] == zero or x.args[3] == one or x.args[3] == zero then
                return ad.select(x.args[1],y*x.args[2],y*x.args[3])
            end
        elseif Apply:is(y) and y.op.name == "select" then
            if y.args[2] == one or y.args[2] == zero or y.args[3] == one or y.args[3] == zero then
                return ad.select(y.args[1],x*y.args[2],x*y.args[3])
            end
        elseif Const:is(y) and Apply:is(x) and x.op.name == "unm" then
            return x.args[1]*-y
        end
    elseif op.name == "add" then
        local x0,x1 = factors(x)
        local y0,y1 = factors(y)
        if x1 ~= one or y1 ~= one or x0 == y0 then 
            if x0 == y0 then return x0*(x1 + y1)
            elseif x1 == y1 then return (x0 + y0)*x1
            end
        end
    elseif op.name == "unm" and Apply:is(x) and x.op.name == "select" then
        return ad.select(x.args[1],-x.args[2],-x.args[3])
    end
    
end

local function simplify(op,config,args)
    local x,y,z = unpack(args)
    if commutes[op.name] and Const:is(x) then
        x,y = y,x
    end
    --print(x,y)
    if allconst(args) then
        local r = op:propagateconstants(op:shapefromarguments(config,args:map("shape")),config,args)
        if r then return r end
    end
    
    if op.name == "mul" then
        if y == one then return x
        elseif y == zero then return zero
        elseif y == negone then return -x
        end
    elseif op.name == "add" then
        if y == zero then return x end
    elseif op.name == "sub" then
        if x == y then return zero
        elseif y == zero then return x
        elseif x == zero then return -y 
        end
    elseif op.name == "div" then
        if y == one then return x
        elseif y == negone then return -x
        elseif x == y then return one 
        end
    elseif op.name == "select" and Const:is(x) then
        return  x.v ~= 0 and y or z
    elseif (op.name == "and_" or op.name == "or_") and x == y then
        return x
    end
    
    
    local r = simplifymore(op,config,args)
    if r then
        return r
    end
    return newapply(op,config,args)
end


local getapply = terralib.memoize(function(op,...)
    local config = terralib.newlist()
    local args = terralib.newlist()
    for i = 1,select('#',...) do
        local e = select(i,...)
        if i > op.nconfig then
            local exp = assert(toexp(e),"attempting to us a value that is not an expression")
            args:insert(exp)
        else
            config:insert(e)
        end
    end
    return simplify(op,config,args)
end)

local function toexps(...)
    local es = terralib.newlist {}
    for i = 1, select('#',...) do
        local e = select(i,...)
        es[i] = assert(toexp(e),"attempting use a value that is not an expression")
    end
    return es
end
    

local getvar = terralib.memoize(function(shape,key)
    return Var:new { shape_ = assert(toshape(shape)), key_ = assert(key) }
end)

-- generates variable names
local v = setmetatable({},{__index = function(self,idx)
    local r = getvar(ScalarShape,idx)
    self[idx] = r
    return r
end})

local x,y,z = v[1],v[2],v[3]


ad.v = v
ad.toexp = toexp
ad.newclass = newclass
ad.newvar = getvar

function ad.Vector(...)
    error("TODO: ad.Vector")
end

setmetatable(ad, { __index = function(self,idx) -- for just this file, auto-generate an op 
    local name = assert(type(idx) == "string" and idx)
    local op = Op:new { name = name }
    rawset(self,idx,op)
    return op
end })

function Op:__call(...)
    return getapply(self,...)
end

local function conform(args)
    local s = ScalarShape
    for i,a in ipairs(args) do
        s = s:meet(a)
        if not s then return nil end
    end
    return s
end

Op.nconfig = 0 -- overriden by custom operators

function Op:shapefromarguments(config,args) -- ops with special behavior should override this
    local r = conform(args)
    if not r then
        error("arguments do not conform "..table.concat(args:map(tostring),", "))
    end
    if self.nparams and self.nparams ~= #args then
        error( ("expected %d arguments but found %d in op %s"):format(self.nparams,#args,self.name))
    end
    return r,args:map(function() return r end) 
end


local getscalarpartials = terralib.memoize(function(exp)
    assert(Apply:is(exp) and exp.op.derivs)
    return exp.op.derivs:map("rename",exp:children())
end)


function Op:applyJ(exp,rhs) -- ops with special behavior override this
    local partials = getscalarpartials(exp)
    local r = zero
    assert(#partials == #rhs)
    for i,p in ipairs(partials) do
        r = r + p*rhs[i]
    end
    return r
end

function Op:define(generatorfn,...)
    self.nparams = debug.getinfo(generatorfn,"u").nparams
    self.generator = generatorfn
    self.derivs = toexps(...)
    return self
end
function Op:definecustom(tbl)
    for k,v in pairs(tbl) do
        self[k] = v
    end
end

function Op:getimpl()
    if self.impl then return self.impl end
    if not self.generator then return nil
    else
        local s = terralib.newlist()
        for i = 1,self.nparams do
            s:insert(symbol(float))
        end
        terra self.impl([s]) return [self.generator(unpack(s))] end
        return self.impl
    end
end
local function extractelement(arg,i)
    if type(arg) == "table" then
        return assert(arg[i+1],"undefined element")
    else assert(type(arg) == "number")
        return arg
    end
end    
function Op:propagateconstants(shape,config,args)
    local impl = self:getimpl()
    if not impl then return nil end
    local function run(shape,constarguments)    
        if Array:is(shape) then
            local r = terralib.newlist()
            for i = 1,shape.N do
                local extractedargs = constarguments:map(extractelement,i-1)
                local v = run(shape.element,extractedargs)
                r:insert(v)
            end
            return r
        else
            return impl(unpack(constarguments))
        end
    end
    local r = run(shape,args:map("v"))
    return getconst(shape,r)
end

function Op:__tostring() return self.name end

local mo = {"add","sub","mul","div"}
for i,o in ipairs(mo) do
    local function impl(a,b) return (ad[o])(a,b) end
    Exp["__"..o] = impl
end
Exp.__unm = function(a) return ad.unm(a) end

function Exp:rename(vars)
    local varsf = type(vars) == "function" and vars or function(k) return vars[k] end
    local visitcached
    local function visit(self) 
        if self.kind == "Apply" then
            return self.op(unpack(self.args:map(visitcached,vars)))
        elseif self.kind == "Const" then
            return self
        elseif self.kind == "Var" then
            local nv = toexp(varsf(self:key()))
            return assert(nv,
                          ("rename: unknown invalid mapping for variable %s which maps to %s"):format(tostring(self:key()),tostring(nv)))
        end
    end
    local cached = {} 
    function visitcached(self)
        local r = cached[self]
        if r then return r end
        r = visit(self)
        cached[self] = r
        return r
    end
    return visitcached(self)
end

local function countuses(es)
    local uses = {}
    local function count(e)
        uses[e] = (uses[e] or 0) + 1
        if uses[e] == 1 and e.kind == "Apply" then
            for i,a in ipairs(e.args) do count(a) end
        end
    end
    for i,a in ipairs(es) do 
        count(a)
    end
    for k,v in pairs(uses) do
       uses[k] = v > 1 or nil
    end
    return uses
end    



local infix = { add = {"+",1}, sub = {"-",1}, mul = {"*",2}, div = {"/",2} }

function Op:prec()
    if not infix[self.name] then return 3
    else return infix[self.name][2] end
end

function Exp:prec()
    if self.kind ~= "Apply" then return 3
    else return self.op:prec() end
end
local function expstostring(es)
    es = (terralib.islist(es) and es) or terralib.newlist(es)
    local n = 0
    local tbl = terralib.newlist()
    local manyuses = countuses(es)
    local emitted = {}
    local function prec(e) return (manyuses[e] and 3) or e:prec() end
    local emit
    local function emitprec(e,p)
        return (prec(e) < p and "(%s)" or "%s"):format(emit(e))
    end
    local function emitapp(e)
        if e.op.name == "unm" then
            return ("-%s"):format(emitprec(e.args[1],3))
        elseif infix[e.op.name] then
            local o,p = unpack(infix[e.op.name])
            return ("%s %s %s"):format(emitprec(e.args[1],p),o,emitprec(e.args[2],p+1))
        else
            local name = tostring(e.op)
            if e.op.nconfig > 0 then
                name = ("%s[%s]"):format(name,e.config:map(tostring):concat(","))
            end
            return ("%s(%s)"):format(name,e.args:map(emit):concat(","))
        end
    end
    function emit(e)
        if "Var" == e.kind then
            local k = e:key()
            return type(k) == "number" and ("v%d"):format(k) or tostring(e:key()) 
        elseif "Const" == e.kind then
            local function getpart(v)
                if type(v) == "number" then return tostring(v)
                else assert(terralib.islist(v))
                    return "{"..v:map(getpart):concat(",").."}"
                end
            end
            return getpart(e.v)
        elseif "Apply" == e.kind then
            if emitted[e] then return emitted[e] end
            local exp = emitapp(e)
            if manyuses[e] then
                local r = ("r%d"):format(n)
                n = n + 1
                emitted[e] = r
                tbl:insert(("  %s : %s = %s\n"):format(r,tostring(e:shape()),exp))            
                return r
            else
                return exp
            end
            
        end
    end
    local r = es:map(emit)
    r = r:concat(", ")
    if #tbl == 0 then return r end
    return ("let\n%sin\n  %s\nend\n"):format(tbl:concat(),r)
end
ad.tostrings = expstostring
function Exp:__tostring()
    return expstostring(terralib.newlist{self})
end

local function defaultgenerator(op) return op.generator end


local function emitprintexpression(e, recursionLevel)
    return quote
        escape
            if "Var" == e.kind then
                emit `printf([tostring(e:key())])
                emit `printf(", ")
            elseif "Const" == e.kind then
                emit `printf("%f, ", float(e.v))
            elseif "Apply" == e.kind then
                emit `printf([e.op.name])
                emit `printf("(")
                if recursionLevel > 0 then
                    for i,newE in ipairs(e.args) do
                        if "Var" == newE.kind then
                            emit `printf([tostring(newE:key())])
                            emit `printf(", ")
                        elseif "Const" == newE.kind then
                            emit `printf("%f, ", float(newE.v))
                        elseif "Apply" == newE.kind then
                            emit `printf([newE.op.name])
                        end
                        -- recursion doesn't work? :(
                        --emit `[emitprintexpression(newE, recursionLevel-1)]

                     --   local recursiveExpression = emitprintexpression(newE)
                     --   emit `[recursiveExpression]
                    end
                end
                emit `printf("), ")
            end
        end
    end
    
end

local Vectors = {}
function ad.isterravectortype(t) return Vectors[t] end
ad.TerraVector = terralib.memoize(function(typ,N)
    N = assert(tonumber(N),"expected a number")
    local ops = { "__sub","__add","__mul","__div" }
    local struct VecType { 
        data : typ[N]
    }
    Vectors[VecType] = true
    VecType.metamethods.type, VecType.metamethods.N = typ,N
    VecType.metamethods.__typename = function(self) return ("%s_%d"):format(tostring(self.metamethods.type),self.metamethods.N) end
    for i, op in ipairs(ops) do
        local i = symbol("i")
        local function template(ae,be)
            return quote
                var c : VecType
                for [i] = 0,N do
                    c.data[i] = operator(op,ae,be)
                end
                return c
            end
        end
        local terra doop(a : VecType, b : VecType) [template(`a.data[i],`b.data[i])]  end
        terra doop(a : typ, b : VecType) [template(`a,`b.data[i])]  end
        terra doop(a : VecType, b : typ) [template(`a.data[i],`b)]  end
       VecType.metamethods[op] = doop
    end
    terra VecType.metamethods.__unm(self : VecType)
        var c : VecType
        for i = 0,N do
            c.data[i] = -self.data[i]
        end
        return c
    end
    terra VecType:dot(b : VecType)
        var c : typ = 0
        for i = 0,N do
            c = c + self.data[i]*b.data[i]
        end
        return c
    end
	terra VecType:size()
        return N
    end
    terra VecType.methods.FromConstant(x : typ)
        var c : VecType
        for i = 0,N do
            c.data[i] = x
        end
        return c
    end
    VecType.metamethods.__apply = macro(function(self,idx) return `self.data[idx] end)
    VecType.metamethods.__cast = function(from,to,exp)
        if from:isarithmetic() and to == VecType then
            return `VecType.FromConstant(exp)
        end
        error(("unknown vector conversion %s to %s"):format(tostring(from),tostring(to)))
    end
    return VecType
end)

function ad.toterra(es,varmap,generatormap) 
    es = terralib.islist(es) and es or terralib.newlist(es)
    generatormap = generatormap or defaultgenerator
    local manyuses = countuses(es)
    local nvars = 0
    local results = terralib.newlist {}
    
    local statements = terralib.newlist()
    local emitted = {}
    local function emit(e)
        e = assert(toexp(e),"expected an expression but found ")
        if "Var" == e.kind then
            return assert(varmap(statements,e:key()),"no mapping for variable key "..tostring(e:key()))
        elseif "Const" == e.kind then
            return `float(e.v)
        elseif "Apply" == e.kind then
            if emitted[e] then return emitted[e] end
            local generator = assert(generatormap(e.op) or defaultgenerator(e.op))
            local exp = generator(unpack(e.args:map(emit)))
            if e.op.name == "select" or manyuses[e] then -- we're turning this off now
                local v = symbol(float,"e")
                emitted[e] = v
                statements:insert(quote 
                        var [v] = exp 
                        --[[
                        if not (exp == exp) then
                            printf("Nan found: %f, \n", exp)
                            [emitprintexpression(e,3)]
                            printf("\n")
                        end
                        ]]
                    end)
                return v
            else
                return exp
            end
        end
    end
    local tes = es:map(emit)
    return #statements == 0 and (`tes) or quote [statements] in [tes] end
end

function Exp:d(v)
    assert(Var:is(v))
    self.derivs = self.derivs or {}
    local r = self.derivs[v]
    if r then return r end
    r = self:calcd(v)
    self.derivs[v] = r
    return r
end

function Var:calcd(v)
   return self == v and toexp(1) or toexp(0)
end
function Const:calcd(v)
    return zero
end

local function sum(xs)
    local s = zero
    for i,x in ipairs(xs) do s = s + x end
    return s
end

function Apply:applyJ(rhs)
    assert(#rhs == #self.args)
    local r = self.op:applyJ(self,rhs)
    return r
end

function Apply:calcd(v)
    assert(v:shape() == ScalarShape, "currently deriv requires a scalar variable")
    local dargsdv = self.args:map("d",v)
    local r = self:applyJ(dargsdv)
    assert(r:shape():meet(self:shape()) == self:shape())
    return r
end

--calc d(thisexpress)/d(exps[1]) ... d(thisexpress)/d(exps[#exps]) (i.e. the gradient of this expression with relation to the inputs) 
--[[function Exp:gradient(exps)
    exps = terralib.islist(exps) and exps or terralib.newlist(exps)
    -- reverse mode ad, work backward from this expression, accumulate stuff.
    local tape = {} -- mapping from exp -> current accumulated derivative
    local postorder = terralib.newlist()
    local function visit(e)
        if tape[e] then return end
        tape[e] = zero
        for i,c in ipairs(e:children()) do visit(c) end
        postorder:insert(e)
    end
    visit(self)
    tape[self] = one -- the reverse ad 'seed'
    for i = #postorder,1,-1 do --reverse post order traversal from self to equation roots, all uses come before defs
        local e = postorder[i]
        local partials = e:partials()
        for j,c in ipairs(e:children()) do
            tape[c] = tape[c] + tape[e] * partials[j]
        end
    end
    return exps:map(function(e) return tape[e] or zero end)
end]]
function Exp:gradient(exps)
    return terralib.newlist(exps):map(function(e) return self:d(e) end)
end

function Scalar:meet(s)
    return s
end
function Array:meet(s)
    if s == self or self.element == s then
        return self
    elseif self:order() > s:order() then
        if self.element:meet(s) then
            return self
        end
    elseif s:order() > self:order() then
        if s.element:meet(self) then
            return s
        end
    end
    return nil
end

ad.add:define(function(x,y) return `x + y end,1,1)
ad.sub:define(function(x,y) return `x - y end,1,-1)
ad.mul:define(function(x,y) return `x * y end,y,x)
ad.div:define(function(x,y) return `x / y end,1/y,-x/(y*y))
ad.unm:define(function(x) return `-x end, -1)
ad.acos:define(function(x) return `C.acos(x) end, -1.0/ad.sqrt(1.0 - x*x))
ad.acosh:define(function(x) return `C.acosh(x) end, 1.0/ad.sqrt(x*x - 1.0))
ad.asin:define(function(x) return `C.asin(x) end, 1.0/ad.sqrt(1.0 - x*x))
ad.asinh:define(function(x) return `C.asinh(x) end, 1.0/ad.sqrt(x*x + 1.0))
ad.atan:define(function(x) return `C.atan(x) end, 1.0/(x*x + 1.0))
ad.atan2:define(function(x,y) return `C.atan2(x*x+y*y,y) end, y/(x*x+y*y),x/(x*x+y*y))
ad.cos:define(function(x) return `C.cos(x) end, -ad.sin(x))
ad.cosh:define(function(x) return `C.cosh(x) end, ad.sinh(x))
ad.exp:define(function(x) return `C.exp(x) end, ad.exp(x))
ad.log:define(function(x) return `C.log(x) end, 1.0/x)
ad.log10:define(function(x) return `C.log10(x) end, 1.0/(ad.log(10.0)*x))
ad.pow:define(function(x,y) return `C.pow(x,y) end, y*ad.pow(x,y)/x,ad.log(x)*ad.pow(x,y)) 
ad.sin:define(function(x) return `C.sin(x) end, ad.cos(x))
ad.sinh:define(function(x) return `C.sinh(a) end, ad.cosh(x))
ad.sqrt:define(function(x) return `C.sqrt(x) end, 1.0/(2.0*ad.sqrt(x)))
ad.tan:define(function(x) return `C.tan(x) end, 1.0 + ad.tan(x)*ad.tan(x))
ad.tanh:define(function(x) return `C.tanh(x) end, 1.0/(ad.cosh(x)*ad.cosh(x)))

ad.select:define(function(x,y,z) 
    return quote
        var r : float
        if bool(x) then
            r = y
        else    
            r = z
        end
    in r end
end,0,ad.select(x,1,0),ad.select(x,0,1))

ad.eq:define(function(x,y) return `int(x == y) end, 0,0)
ad.abs:define(function(x) return `terralib.select(x >= 0,x,-x) end, ad.select(ad.greatereq(x, 0),1,-1))

--ad.and6:define(function(x0, x1, x2, x3, x4, x5) return `int(x0) and int(x1) and int(x2) and int(x3) and int(x4) and int(x5) end,0,0,0,0,0,0)
ad.and_:define(function(x,y) return `int(x) and int(y) end, 0, 0)
ad.or_:define(function(x,y) return `int(x) or int(y) end, 0, 0)
ad.less:define(function(x,y) return `int(x < y) end, 0,0)
ad.greater:define(function(x,y) return `int(x > y) end, 0,0)
ad.lesseq:define(function(x,y) return `int(x <= y) end,0,0)
ad.greatereq:define(function(x,y) return `int(x >= y) end,0,0)
ad.not_:define(function(x) return `int(not bool(x)) end, 0)


local function rep(v,N)
    local r = terralib.newlist()
    for i = 1,N do r:insert(v) end
    return r
end

ad.construct:definecustom {
    shapefromarguments = function(self,config,argshapes)
        local a = conform(argshapes)
        return ArrayShape(a,#argshapes), rep(a,#argshapes)
    end;
    applyJ = function(self,exp,rhs)
        return ad.construct(unpack(rhs))
    end;
    propagateconstants = function(self,shape,config,args)
        return getconst(shape,args:map("v"))
    end
}

ad.extract:definecustom {
    nconfig = 1;
    shapefromarguments = function(self,config,argshapes)
        local idx = assert(tonumber(config[1]),"extract requires a number")
        assert(#argshapes == 1)
        local arg = argshapes[1]
        assert(Array:is(arg),"extract requires an array")
        assert(idx < arg.N, "index out of bounds")
        return arg.element,argshapes
    end;
    applyJ = function(self,exp,rhs)
        return ad.extract(exp.config[1],rhs[1])
    end;
    propagateconstants = function(self,shape,config,args)
        local n = config[1]
        return getconst(shape,extractelement(args[1].v,n))
    end;
}

setmetatable(ad,nil) -- remove special metatable that generates new blank ops
ad.Var,ad.Apply,ad.Const,ad.Exp = Var, Apply, Const, Exp


local x,y,z = ad.v.x,ad.v.y,ad.v.z
print(expstostring(ad.atan2.derivs))
assert(y == y)
assert(ad.cos == ad.cos)
local r = ad.sin(x) + ad.cos(y) + ad.cos(y)
print(r)
print(expstostring( terralib.newlist { r, r + 1} ))
print(x*-(-y+1)/4)
print(r:rename({x = x+y,y = x+y}))
local e = 2*x*x*x*3 -- - y*x
print("WHAT",(ad.sin(x)):d(x))
print("HI",e:d(x)*3+(4*x)*x)


local w = x*x

local exp = (3*w + 4*w + 3*w*z)

print(unpack((3*x*x*x + x):d(x)))

local g = exp:gradient({x,y,z})
print("g",expstostring(g))

local m = { x = symbol("x"), y = symbol("y"), z = symbol("z")}
local t = ad.toterra(g, function(stmts,x) return m[x] end)

t:printpretty()

local float3 = ArrayShape(ScalarShape,3)

local x3 = ad.newvar(float3,"x3")

print(x3)
print( (x3*x3+x*x):d(x) )


local v = ad.construct(x*x,2,3*x*x)
local v = v+ad.extract(1,v)

print(v:d(x))
print(ad.construct(1,2,3)+4)

return ad