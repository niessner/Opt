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

local Shape = newclass("Shape")

local newshape = terralib.memoize(function(...) return Shape:new { keys = terralib.newlist {...} } end)

function Shape:fromkeys(...) return newshape(...) end
ad.scalar = Shape:fromkeys()
 
function Shape:isprefixof(rhs)
    if #self.keys > #rhs.keys then return false end
    for i,k in ipairs(self.keys) do
        if k ~= rhs.keys[i] then return false end
    end
    return true
end
local function joinshapes(shapes)
    local longest = ad.scalar
    for i,s in ipairs(shapes) do
        assert(Shape:is(s),"not a shape")
        if #s.keys > #longest.keys then
            longest = s
        end
    end
    for i = 1,#longest.keys do
        local lk = longest.keys[i]
        for _,s in ipairs(shapes) do
            local k = s.keys[i]
            if k and k ~= lk then return nil end 
        end
    end
    return longest
end
function Shape:fromreduction()
    if #self.keys == 0 then return nil end
    return Shape:fromkeys(unpack(self.keys,1,#self.keys-1))
end
function Shape:__tostring()
    return "{"..table.concat(self.keys:map(tostring),",").."}"
end

local Exp = newclass("Exp") -- an expression involving primitives
local Var = Exp:Variant("Var") -- a variable
local Apply = Exp:Variant("Apply") -- an application
local Const = Exp:Variant("Const") -- a constant, C
local Reduce = Exp:Variant("Reduce") -- summation across the inner-most shape
local ExpVector = newclass("ExpVector")

local nextid = 0
local function allocid()
    nextid = nextid + 1
    return nextid
end

local function sortexpressions(es)
    local function order(a,b)
        return a.id < b.id
    end
    table.sort(es,order)
    return es
end

local empty = terralib.newlist {}
function Exp:children() return empty end
function Apply:children() return self.args end 
function Reduce:children() return self.args end

function Exp:type() 
    assert(self.type_ == bool or self.type_ == float) 
    return self.type_ 
end
function Exp:shape()
    assert(Shape:is(self.shape_),"not a shape?")
    return self.shape_
end
function Const:__tostring() return tostring(self.v) end

local function newapply(op,config,args)
    assert(not op.nparams or #args == op.nparams)
    assert(type(config) == "table")
    local id = allocid()
    return Apply:new { op = op, args = args, config = config, id = id, type_ = op:propagatetype(args), shape_ = joinshapes(args:map("shape")) }
end

local getconst = terralib.memoize(function(n) return Const:new { v = n, id = allocid(), type_ = float, shape_ = ad.scalar } end)
local function toexp(n)
    if n then 
        if Exp:is(n) then return n
        elseif type(n) == "number" then return getconst(n)
        elseif type(n) == "table" then
            local mt = getmetatable(n)
            if mt and type(mt.__toadexp) == "function" then
                return toexp(mt.__toadexp(n))
            end
        end
    end
    return nil
end
local getreduce = terralib.memoize(function(v) 
    assert(Exp:is(v))
    local ns = assert(v:shape():fromreduction(),"attempting to reduce a scalar value") 
    local typ = v:type()
    return Reduce:new { args = terralib.newlist{v}, v = v, type_ = typ, shape_ = ns, id = allocid() }
end)
 
local zero,one,negone = toexp(0),toexp(1),toexp(-1)
local function allconst(args)
    for i,a in ipairs(args) do
        if not Const:is(a) then return false end
    end
    return true
end

function Var:key() return self.key_ end

--[[
  cases: sum + sum -> turn each child into a factor (c,[t]) and join
  cases: sum + prod -> use sum + sum
  cases: sum + pow -> use sum + sum
  
  cases: prod*prod -> new factor with combined terms turn each term into pow (c,t) (done)
  cases: prod*pow -> use prod*prod (done)
  
  cases: prod*sum -> distribute prod into sum
]]

local function asprod(exp)
    if Apply:is(exp) and exp.op.name == "prod" then
        return exp.config.c, exp:children()
    elseif Const:is(exp) then
        return exp.v, empty
    else
        return 1.0,terralib.newlist { exp }
    end
end
local function aspowc(exp)
    if Apply:is(exp) and exp.op.name == "powc" then
        return exp.config.c, exp:children()[1]
    else
        return 1.0,exp
    end
end

local polycache = terralib.memoize(function(op,c,...)
    local args,config = terralib.newlist {...}, { c = c }
    return newapply(op,config,args)
end)

local function orderedexpressionkeys(tbl)
    local keys = terralib.newlist()
    for k,v in pairs(tbl) do
        keys:insert(k)
    end
    sortexpressions(keys)
    return keys
end


local function simplify(op,config,args)
    if allconst(args) then
        local r = op:propagateconstant(args:map("v"))
        if r then return r end
    end
    
    if op.name == "sum" then
        local root = {}
        local function lookup(es)
            local node = root
            for _,e in ipairs(es) do
                local next = node[e]
                if not next then
                    next = {}
                    node[e] = next
                end
                node = next
            end
            return assert(node)
        end
        
        local termnodes = terralib.newlist()
        
        local function insertall(args)
            for i,a in ipairs(args) do
                if Const:is(a) then
                    config.c = config.c + a.v
                elseif Apply:is(a) and a.op.name == "sum" then
                    config.c = config.c + a.config.c
                    insertall(a.args)
                else
                    local c,aa = asprod(a)
                    local tbl = lookup(aa)
                    if not tbl.value then
                        tbl.c,tbl.value = 0,aa
                        termnodes:insert(tbl)
                    end
                    tbl.c = tbl.c + c
                end
            end
        end
        insertall(args)
    
        if #termnodes == 0 then return toexp(config.c) end
        local terms = terralib.newlist()
        for _,t in ipairs(termnodes) do
            if t.c ~= 0 then
                terms:insert(ad.prod(t.c,unpack(t.value)))
            end
        end
        if config.c == 0.0 and #terms == 1 then return terms[1] end
        return polycache(op,config.c,unpack(sortexpressions(terms)))
    elseif op.name == "prod" then
        local expmap = {} -- maps each term to the power it has
        local function insertall(args)
            for i,a in ipairs(args) do
                local c, es = asprod(a)
                config.c = config.c * c
                for _,e in ipairs(es) do
                    local c,ep = aspowc(e)
                    expmap[ep] = (expmap[ep] or 0) + c
                end
            end
        end
        insertall(args)
        if config.c == 0.0 then return zero end
        
        local factors = terralib.newlist()
        local keys = orderedexpressionkeys(expmap)
        for _,k in ipairs(keys) do
            local v = expmap[k]
            if v ~= 0 then
                factors:insert(k^v)
            end
        end
        
        if #factors == 0 then return toexp(config.c) end
        if #factors == 1 and config.c == 1.0 then return factors[1] end
        return polycache(op,config.c,unpack(sortexpressions(factors)))
    end
    
    local x,y,z = unpack(args)
    
    if op.name == "pow" then
        if Const:is(y) then
            if y.v == 1.0 then
                return x
            elseif y.v == 0.0 then
                return one
            else
                local c,xx = aspowc(x)
                return ad.powc(y.v*c,xx)
            end
        end
    elseif op.name == "powc" then
        if x:type() == bool then
            return x
        end
    elseif op.name == "select" then
        if Const:is(x) then
            return  x.v ~= 0 and y or z
        elseif y == zero then
            return ad.not_(x) * z
        elseif z == zero then
            return x * y
        end
    elseif op.name == "or_" then
        if x == y then return x
        elseif Const:is(x) then
            if x.v ~= 0 then return one
            else return y end
        elseif Const:is(y) then
            if y.v  ~= 0 then return one
            else return x end
        end
    end
    
    return newapply(op,config,args)
end

local function getapply(op,...)
    local args = terralib.newlist()
    local config = {}
    for i = 1,select("#",...) do
        local e = select(i,...)
        if i <= #op.config then 
            config[op.config[i]] = assert(tonumber(e),"config must be a number")
        else
            e = assert(toexp(e),"expected an expression")
            args:insert(e)
        end
    end
    return simplify(op,config,args)
end
getapply = terralib.memoize(getapply)

function ExpVector:size() return #self.data end
function ExpVector:__tostring() return "{"..ad.tostrings(self.data).."}" end
function ExpVector:__index(key)
    if type(key) == "number" then
        assert(key >= 0 and key < #self.data, "index out of bounds")
        return self.data[key+1]
    else return ExpVector[key] end
end
ExpVector.__call = ExpVector.__index
local function toexpvectorentry(v)
    return ExpVector:is(v) and v or toexp(v)
end
function ExpVector:__newindex(key,v)
    assert(type(key) == "number", "unknown field in ExpVector: "..tostring(key))
    assert(key >= 0 and key < #self.data, "index out of bounds")
    self.data[key+1] = assert(toexpvectorentry(v), "expected a ExpVector or a valid expression")
end
function ExpVector:map(fn)
    return ad.Vector(unpack(self.data:map(fn)))
end
function ExpVector:expressions() return self.data end    
function ExpVector:sum()
    local s = 0
    for i,e in ipairs(self:expressions()) do
        s = s + e
    end
    return s
end

-- generates variable names
local v = {} 

function ad.getvar(key)
    local r = rawget(v,key)
    if not r then
        
    end
    return r
end
setmetatable(v,{__index = function(self,key)
    local type_ = float
    if type(key) == "table" and type(key.type) == "function" then
        type_ = key:type()
    end 
    assert(type_ == float or type_ == bool, "variable with key exists with a different type")
    local shape = ad.scalar
    if type(key) == "table" and type(key.shape) == "function" then
        shape = key:shape()
    end
    local r = Var:new { type_ = type_, key_ = assert(key), id = allocid(), shape_ = shape }
    v[key] = r
    return r
end})

local x,y,z = v[1],v[2],v[3]


ad.v = v
ad.toexp = toexp
ad.newclass = newclass
ad.Shape = Shape

function ad.Vector(...)
    local data = terralib.newlist()
    for i = 1,select("#",...) do
        local e = select(i,...)
        data[i] = assert(toexpvectorentry(e),"expected a ExpVector or valid expression")
    end
    return ExpVector:new { data = data }
end
ad.ExpVector = ExpVector

setmetatable(ad, { __index = function(self,idx) -- for just this file, auto-generate an op 
    local name = assert(type(idx) == "string" and idx)
    local op = Op:new { name = name }
    rawset(self,idx,op)
    return op
end })

local function conformvectors(start,...)
    local N
    local es = terralib.newlist {}
    for i = start, select('#',...) do
        local e = select(i,...)
        if ExpVector:is(e) then
            assert(not N or N == e:size(), "non-conforming vector sizes")
            N = e:size()
        end
    end
    return N
end


function Op:__call(...)
    local N = conformvectors(#self.config+1,...)
    if not N then return getapply(self,...) end
    local exps = terralib.newlist()
    for i = 0,N-1 do
        local newargs = terralib.newlist()
        for j = 1,select("#",...) do
            local e = select(j,...)
            if j <= #self.config or not ExpVector:is(e) then
                newargs:insert(e)
            else
                newargs:insert(e[i])
            end
        end
        exps:insert(self(unpack(newargs)))
    end
    return ExpVector:new { data = exps }
end

local function insertcast(from,to,exp)
    assert(terralib.types.istype(from) and terralib.types.istype(to))
    if from == to then return exp
    else return `to(exp) end
end
local function insertcasts(exp,args)
    local nargs = terralib.newlist()
    local t,ta = exp.op:propagatetype(exp:children()) 
    for i,from in ipairs(exp:children()) do
        nargs[i] = insertcast(from:type(),ta[i],args[i])
    end
    return nargs
end

Op.config = {} -- operators that have configuration override this
function Op:define(fn,...)
    local dbg = debug.getinfo(fn,"u")
    assert(not dbg.isvararg)
    self.nparams = dbg.nparams
    function self:generate(exp,args) 
        return fn(unpack(insertcasts(exp,args)))
    end
    self.derivs = terralib.newlist()
    for i = 1,select("#",...) do
        local e = select(i,...)
        self.derivs:insert((assert(toexp(e),"expected an expression")))
    end
    
    local syms,vars = terralib.newlist(),terralib.newlist()
    for i = 1,self.nparams do
        syms:insert(symbol(float))
        vars:insert(ad.v[i])
    end
    local cpropexpression = self(unpack(vars))
    local r = self:generate(cpropexpression,syms)
    terra self.impl([syms]) return float(r) end    
    
    return self
end

function Op:propagateconstant(args)
   if not self.impl then return nil end
    return assert(toexp(self.impl(unpack(args))), "result is not an expression")
end

local function rep(N,v) 
    local r = terralib.newlist()
    for i = 1,N do
        r:insert(v)
    end
    return r
end
    
function Op:propagatetype(args) --returns a 2: <returntype>, <castedargumenttypes>
    -- default is 'float', special ops will override this
    return float, rep(#args,float)
end


function Op:__tostring() return self.name end

local mo = {"add","sub","mul","div", "pow"}
for i,o in ipairs(mo) do
    local function impl(a,b) return (ad[o])(a,b) end
    Exp["__"..o], ExpVector["__"..o] = impl,impl
end
Exp.__unm = function(a) return ad.unm(a) end
ExpVector.__unm = function(a) return ad.unm(a) end

function Exp:rename(vars)
    local varsf = type(vars) == "function" and vars or function(k) return vars[k] end
    local visitcached
    local function visit(self) 
        if self.kind == "Apply" then
            local nargs = self.args:map(visitcached,vars)
            for i,c in ipairs(self.op.config) do
                nargs:insert(i,self.config[c])
            end
            return self.op(unpack(nargs))
        elseif self.kind == "Const" then
            return self
        elseif self.kind == "Var" then
            local nv = toexp(varsf(self:key()))
            return assert(nv,
                          ("rename: unknown invalid mapping for variable %s which maps to %s"):format(tostring(self:key()),tostring(nv)))
        elseif self.kind == "Reduce" then
            return getreduce(visitcached(self.v))
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

function Exp:visit(fn) -- cheaper way to find all the variable keys when you are not modifying them
    local visited = {}
    local function visit(self)
        if visited[self] then return end
        visited[self] = true
        if self.kind == "Var" then
            fn(self:key())
        end
        for i,c in ipairs(self:children()) do
            visit(c)
        end
    end
    visit(self)
end

local function countuses(es)
    local uses = {}
    local function count(e)
        uses[e] = (uses[e] or 0) + 1
        if uses[e] == 1  then
            for i,a in e:children() do count(a) end
        end
    end
    for i,a in ipairs(es) do 
        if ExpVector:is(a) then
            for i,e in ipairs(a:expressions()) do
                count(e)
            end
        else
            count(a)
        end 
    end
    for k,v in pairs(uses) do
       uses[k] = v > 1 or nil
    end
    return uses
end    

local ispoly = {sum = 0, prod = 1, powc = 2}
local function expstostring(es)
    es = (terralib.islist(es) and es) or terralib.newlist(es)
    local linearized = terralib.newlist()
    local numfree = terralib.newlist()
    local uses = {}
    local exptoidx = {}
    local function visit(e)
        if not exptoidx[e] then
            for i,c in ipairs(e:children()) do visit(c) end
            linearized:insert(e)
            exptoidx[e] = #linearized
            uses[e] = 0
        end
        uses[e] = uses[e] + 1
    end
    for i,e in ipairs(es) do 
        visit(e) 
    end
    
-------------------------------
    local exptoreg = {}
    local nextregister = 0
    local freeregisters = terralib.newlist()
    local function releaseregister(i)
        freeregisters:insert(i)
    end
    local function registerforexp(e)
        if e.kind == "Apply" or e.kind == "Const" then return -1 end -- no registers for const/var
        if exptoreg[e] then return exptoreg[e] end
        local r
        if #freeregisters > 0 then 
            r = freeregisters:remove()
        else 
            r = nextregister
            nextregister = nextregister + 1
        end
        exptoreg[e] = r
        return r
    end
---------------------------------
    
    local shouldprint = {}
    local function stringforuse(e)
        shouldprint[e] = true
        local r
        if "Var" == e.kind then
            local k = e:key()
            r = type(k) == "number" and ("v%d"):format(k) or tostring(e:key()) 
        elseif "Const" == e.kind then
            r = tostring(e.v)
        else
            r = ("r%d"):format(exptoidx[e])
        end
        if e:type() == bool then
            r = ("<%s>"):format(r)
        end
        return r
    end    

    local function emitpoly(e,l) 
        if not Apply:is(e) or not ispoly[e.op.name] or l > ispoly[e.op.name] then
            return stringforuse(e)
        end
        if e.op.name == "powc" then
            return ("%s^%s"):format(stringforuse(e.args[1]),e.config.c)
        elseif e.op.name == "prod" then
            local r = e.args:map(emitpoly,2):concat("*")
            if e.config.c ~= 1 then
                r = ("%s*%s"):format(e.config.c,r)
            end
            return r
        elseif e.op.name == "sum" then
            local r = e.args:map(emitpoly,1):concat(" + ")
            if e.config.c ~= 0 then
                r = ("%s + %s"):format(r,e.config.c)
            end
            return r
        end
    end
    
    local function emitapp(e)
        if ispoly[e.op.name] then
            return emitpoly(e,0)
        end
        local name = e.op.name
        if #e.op.config > 0 then
            local cfgstr = terralib.list.map(e.op.config,function(name) return e.config[name] end):concat(",")
            name = ("%s[%s]"):format(name,cfgstr)
        end
        return ("%s(%s)"):format(name,e.args:map(stringforuse):concat(","))
    end
    
    local estring = es:map(stringforuse):concat(",")
    
    local tbl = terralib.newlist()
    for i = #linearized,1,-1 do
        local e = linearized[i]
        if e.kind == "Apply" or e.kind == "Reduce" then
            releaseregister(registerforexp(e))
            for i,c in ipairs(e:children()) do
                registerforexp(c)
            end
            if shouldprint[e] then
                local rhs
                if e.kind == "Reduce" then
                    rhs = ("Reduce(%s)"):format(stringforuse(e.v))
                else
                    rhs = emitapp(e)
                end
                tbl:insert(("[%2d,%d]  r%d : %s %s = %s\n"):format(registerforexp(e),e.id,i,e:type(),e:shape(),rhs))
            end
        end
    end
--------------------------------
    local rtbl = terralib.newlist()
    for i = #tbl,1,-1  do
        rtbl:insert(tbl[i])
    end
    if #rtbl == 0 then return estring end
    return ("let (%d reg) \n%sin\n  %s\nend\n"):format(nextregister, rtbl:concat(),estring)
end

ad.tostrings = expstostring
function Exp:__tostring()
    return expstostring(terralib.newlist{self})
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
    terra VecType:abs()
       var c : VecType
       for i = 0,N do
	  -- TODO: use opt.abs
	  if self.data[i] < 0 then
	     c.data[i] = -self.data[i]
	  else
	     c.data[i] = self.data[i]
	  end
       end
       return c
    end
    terra VecType:sum()
       var c : typ = 0
       for i = 0,N do
	  c = c + self.data[i]
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

function Exp:d(v)
    assert(Var:is(v))
    self.derivs = self.derivs or {}
    local r = self.derivs[v]
    if r then return r end
    r = self:calcd(v)
    self.derivs[v] = r
    return r
end

function Exp:partials()
    return empty
end

function Op:getpartials(exp)
    assert(#self.derivs == #exp.args, "number of arguments do not match number of partials")
    return self.derivs:map("rename",exp.args)
end

function Apply:partials()
    self.partiallist = self.partiallist or self.op:getpartials(self)
    return self.partiallist
end

function Var:calcd(v)
   return self == v and toexp(1) or toexp(0)
end
function Const:calcd(v)
    return toexp(0)
end
function Apply:calcd(v)
    local dargsdv = self.args:map("d",v)
    local dfdargs = self:partials()
    local r
    for i = 1,#self.args do
        local e = dargsdv[i]*dfdargs[i]
        r = (not r and e) or (r + e)
    end
    return r
end

function Reduce:calcd(v) return self.args[1]:d(v) end
local reducepartials = terralib.newlist { one }
function Reduce:partials() return reducepartials end
--calc d(thisexpress)/d(exps[1]) ... d(thisexpress)/d(exps[#exps]) (i.e. the gradient of this expression with relation to the inputs) 
function Exp:gradient(exps)
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
        local p = e:partials()
        for j,c in ipairs(e:children()) do
            tape[c] = tape[c] + tape[e]*p[j]
        end
    end
    return exps:map(function(e) return tape[e] or zero end)
end

function ad.sum:generate(exp,args)
    args = insertcasts(exp,args)
    local r = exp.config.c
    for i,c in ipairs(args) do
        r = `r+c
    end
    return r
end
function ad.sum:getpartials(exp) return rep(#exp.args,one) end
ad.sum.config = {"c"}

function ad.add(x,y) return ad.sum(0,x,y) end
function ad.sub(x,y) return ad.sum(0,x,-y) end
function ad.mul(x,y) return ad.prod(1,x,y) end
function ad.div(x,y) return ad.prod(1,x,y^-1) end

function ad.prod:generate(exp,args)
    local r = exp.config.c
    local condition = true
    for i,ce in ipairs(exp:children()) do
        local a = args[i]
        if ce:type() == bool then
            condition = `condition and a
        else
            r = `r*a
        end
    end
    if condition == true then
        return r
    else
        return `terralib.select(condition,r,0.f)
    end
end
function ad.prod:getpartials(exp)
    local r = terralib.newlist()
    for i,a in ipairs(exp.args) do
        local terms = terralib.newlist()
        for j,a2 in ipairs(exp.args) do
            if i ~= j then
                terms:insert(a2)
            end
        end
        r:insert(ad.prod(exp.config.c,unpack(terms)))
    end
    return r
end
ad.prod.config = { "c" }

local genpow = terralib.memoize(function(N)
    local terra pow(a : float) : float
        var r : float = 1.f
        for i = 0,N do
            r = r*a
        end
        return r
    end 
    pow:setname("pow"..tostring(N))
    return pow
end)
function ad.powc:generate(exp,args)
    args = insertcasts(exp,args)
    local c,e = exp.config.c, args[1]
    if c == 1 then
        return e
    elseif c > 0 then
        return `[genpow(c)](e)
    else
        return `1.f/[genpow(-c)](e)
    end
end

function ad.powc:getpartials(exp)
    local x = exp.args[1]
    local c = exp.config.c
    return terralib.newlist { c*ad.pow(x,c-1) }
end
ad.powc.config = { "c" }

function ad.unm(x) return ad.prod(-1.0,x) end

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
ad.sinh:define(function(x) return `C.sinh(x) end, ad.cosh(x))
ad.sqrt:define(function(x) return `C.sqrt(x) end, 1.0/(2.0*ad.sqrt(x)))
ad.tan:define(function(x) return `C.tan(x) end, 1.0 + ad.tan(x)*ad.tan(x))
ad.tanh:define(function(x) return `C.tanh(x) end, 1.0/(ad.cosh(x)*ad.cosh(x)))



function ad.select:propagatetype(args) return float, {bool,float,float} end
ad.select:define(function(x,y,z) 
    return quote
        var r : float
        if x then
            r = y
        else    
            r = z
        end
    in r end
end,0,ad.select(x,1,0),ad.select(x,0,1))

ad.abs:define(function(x) return `terralib.select(x >= 0,x,-x) end, ad.select(ad.greatereq(x, 0),1,-1))

function ad.and_(x,y) return x*y end

function ad.or_:propagatetype(args) return bool,{bool,bool} end
ad.or_:define(function(x,y) return `x or y end, 0, 0)

local comparisons = { "less", "greater", "lesseq", "greatereq", "eq" }
for i,c in ipairs(comparisons) do
    ad[c].propagatetype = function(self,args) return bool, {float,float} end
end 

ad.eq:define(function(x,y) return `x == y end, 0,0)
ad.less:define(function(x,y) return `x < y end, 0,0)
ad.greater:define(function(x,y) return `x > y end, 0,0)
ad.lesseq:define(function(x,y) return `x <= y end,0,0)
ad.greatereq:define(function(x,y) return `x >= y end,0,0)

function ad.not_:propagatetype(args) return bool, {bool} end
ad.not_:define(function(x) return `not x end, 0)
ad.materialize:define(function(x) return x end,1) -- preserved across math optimizations


setmetatable(ad,nil) -- remove special metatable that generates new blank ops

ad.Var,ad.Apply,ad.Const,ad.Exp,ad.Reduce = Var, Apply, Const, Exp, Reduce
function ad.reduce(x)
    if ExpVector:is(x) then
        return x:map(getreduce)
    end
    return getreduce(x)
end
function ad.polysimplify(exps)
    local function sumtoterms(sum)
        assert(Apply:is(sum) and sum.op.name == "sum")
        local terms = terralib.newlist()
        -- build internal list of terms
        for i,f in ipairs(sum:children()) do
            local c,ff = asprod(ad.polysimplify(f))
            local factor = {}
            for j,p in ipairs(ff) do
                local c,pp = aspowc(p)
                factor[pp] = c
            end
            if c ~= 1 then
                factor[toexp(c)] = 1
            end
            
            terms:insert(factor)
        end
        return terms,sum.config.c
    end

    local function createsum(terms,c)
        local factors = terralib.newlist()
        for i,t in ipairs(terms) do
            local pows = terralib.newlist()
            local keys = orderedexpressionkeys(t)
            for _,k in ipairs(keys) do
                local v = t[k]
                pows:insert(k^v)
            end
            factors:insert(ad.prod(1,unpack(pows)))
        end
        local r = ad.sum(assert(tonumber(c),"NaN?"),unpack(factors))
        return r
    end

    local function simplifylist(terms,c)

        -- short circuit when there is going to be no advantage to keep going
        if #terms == 0 then return toexp(c)
        elseif #terms == 1 then
            return createsum(terms,c)
        end

        local minpower,maxnegpower = {},{}
        local uses,neguses = {},{}

        -- count total uses
        for i,t in ipairs(terms) do
            for k,v in pairs(t) do
                local v = t[k]
                if v > 0 then
                    uses[k] = (uses[k] or 0) + 1
                    minpower[k] = math.min(minpower[k] or math.huge, v)
                else
                    neguses[k] = (neguses[k] or 0) + 1
                    maxnegpower[k] = math.max(maxnegpower[k] or -math.huge, v)
                end
            end
        end
        -- find maximum uses
        local maxuse,power,maxkey = 0
        
        local keys = orderedexpressionkeys(uses)
        for _,k in ipairs(keys) do
            local u = uses[k]
            if u > maxuse then
                maxuse,maxkey,power = u,k,minpower[k]
            end
        end
        local keys = orderedexpressionkeys(neguses)
        for _,k in ipairs(keys) do
            local u = neguses[k]
            if u > maxuse then
                maxuse,maxkey,power = u,k,maxnegpower[k]
            end
        end
        if maxuse < 2 then
            return createsum(terms,c) -- no benefit, so stop here
        end
        --print("FACTORING",maxuse,power,maxkey)
        --partition terms
        local used,notused = terralib.newlist(),terralib.newlist()
        for i,t in ipairs(terms) do
            local v = t[maxkey]
            if v and ((v > 0 and power > 0) or (v < 0 and power < 0)) then
                local newv = v - power
                if newv == 0 then newv = nil end
                t[maxkey] = newv
                used:insert(t) 
            else
                notused:insert(t)
            end
        end

        -- simplify both sides, and make a new term

        --print("recurse",#notused,#used)
        local lhs = simplifylist(notused,0)
        local rhs = simplifylist(used,0)
        local r = ad.sum(c,lhs,maxkey^power * rhs)
        return r
    end
    
    local function dosimplify(exp)
        if Apply:is(exp) then
            if exp.op.name == "sum" then
                return simplifylist(sumtoterms(exp))
            else
                local nargs = exp:children():map(ad.polysimplify)
                if exp.config.c then -- HACK! only works for fake config we have now
                    return exp.op(exp.config.c,unpack(nargs))
                end
                local r = exp.op(unpack(nargs))
                return r
            end
        else 
            return exp
        end
    end
    return terralib.islist(exps) and exps:map(dosimplify) or dosimplify(exps)
end

--[[


local exp = v.param_w_regSqrt * (v.X_0_0_0 + -1*v.X_1_0_0 - ( ad.cos(v.X_0_0_2) * (v.UrShape_0_0_0 + -1*v.UrShape_1_0_0) -ad.sin(v.X_0_0_2) * (v.UrShape_0_0_1 - v.UrShape_1_0_1)))
print(exp)
print(exp:d(v.X_0_0_2))



local x,y,z = ad.v.x, ad.v.y, ad.v.z
local bx,by,bz = ad.getvar(bool,"bx"),ad.getvar(bool,"by"),ad.getvar(bool,"bz")
print(ad.or_(ad.eq(x,0),ad.eq(y,0))/4.0)
local p = 4*x^2*y*y + 4*x*y*y + 4*y*y  + -1*x^-1 - 2*x^-2*y*y

print(p)
print(ad.polysimplify(p))
local f = -3*ad.pow(x,2)
print(f:d(x))

print(expstostring(ad.atan2.derivs))
assert(y == y)
assert(ad.cos == ad.cos)
local r = ad.sin(x) + ad.cos(y) + ad.cos(y)
print(r)
print(expstostring( terralib.newlist { r, r + 1} ))
print(x*-(-y+1)/4)
print(r:rename({x = x+y,y = x+y}))
local e = 2*x*x*x*3 -- - y*x
print((ad.sin(x)*ad.sin(x)):d(x))
print(e:d(x)*3+(4*x)*x)

local w = x*x

local exp = (3*w + 4*w + 3*w*z)

--6x+8x+6xz

-- 7x + (7x + (z*7x

-- 7x + (r1 
print(unpack((3*x*x + x):gradient{x}))
local g = exp:gradient({x,y,z})
print(expstostring(g))
local t = ad.toterra(g, function(stmts,v) return assert(({x = symbol("x"), y = symbol("y"), z = symbol("z")})[v]) end)
t:printpretty()

print(ad.select(ad.v.x,ad.select(ad.v.y,ad.v.z,0),0))

print(3*ad.v.x*ad.v.y)
print(3*ad.v.y*ad.v.x^3/ad.v.x^2)

print(3*ad.v.x+4*ad.v.x*ad.v.y-3*ad.v.y*ad.v.x-ad.v.y*ad.v.x)


local foo = 3*x^50 - x*2 - 1

print(foo)
print(foo:d(x))
local VC = newclass("VC")
function VC:__tostring() return "VC" end
function VC:shape() return Shape:fromkeys{1,2} end 

local vx = ad.v[ VC:new {} ]
local r = (x + getreduce(y + vx))
print(r)
local rr = r:rename(function(x) if x == "y" then return ad.v.x else return ad.v[x] end end)
print(rr)
]]
local x,y,z = ad.v.x, ad.v.y, ad.v.z


return ad
