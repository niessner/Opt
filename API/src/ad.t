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

function Const:__tostring() return tostring(self.v) end

local function newapply(op,config,args)
    assert(not op.nparams or #args == op.nparams)
    assert(type(config) == "table")
    local id = allocid()
    return Apply:new { op = op, args = args, config = config, id = id }
end

local getconst = terralib.memoize(function(n) return Const:new { v = n, id = allocid() } end)
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
        for k,v in pairs(expmap) do
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
        if Apply:is(x) and x.op.name == "bool" then
            return x
        end
    elseif op.name == "select" then
        if Const:is(x) then
            return  x.v ~= 0 and y or z
        elseif y == zero then
            return x * z
        elseif z == zero then
            return x * y
        end
    elseif (op.name == "and_" or op.name == "or_") and x == y then
        return x
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

-- generates variable names
local v = setmetatable({},{__index = function(self,idx)
    local r = Var:new { key_ = assert(idx), id = allocid() }
    self[idx] = r
    return r
end})

local x,y,z = v[1],v[2],v[3]


ad.v = v
ad.toexp = toexp
ad.newclass = newclass

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
Op.config = {} -- operators that have configuration override this
function Op:define(fn,...)
    local dbg = debug.getinfo(fn,"u")
    assert(not dbg.isvararg)
    self.nparams = dbg.nparams
    function self:generate(exp,emit) return fn(unpack(exp:children():map(emit))) end
    local s = terralib.newlist()
    for i = 1,self.nparams do
        s:insert(symbol(float))
    end
    terra self.impl([s]) return float([ fn(unpack(s)) ]) end    
    self.derivs = terralib.newlist()
    for i = 1,select("#",...) do
        local e = select(i,...)
        self.derivs:insert((assert(toexp(e),"expected an expression")))
    end
    return self
end

function Op:propagateconstant(args)
   if not self.impl then return nil end
    return assert(toexp(self.impl(unpack(args))), "result is not an expression")
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

local ispoly = {sum = 0, prod = 1, powc = 2, bool = 3}
--local ispoly = {}
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
        if e.kind ~= "Apply" then return -1 end -- no registers for const/var
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
        if "Var" == e.kind then
            local k = e:key()
            return type(k) == "number" and ("v%d"):format(k) or tostring(e:key()) 
        elseif "Const" == e.kind then
            return tostring(e.v)
        else
            return ("r%d"):format(exptoidx[e])
        end
    end    

    local function emitpoly(e,l) 
        if not Apply:is(e) or not ispoly[e.op.name] or l > ispoly[e.op.name] then
            return stringforuse(e)
        end
        if e.op.name == "powc" then
            return ("%s^%d"):format(stringforuse(e.args[1]),e.config.c)
        elseif e.op.name == "prod" then
            local r = e.args:map(emitpoly,2):concat("*")
            if e.config.c ~= 1 then
                r = ("%d*%s"):format(e.config.c,r)
            end
            return r
        elseif e.op.name == "sum" then
            local r = e.args:map(emitpoly,1):concat(" + ")
            if e.config.c ~= 0 then
                r = ("%s + %d"):format(r,e.config.c)
            end
            return r
        elseif e.op.name == "bool" then
            return ("<%s>"):format(stringforuse(e.args[1]))
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
        if e.kind == "Apply" then
            releaseregister(registerforexp(e))
            for i,c in ipairs(e:children()) do
                registerforexp(c)
            end
            if shouldprint[e] then
                tbl:insert(("[%2d,%d]  r%d = %s\n"):format(registerforexp(e),e.id,i,emitapp(e)))
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

local function rep(N,v) 
    local r = terralib.newlist()
    for i = 1,N do
        r:insert(v)
    end
    return r
end


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

local function emitpow(emit,c,e)
    assert(c ~= 0)
    e = emit(e)
    if c == 1 then
        return e
    elseif c > 0 then
        return `[genpow(c)](e)
    else
        return `1.f/[genpow(-c)](e)
    end 
end
local function emitfactor(emit,c,es)
    local exp
    if c ~= 1 then
        exp = `c
    end
    for i,e in ipairs(es) do
        local c,ee = aspowc(e)
        if not exp then
            exp = emitpow(emit,c,ee)
        elseif c > 0 then
            local d = emitpow(emit,c,ee)
            exp = `exp * d
        else 
            local d = emitpow(emit,-c,ee)
            exp = `exp / d
        end
    end
    return exp
end
local function emitsum(emit,c,es)
    local exp
    if c ~= 0 then
        exp = `c
    end
    for i,e in ipairs(es) do
        local c,ee = asprod(e)
        if not exp then
            exp = emitfactor(emit,c,ee)
        elseif c > 0 then
            local d = emitfactor(emit,c,ee)
            exp = `exp + d
        else 
            local d = emitfactor(emit,-c,ee)
            exp = `exp - d
        end
    end
    return exp
end

function ad.sum:generate(exp,emit)
    return emitsum(emit,exp.config.c,exp:children())
end
function ad.sum:getpartials(exp) return rep(#exp.args,one) end
ad.sum.config = {"c"}

function ad.add(x,y) return ad.sum(0,x,y) end
function ad.sub(x,y) return ad.sum(0,x,-y) end
function ad.mul(x,y) return ad.prod(1,x,y) end
function ad.div(x,y) return ad.prod(1,x,y^-1) end

function ad.prod:generate(exp,emit)
    return emitfactor(emit,exp.config.c,exp:children())
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
function ad.powc:generate(exp,emit)
    return emitpow(emit,exp.config.c,exp:children()[1])
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

ad.bool:define(function(x) return `terralib.select(bool(x),1.f,0.f) end, x)
--ad.select:define(function(x,y,z) return `terralib.select(bool(x),y,z) end,0,ad.select(x,1,0),ad.select(x,0,1))
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

ad.eq_:define(function(x,y) return `x == y end, 0,0)
function ad.eq(x,y) return ad.bool(ad.eq_(x,y)) end
ad.abs:define(function(x) return `terralib.select(x >= 0,x,-x) end, ad.select(ad.greatereq(x, 0),1,-1))

ad.and6:define(function(x0, x1, x2, x3, x4, x5) return `int(x0) and int(x1) and int(x2) and int(x3) and int(x4) and int(x5) end,0,0,0,0,0,0)
ad.and_:define(function(x,y) return `int(x) and int(y) end, 0, 0)
ad.or_:define(function(x,y) return `int(x) or int(y) end, 0, 0)
ad.less:define(function(x,y) return `int(x < y) end, 0,0)
ad.greater:define(function(x,y) return `int(x > y) end, 0,0)
ad.lesseq:define(function(x,y) return `int(x <= y) end,0,0)
ad.greatereq_:define(function(x,y) return `int(x >= y) end,0,0)
function ad.greatereq(x,y) return ad.bool(ad.greatereq_(x,y)) end

ad.not_:define(function(x) return `int(not bool(x)) end, 0)

setmetatable(ad,nil) -- remove special metatable that generates new blank ops

ad.Var,ad.Apply,ad.Const,ad.Exp = Var, Apply, Const, Exp

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
            for k,v in pairs(t) do
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
        for k,u in pairs(uses) do
            if u > maxuse then
                maxuse,maxkey,power = u,k,minpower[k]
            end
        end
        for k,u in pairs(neguses) do
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
        for i,t in pairs(terms) do
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

function ad.removepoly(exp)
    
end


--[[


local exp = v.param_w_regSqrt * (v.X_0_0_0 + -1*v.X_1_0_0 - ( ad.cos(v.X_0_0_2) * (v.UrShape_0_0_0 + -1*v.UrShape_1_0_0) -ad.sin(v.X_0_0_2) * (v.UrShape_0_0_1 - v.UrShape_1_0_1)))
print(exp)
print(exp:d(v.X_0_0_2))



local x,y,z = ad.v.x, ad.v.y, ad.v.z

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
]]
return ad