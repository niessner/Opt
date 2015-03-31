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

function Var:N() return self.p end
function Const:N() return 0 end
function Apply:N() return self.nvars end


function Var:cost() return 1 end
function Const:cost() return 1 end
function Apply:cost() 
    local c = 1
    for i,a in ipairs(self.args) do
        c = c + a:cost()
    end
    return c
end

function Const:__tostring() return tostring(self.v) end


local function newapply(op,args)
    local nvars = 0
    assert(not op.nparams or #args == op.nparams)
    for i,a in ipairs(args) do
        nvars = math.max(nvars,a:N())
    end
    return Apply:new { op = op, args = args, nvars = nvars }
end

local getconst = terralib.memoize(function(n) return Const:new { v = n } end)
local function toexp(n)
    return Exp:is(n) and n or tonumber(n) and getconst(n)
end

local zero,one = toexp(0),toexp(1)
local function allconst(args)
    for i,a in ipairs(args) do
        if not Const:is(a) then return false end
    end
    return true
end
local function factor(e)
    if e.kind == "Apply" and e.op.name == "mul" and Const:is(e.args[1]) then
        return e.args[1].v, e.args[2]
    end
    return 1,e
end
local function simplify(op,args)
    local x,y = unpack(args)
    if allconst(args) and op:getimpl() then
        return toexp(op:getimpl()(unpack(args:map("v"))))
    elseif op.name == "add" then
        if Const:is(x) and not Const:is(y) then
            return y + x
        elseif y == zero then return x end
        local c,a = factor(x)
        local k,b = factor(y)
        if a == b then return (c + k) * a end
    elseif op.name == "mul" then
        if Const:is(y) and not Const:is(x) then
            return y*x
        elseif x == one then return y
        elseif x == zero then return zero
        elseif Const:is(x) then
            local c,a = factor(y)
            if c ~= 1 then
                return (c*x.v)*y
            end
        end
    elseif op.name == "div" then
        if y == one then return x end
    elseif op.name == "sub" then
        if y == zero then return x end
        elseif x == zero then return -y
    end
    return newapply(op,args)
end


local getapply = terralib.memoize(function(op,...)
    local args = terralib.newlist {...}
    return simplify(op,args)
end)


local function toexps(...)
    local es = terralib.newlist {...}
    for i,e in ipairs(es) do
        es[i] = assert(toexp(e))
    end
    return es
end
    

-- generates variable names
local v = setmetatable({},{__index = function(self,idx)
    local r = Var:new { p = assert(tonumber(idx)) }
    self[idx] = r
    return r
end})

local x,y,z = v[1],v[2],v[3]

local ad = {}
setmetatable(ad, { __index = function(self,idx)
    local name = assert(type(idx) == "string" and idx)
    local op = Op:new { name = name }
    rawset(self,idx,op)
    return op
end })

function Op:__call(...)
    local args = toexps(...)
    return getapply(self,unpack(args))
end
function Op:define(fn,...)
    self.nparams = debug.getinfo(fn,"u").nparams
    self.generator = fn
    self.derivs = toexps(...)
    for i,d in ipairs(self.derivs) do
        assert(d:N() <= self.nparams)
    end
    return self
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

function Op:__tostring() return self.name end

local mo = {"add","sub","mul","div"}
for i,o in ipairs(mo) do
    Exp["__"..o] = function(a,b) return (ad[o])(a,b) end
end
Exp.__unm = function(a) return ad.unm(a) end

function Var:rename(vars)
    assert(self.p <= #vars)
    return vars[self.p]
end
function Const:rename(vars) return self end
function Apply:rename(vars)
    assert(self:N() <= #vars) 
    return self.op(unpack(self.args:map("rename",vars)))
end

local function countuses(es)
    local uses = {}
    local function count(e)
        uses[e] = (uses[e] or 0) + 1
        if uses[e] == 1 and e.kind == "Apply" then
            for i,a in ipairs(e.args) do count(a) end
        end
    end
    for i,a in ipairs(es) do count(a) end
    for k,v in pairs(uses) do
       uses[k] = v > 1 or nil
    end
    return uses
end    


local infix = { add = {"+",1}, sub = {"-",1}, mul = {"*",2}, div = {"/",2} }


local function expstostring(es)
    local n = 0
    local tbl = terralib.newlist()
    local manyuses = countuses(es)
    local emitted = {}
    local function prec(e)
        if e.kind ~= "Apply" or e.op.name == "unm" or not infix[e.op.name] or manyuses[e] then return 3
        else return infix[e.op.name][2] end
    end
    local emit
    local function emitprec(e,p)
        return (prec(e) < p and "(%s)" or "%s"):format(emit(e))
    end
    local function emitapp(e)
        if e.op.name == "unm" then
            return ("-%s"):format(emitprec(e.args[1],3))
        elseif infix[e.op.name] then
            local o,p = unpack(infix[e.op.name])
            return ("%s %s %s"):format(emitprec(e.args[1],p),o,emitprec(e.args[2],p))
        else
            return ("%s(%s)"):format(tostring(e.op),e.args:map(emit):concat(","))
        end
    end
    function emit(e)
        if "Var" == e.kind then
            return ("v%d"):format(e.p)
        elseif "Const" == e.kind then
            return tostring(e.v)
        elseif "Apply" == e.kind then
            if emitted[e] then return emitted[e] end
            local exp = emitapp(e)
            if manyuses[e] then
                local r = ("r%d"):format(n)
                n = n + 1
                emitted[e] = r
                tbl:insert(("  %s = %s\n"):format(r,exp))            
                return r
            else
                return exp
            end
            
        end
    end
    local r = es:map(emit)
    r = r:concat(",")
    if #tbl == 0 then return r end
    return ("let\n%sin\n  %s\nend\n"):format(tbl:concat(),r)
end

function Exp:__tostring()
    return expstostring(terralib.newlist{self})
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
    return toexp(0)
end
function Apply:calcd(v)
    local dargsdv = self.args:map("d",v)
    local dfdargs = self.op.derivs:map("rename",self.args)
    local r
    for i = 1,#self.args do
        local e = dargsdv[i]*dfdargs[i]
        r = (not r and e) or (r + e)
    end
    return r
end


ad.add:define(function(x,y) return `x + y end,1,1)
ad.sub:define(function(x,y) return `x - y end,1,-1)
ad.mul:define(function(x,y) return `x * y end,y,x)
ad.div:define(function(x,y) return `x / y end,1/y,-x/(y*y))
ad.unm:define(function(x) return `-x end, 1)
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
ad.tanh:define(function(x) return `C.tanh(a) end, 1.0/(ad.cosh(x)*ad.cosh(x)))



--[[
print(expstostring(ad.atan2.derivs))



assert(y == y)
assert(ad.cos == ad.cos)
local r = ad.sin(x) + ad.cos(y) + ad.cos(y)
print(r)
print(expstostring( terralib.newlist { r, r + 1} ))

print(x*-(-y+1)/4)

print(r:rename({x+y,x+y}))]]


local e = 2*x*x*x - y*x
print(e:d(x))
