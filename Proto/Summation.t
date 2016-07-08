--[[

A = ...
x = ...
b = ...


o:set(<expr>)

i,j,k, ii,jj,kk = 

(i)((+j)(A(i,j)*x(j)) + b(i))


J = ...
F = ...

J = jacobean(F)

J' =J(X)

local r = (i)((+j)(2*J(j,i)*F(j)))

for i = 1, 10

    R:set(r)

JTJp

(i)((+j)(J(j,i)*(+k)(J(j,k)*p(k))))

(i, +j)( (J(j,i)*(i,+j)J(i,j)*p(j)(j))

(i) (+j) Jt(i,j)* (+k)(J(j,k)*p(k)

(mapexpr) (scalarexpr)

(i,j)

(i .. j)(sum(j)(sum(k) (sum(j)(<body>) .. sum(j) .. sum(k))

(i .. sum(j))(expr) --> (i)(sum(j)(expr))

]]

asdl = require('asdl')
local List = asdl.List
C = asdl:NewContext()
C:Define [[
    
    Tensor = (string name, number* dims, number allocatedsize, any data)
    
    Index = (string name)
    
    MapExp = (Index* indices, ScalarExp exp)
    
    Op = (string name)
    
    ScalarExp = Apply(Op op,ScalarExp* expressions)
              | Reduce(Op op, Index index, ScalarExp exp)
              | Constant(number c)
              | Load(Tensor tensor, Index* indices)


    IndexList = (any* indices)
    ReduceObj = (Op op, Index index)
]]

local add,mul = C.Op("+"), C.Op("*")
local i,j,k = C.Index("i"),C.Index("j"),C.Index("k")


local function indexconcat(...)
    local r = List{}
    for _,o in ipairs {...} do
        if C.IndexList:isclassof(o) then
            r:insertall(o.indices)
        else
            r:insert(o)
        end
    end
    local il = C.IndexList(r)
    return il
end

C.Index.__concat = indexconcat 
C.IndexList.__concat = indexconcat
C.ReduceObj.__concat = indexconcat

function C.IndexList:__call(exp)
    for i,idx in ipairs(self.indices) do
        if C.ReduceObj:isclassof(idx) then
            print(i,idx)
            for j = #self.indices,i,-1 do
                local ridx = self.indices[j]
                assert(C.ReduceObj:isclassof(ridx),"reduce indices must follow map indices")
                exp = ridx(exp)
            end 
            print(exp)
            return C.MapExp(List { unpack(self.indices,1,i-1) },exp)
        end
    end
    return C.MapExp(self.indices,exp)
end
function C.Index:__call(exp)
    return C.MapExp(List{self},exp)
end

local function sum(i)
    return C.ReduceObj(add,i)
end

function C.ReduceObj:__call(exp)
    return C.Reduce(self.op,self.index,exp)
end

function C.Tensor:__call(...)
    local indices = List {...}
    assert(#indices == #self.dims,"wrong dim")
    return C.Load(self, indices)
end
function C.Tensor:get(...)
    assert(#self.dims > 0, "empty tensor")
    local values = {...}
    local exp = 0
    for i = 1,#values do
        assert(values[i] < self.dims[i],"index of of bounds")
        exp = exp*self.dims[i] + values[i]
    end
    return self.data[exp]
end

function requiredsize(dims)
    local size = 1
    for _,d in ipairs(dims) do
        size = size * d
    end
    return size
end
    
function Iota(name,...)
    local dims = List {...}
    local size = requiredsize(dims)
    local data = terralib.new(float[size])
    for i = 0,size-1 do
        data[i] = i
    end
    return C.Tensor(name,dims,size,data)
end

local function toscalar(x)
    if C.ScalarExp:isclassof(x) then
        return x
    end
    if type(x) == "number" then
        return C.Constant(x)
    else error("not a scalar expression") end
end

function C.ScalarExp.__add(a,b)
   return C.Apply(add, List {toscalar(a),toscalar(b)})
end
function C.ScalarExp.__mul(a,b)
    return C.Apply(mul, List {toscalar(a),toscalar(b)})
end

function C.Op:__tostring() return self.name end

C.ScalarExp.__tostring = nil
function C.Load:__tostring() return ("%s(%s)"):format(tostring(self.tensor),self.indices:map(tostring):concat(",")) end
function C.Constant:__tostring() return tostring(self.c) end
function C.Reduce:__tostring() return ("(%s%s)(%s)"):format(tostring(self.op),tostring(self.index),tostring(self.exp)) end
function C.Apply:__tostring()
    local left = tostring(self.expressions[1])
    local right = tostring(self.expressions[2])
    return ("(%s %s %s)"):format(left,tostring(self.op),right)
end
function C.Index:__tostring() return self.name end
function C.Tensor:__tostring() return self.name end
function C.MapExp:__tostring() return ("(%s)(%s)"):format(self.indices:map(tostring):concat(","),tostring(self.exp)) end

function listeq(a,b)
    if #a ~= #b then return false end
    for i = 1,#a do
        if a[i] ~= b[i] then return false end
    end
    return true
end

function Scalar(name,value)
    local data = terralib.new(float[1], { value or 0 } )
    return C.Tensor(name,List{},1,data)
end

function C.Tensor:set(exp)
    assert(C.MapExp:isclassof(exp),"expected a map expression")
    local dims,terrafn = exp:compile()
    local newsize = requiredsize(dims)
    if self.allocatedsize < newsize then
        self.data = terralib.new(float[newsize])
        self.allocatedsize = newsize
    end
    terrafn(self.data)
    self.dims = dims
end

function C.MapExp:typecheck()
    print(self)
    local indextosize = {} -- map from abstract index object to concrete dimension that the index maps over
    local definedindices = {}
    for _,d in ipairs(self.indices) do
        definedindices[d] = true
    end
    local function check(exp)
        if "Constant" == exp.kind then
        elseif "Apply" == exp.kind then
            exp.expressions:map(check)
        elseif "Load" == exp.kind then
            for dimidx,i in ipairs(exp.indices) do
                assert(definedindices[i],"free index found")
                indextosize[i] = indextosize[i] or exp.tensor.dims[dimidx]
                assert(indextosize[i] == exp.tensor.dims[dimidx],"indices do not match dimension")
            end
        elseif "Reduce" == exp.kind then
            assert(not definedindices[exp.index], "index already in use")
            definedindices[exp.index] = true
            check(exp.exp)
            assert(indextosize[exp.index],"size of reduction index not defined")
        else error("unknown kind") end
    end 
    
    check(self.exp)
    for _,d in ipairs(self.indices) do
        assert(indextosize[d], "size of map index not defined")
    end
    for k,v in pairs(indextosize) do print(k,v) end
    return indextosize
end

function add:emit(x,y) return `x + y end
add.identity = `0
function mul:emit(x,y) return `x * y end
mul.identity = `1
local function indicestoaddr(dims, indices)
    local addr = nil
    for i = 1,#dims do
        addr = addr and (`addr*[dims[i]] + [indices[i]]) or indices[i]
    end
    return addr
end

function C.MapExp:compile()
    local indextosize = self:typecheck()
    local indextosym = {}
    local syms = self.indices:map(function(index)
        local sym = symbol(int,index.name)
        indextosym[index] = sym
        return sym
    end)
    local sizes = self.indices:map(function(i) return indextosize[i] end)
    
    local function emit(exp)
        if "Constant" == exp.kind then return `exp.c
        elseif "Apply" == exp.kind then
            local args = exp.expressions:map(emit)
            return exp.op:emit(unpack(args))
        elseif "Load" == exp.kind then
            local isyms = exp.indices:map(function(x) return indextosym[x] end)
            local sizes = exp.indices:map(function(x) return indextosize[x] end)
            local addr = indicestoaddr(sizes,isyms)
            return `exp.tensor.data[addr]
        elseif "Reduce" == exp.kind then
            local size = indextosize[exp.index]
            local i = symbol(int,exp.index.name)
            indextosym[exp.index] = i
            local e = emit(exp.exp)
            return quote
                var s = exp.op.identity
                for [i] = 0,size do
                    s = [ exp.op:emit(s, e) ]
                end
            in s end
        else error("unknown kind") end
    end
    local value = emit(self.exp)
    local r = symbol(&float,"r")
    local body = quote
        r[ [indicestoaddr(sizes,syms)] ] = value 
    end
    for i = #syms,1,-1 do
        body = quote 
            for [syms[i]] = 0,[ sizes[i] ] do
                [ body ]
            end
        end
    end
    local terra impl([r])
        [body]
    end
    impl:printpretty(false)
    return sizes,impl
end
function C.Tensor:dump()
    local idx = 0
    local body = List {}
    local function emitdim(i)
        if i > #self.dims then 
            body:insert(tostring(self.data[idx]))
            idx = idx + 1
        else
            local sep = (i == #self.dims) and " " or ("\n"):rep(#self.dims - i)
            for j = 1,self.dims[i] do
                emitdim(i+1)
                body:insert(sep)
            end
        end
    end
    emitdim(1)
    body:insert("\n")
    return body:concat()
end

local A = Iota("A",2,3)
local x = Iota("x",3)
local c = Iota("b",2)

print(A:dump())

local a = (i..j)(A(i,j) + A(i,j))
local b = (i..sum(j))(A(i,j)*x(j) + c(i))

local r = Scalar("r",0)
r:set(b)

local B = Iota("B",3,4)



print(A:dump())
print(x:dump())
print(c:dump())
print(r:dump())

print("----------")

local mm = (i..j)(sum(k)(A(i,k)*B(k,j))+1)

r:set(mm)

print(B:dump())
print(r:dump())