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


    IndexList = (Index* indices)
    ReduceObj = (Op op, Index index)
]]

local add,mul = C.Op("+"), C.Op("*")
local i,j,k = C.Index("i"),C.Index("j"),C.Index("k")

function C.Index:__concat(rhs)
    return C.IndexList(List {self,rhs})
end
function C.IndexList:__concat(rhs)
    local newlist = List()
    newlist:insertall(self.indices)
    newlist:insert(rhs)
    return C.IndexList(newlist)
end

function C.IndexList:__call(exp)
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
    local exp = values[1]
    assert(exp < self.dims[1],"index out of bounds")
    for i = 2,#values do
        assert(values[i] < self.dims[i],"index of of bounds")
        exp = exp*self.dims[i]+values[i]
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

function Empty(name)
    return C.Tensor(name,List{},0,nil)
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
    local indextosize = {}
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

function C.MapExp:compile()
    local indextosize = self:typecheck()

end

local A = Iota("A",2,3)
local x = Iota("x",3)
local b = Iota("b",2)

local a = (i..j)(A(i,j))
local b = (i)(sum(j)(A(i,j)*x(j))+b(i))

a:typecheck()
b:typecheck()
