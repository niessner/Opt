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
    
    Tensor = (string name, number* dims, number* strides, number allocatedsize, any data)
    
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
    local values = {...}
    local offset = 0
    local sizes = List{}
    local strides = List{}
    for i = 1,#values do
        if values[i] == ":" then
            sizes:insert(self.dims[i])
            strides:insert(self.strides[i])
        else
            offset = offset + self.strides[i]*values[i]
        end
    end
    if #sizes == 0 then return self.data[offset]
    else return C.Tensor("slice", sizes, strides, 0, self.data + offset) end
end

function requiredsize(dims)
    local size = 1
    for _,d in ipairs(dims) do
        size = size * d
    end
    return size
end

local function densestride(dims)
    local strides = List{}
    local s = 1
    for i = #dims,1,-1 do
        strides[i] = s
        s = s * dims[i]
    end
    return strides
end

function Iota(name,...)
    local dims = List {...}
    local strides = densestride(dims)
    local size = requiredsize(dims)
    local data = terralib.new(float[size])
    for i = 0,size-1 do
        data[i] = i
    end
    return C.Tensor(name,dims,strides,size,data)
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
    return C.Tensor(name,List{},List{},1,data)
end

function C.Tensor:set(exp)
    assert(C.MapExp:isclassof(exp),"expected a map expression")
    local dims,strides,terrafn = exp:compile()
    local newsize = requiredsize(dims)
    if self.allocatedsize < newsize then
        self.data = terralib.new(float[newsize])
        self.allocatedsize = newsize
    end
    terrafn(self.data)
    self.dims,self.strides = dims,strides
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
local function indicestoaddr(strides, indices)
    local addr = nil
    for i = 1,#strides do
        local ii = strides[i] == 1 and indices[i] or `[strides[i]]*[indices[i]]
        addr = addr and (`addr + ii) or ii
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
            local addr = indicestoaddr(exp.tensor.strides,isyms)
            return `exp.tensor.data[addr]
        elseif "Reduce" == exp.kind then
            local size = indextosize[exp.index]
            local i = symbol(int,exp.index.name)
            indextosym[exp.index] = i
            local e = emit(exp.exp)
            return quote
                var s : float = exp.op.identity
                for [i] = 0,size do
                    s = [ exp.op:emit(s, e) ]
                end
            in s end
        else error("unknown kind") end
    end
    local value = emit(self.exp)
    local r = symbol(&float,"r")
    local strides = densestride(sizes)
    local body = quote
        r[ [indicestoaddr(strides,syms)] ] = value 
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
    return sizes,strides,impl
end
function C.Tensor:dump()
    local body = List {}
    local function emitdim(idx,i)
        if i > #self.dims then 
            body:insert(tostring(self.data[idx]))
        else
            local sep = (i == #self.dims) and " " or ("\n"):rep(#self.dims - i)
            for j = 1,self.dims[i] do
                emitdim(idx,i+1)
                body:insert(sep)
                idx = idx + self.strides[i]
            end
        end
    end
    emitdim(0,1)
    body:insert("\n")
    return body:concat()
end

local A = Iota("A",2,3)
local x = Iota("x",3)
local c = Iota("b",2)
local d = Iota("d",3)   --is sparse

print(A:dump())

local a = (i..j)(A(i,j) + A(i,j))

local b = (i..sum(j))(A(i,j)*x(j) )

local r = Scalar("r",0)
r:set(b)

local B = Iota("B",3,4)



print(A:dump())
print(x:dump())
print(c:dump())
print(r:dump())

print("----------")

local mm = (i..j..sum(k))(A(i,k)*B(k,j))

r:set(mm)

print(B:dump())
print(r:dump())


J = Iota("J",4,5)
local p = Iota("p",5)

local jtjp = (i..sum(j)..sum(k))( J(j,i)*J(j,k)*p(k) )

print(jtjp)
r:set(jtjp)
print(r:dump())

print(J:dump())
print(J:get(":",4):dump())
--[[
    TODO: 
    separate compilation from execution
    figure out dense/sparse iterator interaction


(i,j,+k,+l) Im(i+k,j+l)*Filt(k,l)

(i,+j,k) A(i,j)*h(k)*v(j)

s = allocate (i,k)

for i:
    for j:
        if (j == 0) s[i,k] = 0;
        for k:
            s[i,k] += exp(i,j,k)




r = A*x + b

a' = (i,+j)(A(i,j)*x(j))
r = (i) a'(i) +b(i)

loop = fuse(a',r)

for i:
  for j:
    A(i,j)*x(j)
   + b(i)



(i,+k) A(i,k)*B(k)

r,c ==> dense, sparse

c,r ==> dense, sparse (unused)

A

sparse_it:
   num_elements() -> how many non-zeros: 0 ... S
   getelement(i) -> (0...S) -> N
   getlocation(i) -> (0...S) -> something you can use to look up the value of A
   
for i (0 to N):
    s = 0
    sparse_it = iterator_for_row(i) -- maybe a crs data-structure
    
    for k (in spares_it): --num_elements determines sizehh
       s += A(k:getlocation())*B(k:getelement(k))
    o(i) = s


Iterator Model:

A Tensor 'A' of dim N can be turned into a tensor of dim N-1 by indexing with _any_ of dimensions.

A' = A(:,i)
B' = A(i,:) -- note: syntax here won't actually exist outside of the IR for generating loop nests.

At compile time, the tensor knows the type of iterator associated with any of its remaining dimensions. The value that the
iterator produces is what is passed to object for indexing.

Whenever an iterator union or intersection happens, that operation also provides a transformation from the new iterators range
into the range of the original iterator.

Examples:
  dense * sparse => sparse, indices for dense objects are updated to first abstract the dense index from the sparse one
  dense + sparse => dense,  indices for sparse object are updated to find the sparse index from the dense index (which may be slow)
  
For prototyping, hard or slow cases can simply report 'NYI' for certain operations (e.g. union/intersection that is tricky, or for iteration of the object in the wrong order).

Example:

A is a CRS matrix, and x is a dense vector.

(i,+j) A(i,j)*x(j)

step 1: derive iterator for i:
       r1 = A(i,j) -> dense iterator over first index
       r2 = x(j) -> dense constant expression
       r3 = r1*r2 -> dense iterator over first index
       
step 2: derive iterator for j:
        A' = A(i,:)
        
       r1 = A'(j) -> sparse iterator over first index
       r2 = x(j) -> dense iterator over first index
       r3 = r1*r2 -> sparse iterator from r1, indexing expression for x gets updated with (sparseidxtodense(j))
       
]]
