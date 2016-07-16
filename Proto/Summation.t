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
    
    AbstractTensor = Tensor(string name, number* dims, number* strides, number allocatedsize, any data)
                   | DummyTensor(string name, number * dims, function generator)
    Index = (string name)
    
    IndexDef = (number dim, Iterator iterator)
          
    MapExp = (Index* indices, ScalarExp exp, IndexDef* defs)
    
    Op = (string name)
    
    ScalarExp = Apply(Op op,ScalarExp* expressions)
              | Reduce(Op op, Index index, ScalarExp exp, IndexDef? def)
              | Constant(number c)
              | Load(AbstractTensor tensor, Index* indices)


    IndexList = (any* indices)
    ReduceObj = (Op op, Index index)
    
    Iterator = Dense
             | Compressed(any key) unique

    Shape = (Iterator* iterators) unique # added to each ScalarExp on typechecking
]]

local function Environment(parent)
    return parent and setmetatable({},{__index = parent}) or {}
end

local function Copy(ref,newfields) -- copy an object, extracting any new replacement fields from newfields table
    newfields = newfields or {}
    local class = getmetatable(ref)
    local fields = class.__fields
    assert(fields,"not a asdl object?")
    local function handlefield(i,...) -- need to do this with tail recursion rather than a loop to handle nil values
        if i == 0 then
            return class(...)
        else
            local f = fields[i]
            local a = newfields[f.name] or ref[f.name]
            newfields[f.name] = nil
            return handlefield(i-1,a,...)
        end
    end
    local r = handlefield(#fields)
    for k,v in pairs(newfields) do
        error("unused field in copy: "..tostring(k))
    end
    return r
end


local add,mul = C.Op("+"), C.Op("*")
local i,j,k = C.Index("i"),C.Index("j"),C.Index("k")

function C.Iterator:Union(rhs)
    if self.kind == "Dense" or rhs.kind == "Dense" then return "Dense"
    elseif self == rhs then return self
    else error(("cannot union iterators %s and %s"):format(tostring(self),tostring(rhs))) end
end
function C.Iterator:Intersect(rhs)
    if self.kind == "Dense" then return rhs
    elseif rhs.kind == "Dense" then return self
    elseif self == rhs then return self
    else error(("cannot intersect iterators %s and %s"):format(tostring(self),tostring(rhs))) end
end

local function combineshapes(expressions,merge)
    assert(#expressions > 0, "no expressions?")
    local iterators = List {}
    local first = expressions[1].shape
    for i,s in ipairs(first.iterators) do
        for j = 2,#expressions do
            s = merge(s,expressions[j].shape.iterators[i])
        end
        iterators:insert(s)
    end
    return C.Shape(iterators)
end

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
local empty = List {}
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
            return C.MapExp(List { unpack(self.indices,1,i-1) },exp, empty)
        end
    end
    return C.MapExp(self.indices,exp, empty)
end
function C.Index:__call(exp)
    return C.MapExp(List{self},exp, emtpty)
end

local function sum(i)
    return C.ReduceObj(add,i)
end

function C.ReduceObj:__call(exp)
    return C.Reduce(self.op,self.index,exp)
end

function C.AbstractTensor:__call(...)
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

function Dummy(name,a,b,exp)
    return C.DummyTensor(name,List {a,b}, exp)
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
function C.AbstractTensor:__tostring() return self.name end
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

function C.ScalarExp:withshape(shape)
    self.shape = shape
    return self
end
function C.Shape:afterreduce()
    return C.Shape( List {unpack(self.iterators,1,#self.iterators-1)} )
end

function C.Tensor:iteratorkindsfornesting(depths)
    return depths:map(function(d) return C.Dense end)
end

local EveryOther = {}

function EveryOther:indexterratype() return int end
function EveryOther:todense(sym) return `sym end

function C.DummyTensor:iteratorkindsfornesting(depths)
    assert(#depths == 2)
    if depths[1] < depths[2] then
        return List { C.Dense, C.Compressed(EveryOther) }
    else
        return List { C.Compressed(EveryOther), C.Dense }
    end
end


local function rep(v,n)
    local r = List{}
    for i = 1,n do r[i] = v end
    return r
end
add.merge = function(a,b) return a:Union(b) end
mul.merge = function(a,b) return a:Intersect(b) end

function C.MapExp:typecheck()
    
    local indextodepth = Environment()
    local currentdims = List {}
    
    local function defineindex(i)
        currentdims:insert("undefined")
        indextodepth[i] = #currentdims
    end
    
    local function check(exp)
        if "Constant" == exp.kind then
            return Copy(exp):withshape(rep(C.Dense,#currentdims))
        elseif "Apply" == exp.kind then
            local nexp = exp.expressions:map(check)
            local nshape = combineshapes(nexp,exp.op.merge)
            return Copy(exp,{expressions = nexp}):withshape(nshape)
        elseif "Load" == exp.kind then
            local depths = List {}
            for dimidx,i in ipairs(exp.indices) do
                local depth = assert(indextodepth[i],"free index found")
                if currentdims[depth] == "undefined" then
                    currentdims[depth] = exp.tensor.dims[dimidx]
                end
                assert(currentdims[depth] == exp.tensor.dims[dimidx],"indices do not match dimension")
                depths:insert(depth)
            end
            local iteratorsfortensor = exp.tensor:iteratorkindsfornesting(depths)
            local iterators = rep(C.Dense,#currentdims)
            for i,d in ipairs(depths) do
                iterators[d] = iteratorsfortensor[i]
            end
            local shape = C.Shape(iterators)
            return Copy(exp):withshape(shape)
        elseif "Reduce" == exp.kind then
            local oindextodepth = indextodepth
            indextodepth = Environment(indextodepth)
            defineindex(exp.index)
            local nexp = check(exp.exp)
            local dim = currentdims:remove()
            assert(dim ~= "undefined","unused dimension")
            indextodepth = oindextodepth
            local def = C.IndexDef(dim,nexp.shape.iterators[#currentdims+1])
            return Copy(exp,{ exp = nexp, def = def }):withshape(nexp.shape:afterreduce())
        else error("unknown kind") end
    end 
    
    for _,i in ipairs(self.indices) do
        defineindex(i)
    end
    local nexp = check(self.exp)
    local defs = List{}
    for i,d in ipairs(currentdims) do
        assert(d ~= "undefined", "unused dimension")
        defs:insert(C.IndexDef(d,nexp.shape.iterators[i]))
    end
    
    return Copy(self,{ exp = nexp, defs = defs })
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

function C.Iterator:indexterratype()
    if self.kind == "Dense" then return int
    elseif self.kind == "Compressed" then return self.key:indexterratype() end
end

function C.Iterator:transformindex(fromiterator,fromindex)
    if self == C.Dense and fromiterator.kind == "Compressed" then
        return fromiterator.key:todense(fromindex)
    end
    error("NYI - transform iterator")
end

function C.Tensor:access(isyms)
    local addr = indicestoaddr(self.strides,isyms)
    return `self.data[ addr ]
end
function C.DummyTensor:access(isyms)
    return self.generator(unpack(isyms))
end

function C.Dense:generateloop(dim,sym,body)
    return quote for [sym] = 0,dim do body end end
end
function C.Compressed:generateloop(dim,sym,body)
    return self.key:generateloop(dim,sym,body)
end
function EveryOther:generateloop(dim,sym,body)
    return quote for [sym] = 0,dim,2 do body end end
end

function C.MapExp:compile()
    local indexstack = List {}
    local indextosym = Environment()
    local typedexp = self:typecheck()
    local function emit(exp)
        if "Constant" == exp.kind then return `exp.c
        elseif "Apply" == exp.kind then
            local args = exp.expressions:map(function(e)
                if e.shape == exp.shape then
                    return emit(e)
                end
                local oindex = indextosym
                indextosym = Environment(indextosym)
                -- adjust indices for new shape
                local newindices = List {}
                for i,it in ipairs(e.shape.iterators) do
                    if it ~= exp.shape.iterators[i] then
                        local index = indexstack[i]
                        local sym = symbol(it:indexterratype(),index.name)
                        local oldsym = assert(indextosym[index],"index not defined?")
                        indextosym[index] = sym
                        newindices:insert quote
                            var [sym] = [ it:transformindex(exp.shape.iterators[i],oldsym) ]
                        end
                    end
                end
                local body = emit(e)
                indextosym = oindex
                return quote
                    [newindices]
                in body end
            end)
            return exp.op:emit(unpack(args))
        elseif "Load" == exp.kind then
            local isyms = exp.indices:map(function(x) return indextosym[x] end)
            return exp.tensor:access(isyms)
        elseif "Reduce" == exp.kind then
            local oindex = indextosym
            local sym = symbol(exp.def.iterator:indexterratype(),exp.index.name)
            indextosym = Environment(indextosym)
            indextosym[exp.index] = sym
            indexstack:insert(exp.index)
            local e = emit(exp.exp)
            
            local result = quote
                var s : float = exp.op.identity
                [ exp.def.iterator:generateloop(exp.def.dim,sym, quote s = [ exp.op:emit(s,e)] end) ]
            in s end
            
            indextosym = oindex
            indexstack:remove()
            return result
        else error("unknown kind") end
    end
    
    -- TODO: support sparse output dimensions.
    for _,index in ipairs(typedexp.indices) do
        indextosym[index] = symbol(int,index.name)
        indexstack:insert(index)
    end
    local value = emit(typedexp.exp)
    local r = symbol(&float,"r")
    local sizes = typedexp.defs:map("dim")
    local strides = densestride(sizes)
    local syms = indexstack:map(function(i) return indextosym[i] end)
    local addr = indicestoaddr(strides,syms)
    local body = quote
        r[addr] = value 
    end
    for i = #typedexp.indices,1,-1 do 
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


-- Iterator tests:

local d = C.Dense
local s1 = C.Compressed(1)
print(d:Union(s1))
print(d:Union(d))
print(s1:Union(d))
print(d:Intersect(s1))
print(d:Intersect(d))
print(s1:Intersect(d))

local A = Dummy("A",2,4,function(x,y) return `x*1000+y end)
local x = Iota("p",4)

local exp = (i..sum(j))(A(i,j)*x(j))

r:set(exp)
print(r:dump())

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

A -> (Dense,Compressed(1))
x -> (Dense,Dense)

A*x -> (Dense,Compressed(i))


Dense,Compressed(1),Dense
Compressed(1),Dense,Compressed(2)


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
