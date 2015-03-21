local L = {}

terra L.foo()
    print(3)
end

terra L.add(a : int, b : int) 
    return a + b 
end

local struct Foo { a : int }
terra L.DoFoo(a : &Foo)
    print(4)
    return 4
end

return L