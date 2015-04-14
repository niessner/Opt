-- partial implementation of dstar work, moved out of ad.t for now to keep ad.t stuff we are actually using

function Apply:calcd(X)
    -- extract derivative graph
    assert(Var:is(X))
    local succ = {}
    local pred = {}
    local function insertv(g,k,v)
        g[k] = g[k] or terralib.newlist()
        g[k]:insert(v)
    end
    local function deletev(g,k,v)
        local es = g[k]
        for i,e in ipairs(es) do
            if e == v then
                es:remove(i)
                return
            end
        end
        error("not found")
    end
    local function insert(e)
        insertv(succ,e.from,e)
        insertv(pred,e.to,e)
    end
    local function remove(e)
        removev(succ,e.from,e)
        removev(pred,e.to,e)
    end
   
    local ingraph = {}
    local function visit(n)
        if ingraph[n] ~= nil then return ingraph[n] end
        local partials = n:partials()
        local include = n == X
        for i,c in ipairs(n:children()) do
            if visit(c) then    
                include = true
                insert { from = c, to = n, d = partials[i] }
            end
        end
        ingraph[n] = include
        return include
    end
    visit(self)

    local function simple(x)
        if not pred[x] then return terralib.newlist{ toexp(1) } end
        local r = terralib.newlist()
        for i,p in ipairs(pred[x]) do
            local s = simple(p.from)
            r:insertall(s:map(function(v) return v*p.d end))
        end
        return r
    end
    local r
    local factors = simple(self)
    for i,f in ipairs(factors) do
        r = (r and r + f) or f
    end
    return r or toexp(0)
end
    
local function dominators(startnode,succ)
    -- calculate post order traversal order
    local visited = {}
    local nodes = terralib.newlist()
    local nodeindex = {}
    local function visit(n)
        if visited[n] then return end
        visited[n] = true
        for i,c in ipairs(succ(n)) do
            visit(c)
        end
        nodes:insert(n)
        nodeindex[n] = #nodes
    end
    visit(startnode)
    
    -- calculate predecessors (postorderid -> list(postorderid))
    local pred = {}
    
    for i,n in ipairs(nodes) do
        for j,c in ipairs(succ(n)) do
            local ci = nodeindex[c]
            pred[ci] = pred[ci] or terralib.newlist()
            pred[ci]:insert(i)
        end
    end
    
    assert(nodeindex[startnode] == #nodes)
    -- calculate immediate dominators
    local doms = terralib.newlist{}
    
    doms[#nodes] = #nodes
    
    local function intersect(finger1,finger2)
        while finger1 ~= finger2 do
            while finger1 < finger2 do
                finger1 = assert(doms[finger1])
            end
            while finger2 < finger1 do
                finger2 = assert(doms[finger2])
            end
        end
        return finger1
    end
    
    local changed = true
    while changed do
        changed = false
        --[[
        for i = #nodes,1,-1 do
            print(i,doms[i])
        end
        print()]]
        for b = #nodes-1,1,-1 do
            local bpred = pred[b]
            local newidom
            for i,p in ipairs(bpred) do
                if doms[p] then
                    newidom = newidom and intersect(p,newidom) or p
                end
            end
            if doms[b] ~= newidom then
                doms[b] = newidom
                changed = true
            end
        end
    end
    
    local r = {}
    for i,n in ipairs(nodes) do
        r[n] = nodes[doms[i]]
    end
    return r
end

if false then
    local n1,n2,n3 = {name = "n1"},{name = "n2"},{name = "n3"}
    n1[1] = n2
    n2[1] = n1
    n2[2] = n3
    n3[1] = n2

    local n4 = { n3, n2 , name = "n4" }
    local n5 = { n1, name = "n5"}
    local n6 = {n4,n5, name = "n6"}

    local d = dominators(n6,function(x) return x end)

    for k,v in pairs(d) do
        print(k.name," <- ",v.name)
    end
end