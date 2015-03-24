local ffi = require("ffi")

local libraryformat = "lib%s.so"
if ffi.os == "Windows" then
    libraryformat = "%s.dll"
end

local headerformat = "%s.h"
terralib.includepath = terralib.terrahome.."/include"
local C = terralib.includecstring[[ 
    #include <stdio.h>
    #include <stdlib.h>
    #include "terra.h"
]]

local function saveaslibrary(libraryname, terrasourcefile)
    local success, tbl = xpcall(function() return assert(terralib.loadfile(terrasourcefile))() end,
                         function(err) return debug.traceback(err,2) end)
    if not success then error(tbl,0) end
    local apifunctions = terralib.newlist()
    for k,v in pairs(tbl) do
        if terralib.isfunction(v) then
            apifunctions[k] = v
        end
    end
    local wrappers = {}
    local statename = libraryname.."State"
    local LibraryState = terralib.types.newstruct(statename)
    
    local ctypes = terralib.newlist()-- declarations for struct types
    local cfunctions = terralib.newlist() -- list of lines in the header
    
    LibraryState.entries:insert { "L", &C.lua_State }
    for k,v in pairs(apifunctions) do
        LibraryState.entries:insert { k, &v:gettype() }
    end
    
    local terra doerror(L : &C.lua_State)
        C.printf("%s\n",C.luaL_checklstring(L,-1,nil))
        return nil
    end
    
    local source = io.open(terrasourcefile):read("*all")
    local source_sz = source:len()
    local terra NewState() : &LibraryState
        var S = [&LibraryState](C.malloc(sizeof(LibraryState)))
        var L = C.luaL_newstate();
        S.L = L
        if L == nil then return doerror(L) end
        C.luaL_openlibs(L)
        C.terra_init(L)
    
        if C.terra_loadbuffer(L,source,source_sz,terrasourcefile) ~= 0 or C.lua_pcall(L, 0, -1, 0) ~= 0 then return doerror(L) end
        
        escape
            for k,v in pairs(apifunctions) do
                emit quote
                    C.lua_getfield(L,-1,k)
                    C.lua_getfield(L,-1,"getpointer")
                    C.lua_insert(L,-2)
                    C.lua_call(L,1,1) 
                    S.[k] = @[&&v:gettype()](C.lua_topointer(L,-1))
                    C.lua_settop(L, -2)
                end
            end
        end
        return S
    end
    wrappers[libraryname.."_NewState"] =  NewState
    
    for k,v in pairs(apifunctions) do
        local typ = v:gettype()
        local syms = typ.parameters:map(symbol)
        local terra wfn(s : &LibraryState, [syms]) : typ.returntype
            return s.[k]([syms])
        end
        wrappers[libraryname.."_"..k] = wfn 
    end 
    
    local typeemitted = {}
    local function typetostring(T)
        if T == rawstring then
            return "const char *" --HACK: make it take strings
        elseif T:ispointer() then
            local p
            if T.type:isstruct() then
                p = tostring(T.type):match("%.?([^.]*)$")
                if not typeemitted[T.type] then
                    ctypes:insert("typedef struct "..p.." "..p..";\n")
                    typeemitted[T.type] = true
                end
            else
                p = typetostring(T.type)
            end
            return p.." *"
        elseif T:isprimitive() then
            return T:cstring()
        elseif T == terralib.types.unit or T == opaque then
            return "void"
        else
            error("unsupported type: "..tostring(T))
        end
    end
    
    local names = terralib.newlist()
    for k,v in pairs(wrappers) do names:insert(k) end
    names:sort()
    for i,k in ipairs(names) do
        local v = wrappers[k]
        local typ = v:gettype()
        cfunctions:insert(string.format("%s %s(%s);\n",typetostring(typ.returntype),k,table.concat(typ.parameters:map(typetostring),",")))
    end
    
    local cheader = io.open(string.format(headerformat,libraryname),"w")
    cheader:write '#include "terra.h"\n#include <stdint.h>\n'
    for i,l in ipairs(ctypes) do cheader:write(l) end
    for i,l in ipairs(cfunctions) do cheader:write(l) end
    cheader:close()
    local libraryfmtname = string.format(libraryformat,libraryname)
    
    local flags = {}
    if ffi.os == "OSX" then
        flags = { "-install_name", "@rpath/"..libraryfmtname }
    elseif ffi.os == "Windows" then
		flags = terralib.newlist { string.format("/IMPLIB:%s.lib",libraryname),terralib.terrahome.."\\terra.lib",terralib.terrahome.."\\lua51.lib" }
        
		for i,k in ipairs(names) do
            flags:insert("/EXPORT:"..k)
        end
    end
    terralib.saveobj(libraryfmtname,wrappers,flags)
end

--[[
saveaslibrary("mylib","testlib.t")

local C2 = terralib.includec("mylib.h","-I.")
terra main()
    var r = C2.mylib_newstate()
    C2.mylib_foo(r)
    return 0
end

local flags = terralib.newlist {"-L.","-L", terralib.terrahome,"-Wl,-rpath,"..terralib.terrahome,"-lterra","-lmylib"}
if require("ffi").os == "OSX" then
    flags:insertall {"-pagezero_size","10000", "-image_base", "100000000"}
end

terralib.saveobj("dynlib",{main = main},flags)
assert(0 == os.execute("./dynlib"))
]]

saveaslibrary(arg[1],arg[2])