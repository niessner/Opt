local ffi = require("ffi")

local setupsigsegv

local libraryformat = "lib%s.so"

local headerformat = "%s.h"
terralib.includepath = terralib.terrahome.."/include"

local C,CN = terralib.includecstring[[ 
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #ifndef _WIN32
    #include <dlfcn.h>
    #include <libgen.h>
    #include <signal.h>
    sig_t SIG_DFL_fn() { return SIG_DFL; }
    #else
    #define NOMINMAX
    #include <Windows.h>
    #include <Shlwapi.h>
    #endif
    #include "terra.h"
]]



local LUA_GLOBALSINDEX = -10002

local pathforsymbol

if ffi.os == "Windows" then
    libraryformat = "%s.dll"
    terra setupsigsegv(L : &C.lua_State) end
    terra directoryforsymbol(sym : &opaque, buf : rawstring, N : int)
        var mbi : C.MEMORY_BASIC_INFORMATION;
        C.VirtualQuery(sym, &mbi, sizeof(C.MEMORY_BASIC_INFORMATION));
        C.GetModuleFileNameA([C.HINSTANCE](mbi.AllocationBase), buf, N);
        C.PathRemoveFileSpecA(buf);
    end  
else
    
    local terratraceback = global(&opaque -> {})
    
    local terra sigsegv(sig : int, info : &C.siginfo_t, uap : &opaque)
        C.signal(sig,C.SIG_DFL_fn())  --reset signal to default, just in case traceback itself crashes
        terratraceback(uap)
        C.raise(sig)
    end
    terra setupsigsegv(L : &C.lua_State)
        C.lua_getfield(L, LUA_GLOBALSINDEX,"terralib");
        C.lua_getfield(L, -1, "traceback");
        var tb = C.lua_topointer(L,-1);
        if tb == nil then return end
        terratraceback = @[&(&opaque -> {})](tb)
        var sa : CN.sigaction
        sa.sa_flags = C.SA_RESETHAND or C.SA_SIGINFO
        C.sigemptyset(&sa.sa_mask)
        sa.__sigaction_u.__sa_sigaction = sigsegv
        C.sigaction(C.SIGSEGV, &sa, nil)
        C.sigaction(C.SIGILL, &sa, nil)
        C.lua_settop(L,-3)
    end
    
    terra directoryforsymbol(sym : &opaque, buf : rawstring, N : int)
        var info : C.Dl_info
	    C.dladdr(sym, &info)
	    var full = C.realpath(info.dli_fname, buf);
	    buf[C.strlen(C.dirname(full))] = 0
    end
end

local function saveaslibrary(libraryname, terrasourcefile)
    local filename = terrasourcefile:match("[^/]*.t$")
    local terrapath, packagename = "/"..terrasourcefile:sub(1,-#filename-1).."?.t;",filename:sub(1,-3)
    package.terrapath = package.terrapath..";."..terrapath
    local success, tbl = xpcall(function() return assert(require(packagename)) end,
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
    
    local terra NewState() : &LibraryState
        var S = [&LibraryState](C.malloc(sizeof(LibraryState)))
        var L = C.luaL_newstate();
        S.L = L
        if L == nil then return doerror(L) end
        C.luaL_openlibs(L)
        var o  = C.terra_Options { verbose = 0, debug = 1, usemcjit = 0 }
        C.terra_initwithoptions(L,&o)
        
        setupsigsegv(L)
        var path : int8[4096]
        directoryforsymbol(NewState,path,4096)
        C.strncat(path,terrapath,4096)
        C.lua_getfield(L,LUA_GLOBALSINDEX,"package")
        C.lua_getfield(L,-1,"terrapath")
        C.lua_pushstring(L,path)
        C.lua_concat(L,2)
        C.lua_setfield(L,-2,"terrapath")
        C.lua_getfield(L,LUA_GLOBALSINDEX,"require")
        C.lua_pushstring(L,packagename)
        if C.lua_pcall(L,1,1,0) ~= 0 then return doerror(L) end
        
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
		flags = terralib.newlist { string.format("/IMPLIB:%s.lib",libraryname),terralib.terrahome.."\\terra.lib",terralib.terrahome.."\\lua51.lib","Shlwapi.lib" }
        
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