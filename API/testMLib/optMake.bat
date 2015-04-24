echo Remaking Opt.dll
set TERRAHOME=..\..\..\terra
set TERRA=%TERRAHOME%\release\terra
set LUAHOME=..\..\..\LuaJIT-2.0.3

cd ..\build\
%TERRA% ..\src\createWrapper.t Opt ..\src\windowsErrorContextWrapper.t
cd ..\testMLib\
copy ..\build\Opt.dll .
copy ..\build\Opt.lib .
copy ..\build\Opt.h .
