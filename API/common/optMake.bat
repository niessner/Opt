echo Remaking Opt.dll
set TERRAHOME=%1
set TERRA=%TERRAHOME%\release\terra
set LUAHOME=%TERRAHOME%\..\LuaJIT-2.0.3
set OPT_DIR=%2
set TARGET_DIR=%3


cd %OPT_DIR%
%TERRA% ..\..\API\src\createWrapper.t Opt ..\..\API\src\windowsErrorContextWrapper.t
cd %TARGET_DIR%
copy %OPT_DIR%\Opt.dll .
copy %OPT_DIR%\Opt.lib .
copy %OPT_DIR%\Opt.h .
