echo Remaking Opt.dll
set TERRAHOME=%1
set TERRA=%TERRAHOME%\release\bin\terra
set OPT_DIR=%2

%TERRA% %OPT_DIR%\src\createWrapper.t Opt %OPT_DIR%\src o %OPT_DIR%\release\include\Opt.h .\Opt.dll true
MOVE "./Opt.dll" "%OPT_DIR%\release\bin\"
MOVE "./Opt.exp" "%OPT_DIR%\release\bin\"
MOVE "./Opt.lib" "%OPT_DIR%\release\lib\"