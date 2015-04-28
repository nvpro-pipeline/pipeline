@echo off

set USE_VS2013=1
REM set USE_VS2012=1
REM set USE_VS2010=1

set VS2013_ENV_CMD="%VS120COMNTOOLS%..\..\VC\vcvarsall.bat"
set VS2012_ENV_CMD="%VS110COMNTOOLS%..\..\VC\vcvarsall.bat"
set VS2010_ENV_CMD="%VS100COMNTOOLS%..\..\VC\vcvarsall.bat"

if "%USE_VS2013%"=="1" (
  call %VS2013_ENV_CMD% amd64
  if not exist builds\vc12-amd64 mkdir builds\vc12-amd64
) else if "%USE_VS2012%"=="1" (
  call %VS2012_ENV_CMD% amd64
  if not exist builds\vc11-amd64 mkdir builds\vc11-amd64
) else if "%USE_VS2010%"=="''" (
  call %VS2010_ENV_CMD% amd64
  if not exist builds\vc10-amd64 mkdir builds\vc10-amd64
) else (
    goto visual_studio_not_found
)

set DPHOME=%~dp0
set DP_3RDPARTY_PATH=%DPHOME%3rdparty

:build_3rdparty
cmake -P 3rdPartyBuild.cmake
pause
goto :eof

:visual_studio_not_found
echo Error: No compatible version of Visual Studio was found
