@echo off

set VS2013_ENV_CMD="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat"
set VS2012_ENV_CMD="C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\vcvarsall.bat"

if exist %VS2013_ENV_CMD% (
  call %VS2013_ENV_CMD%
) else if exist %VS2012_ENV_CMD% (
  call %VS2012_ENV_CMD%
) else (
  goto visual_studio_not_found
)

set DPHOME=%~dp0
set DP_3RDPARTY_PATH=%DPHOME%3rdparty

:build_3rdparty
cmake -P 3rdPartyBuild.cmake
mkdir builds
goto :eof

:visual_studio_not_found
echo Error: No compatible version of Visual Studio was found
