@echo off
SETLOCAL enabledelayedexpansion
:: --------------------------------------------------------------------------
:: -- Set some default settings --
:: --------------------------------------------------------------------------
set SCENE_RESULTS=results.txt
set SCENE_REPETITIONS=32
set RDR_SELECTED=RiXGL.rdr
set RDR_RENDERENGINE=Bindless
set RDR_SHADERMANAGER=rixfx:uniform
set RDR_CULLINGENGINE=none

set /p TEST_LAUNCH_SITE=< ../test_launch_site.txt  
:: --------------------------------------------------------------------------
:: -- SET DEFAULTS DONE --
:: --------------------------------------------------------------------------

:: --------------------------------------------------------------------------
:: -- parse available command line options --
:: --------------------------------------------------------------------------
:PARAMPARSE
if "%1"=="" goto PARAMPARSE_DONE
set ARG=%1
shift
if /i "%ARG%"=="--directory"  goto PARAMSET_DIRECTORY
if /i "%ARG%"=="--filter"  goto PARAMSET_FILTER
if /i "%ARG%"=="--results"  goto PARAMSET_RESULTS
if /i "%ARG%"=="--repetitions"  goto PARAMSET_REPETITIONS
if /i "%ARG%"=="--renderer"  goto PARAMSET_RENDERER
if /i "%ARG%"=="--renderengine"  goto PARAMSET_RENDERENGINE
if /i "%ARG%"=="--shadermanager"  goto PARAMSET_SHADERMANAGER
if /i "%ARG%"=="--cullingengine"  goto PARAMSET_CULLINGENGINE
goto PARAMPARSE_ERROR

:: ------------------------------------------------------
:: -- set currently parsed command line option --
:: ------------------------------------------------------
:PARAMSET_DIRECTORY
set SCENE_DIRECTORY=%1
goto PARAMPARSE_NEXT

:PARAMSET_FILTER
set SCENE_FILTER=%1
goto PARAMPARSE_NEXT

:PARAMSET_RESULTS
set SCENE_RESULTS=%1
goto PARAMPARSE_NEXT

:PARAMSET_RESULTS
set SCENE_RESULTS=%1
goto PARAMPARSE_NEXT

:PARAMSET_REPETITIONS
set SCENE_REPETITIONS=%1
goto PARAMPARSE_NEXT

:PARAMSET_RENDERER
set RDR_SELECTED=%1
goto PARAMPARSE_NEXT

:PARAMSET_RENDERENGINE
set RDR_RENDERENGINE=%1
goto PARAMPARSE_NEXT

:PARAMSET_SHADERMANAGER
set RDR_SHADERMANAGER=%1
goto PARAMPARSE_NEXT

:PARAMSET_CULLINGENGINE
set RDR_CULLINGENGINE=%1
goto PARAMPARSE_NEXT
:: ------------------------------------------------------
:: -- SETTING DONE --
:: ------------------------------------------------------

:: ----------------------------------------------------------------
:: -- parse next command line option --
:: ----------------------------------------------------------------
:PARAMPARSE_NEXT
shift
goto PARAMPARSE

:: ----------------------------------------------------------------
:: -- report unavailable command line option --
:: ----------------------------------------------------------------
:PARAMPARSE_ERROR
echo unavailable command line option
goto ALL_DONE

:PARAMPARSE_DONE
:: --------------------------------------------------------------------------
:: -- PARSING DONE --
:: --------------------------------------------------------------------------


echo ---------------------
echo SCENE_DIRECTORY=%SCENE_DIRECTORY%
echo SCENE_FILTER=%SCENE_FILTER%
echo SCENE_REPETITIONS=%SCENE_REPETITIONS%
echo SCENE_RESULTS=%SCENE_RESULTS%
echo RDR_SELECTED=%RDR_SELECTED%
echo RDR_RENDERENGINE=%RDR_RENDERENGINE%
echo RDR_SHADERMANAGER=%RDR_SHADERMANAGER%
echo RDR_CULLINGENGINE=%RDR_CULLINGENGINE%
echo TEST_LAUNCH_SITE=%TEST_LAUNCH_SITE%
echo ---------------------

set launch_DIR=%CD%
cd %TEST_LAUNCH_SITE%

dir /b %SCENE_DIRECTORY%\%SCENE_FILTER% >models.txt

FOR /F "tokens=1 delims=," %%A IN (models.txt) DO (
    
	set MEASURED_TIME_RESULT_FILE_SUFFIX=_%%A_%RDR_RENDERENGINE%_TEMP_TIME_DUMP
    DPTApp.exe --tests sg_benchmarks --backend DPTSgRdr.bkd --renderer %RDR_SELECTED% --renderengine %RDR_RENDERENGINE% --cullingengine %RDR_CULLINGENGINE% --shadermanager %RDR_SHADERMANAGER% --width 1024 --height 768 --resultsDir %CD% --resultsFilenameSuffix !MEASURED_TIME_RESULT_FILE_SUFFIX! --formattedOutputType csv --mf Timer --filename %SCENE_DIRECTORY%\%%A --repetitions %SCENE_REPETITIONS%
	%launch_DIR%\calc_FPS.py %CD%\benchmark_model!MEASURED_TIME_RESULT_FILE_SUFFIX!_run.csv > tmpFilefps.txt
	
	echo %%A,%RDR_RENDERENGINE%,%RDR_SHADERMANAGER%,!ERRORLEVEL! >>%SCENE_RESULTS%
	
)

cd %launch_DIR%

:ALL_DONE