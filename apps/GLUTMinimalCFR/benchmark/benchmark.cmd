REM run 'benchmark <folder> <filter> <params> <result>
REM params.txt has the following format
REM alias1;parameters for benchmark tool
REM alias2;parameters for benchmark tool
REM example: benchmark models *.nbf params.txt result.txt
REM result.txt contains
REM alias1;model1;fps
REM alias2;model1;fps
REM alias1;model2;fps
REM alias2;model2;fps

@ECHO off
SETLOCAL enabledelayedexpansion
set DPHOME=
set MODELPATH=%1%
set FILTER=%2%
set RESULT=%4%
set PARAMS=%3%
dir /b %MODELPATH%\%FILTER% >models.txt

erase /Q %RESULT%

FOR /F "tokens=1 delims=," %%A IN (models.txt) DO (
  SET MODELBASE=%1%\%%A
  FOR /F "tokens=1,2 delims=;" %%B IN (%PARAMS%) DO (
    Bench.exe --filename %MODELPATH%\%%A %%C
    echo %%A,%%B,!ERRORLEVEL! >>%RESULT%
  )
)
