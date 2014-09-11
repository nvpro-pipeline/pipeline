set DESTINATION=Viewer
mkdir %DESTINATION%
copy Viewer.exe  %DESTINATION%
copy DAELoader.nxm %DESTINATION%
copy NBFLoader.nxm %DESTINATION%
copy NBFSaver.nxm %DESTINATION%
copy iltexloader.nxm %DESTINATION%
copy RiXGl.rdr  %DESTINATION%
copy RixCore.dll %DESTINATION%
COPY RIXfx.dll %DESTINATION%
copy DevIL.dll %DESTINATION%
copy Qt*.dll %DESTINATION%
copy cg*.dll %DESTINATION%
copy DP.dll %DESTINATION%
copy Dpfx.dll %DESTINATION%
copy dpsg*.dll %DESTINATION%
copy dpmath.dll %DESTINATION%
copy dputil.dll %DESTINATION%
copy dpgl.dll %DESTINATION%
copy dpui.dll %DESTINATION%
copy dpcuda*.dll %DESTINATION%
copy dpculling*.dll %DESTINATION%
copy dpsg*.dll %DESTINATION%
copy libmdl.dll %DESTINATION%
copy glew32.dll %DESTINATION%
copy freeglut.dll %DESTINATION%
copy optix*.dll %DESTINATION%
copy cuda*.dll %DESTINATION%

xcopy ptx %DESTINATION%\ptx /S /C /I /Y
xcopy %DPHOME%\media\dpfx %DESTINATION%\media\dpfx /S /C /I /Y
xcopy %DPHOME%\media\effects %DESTINATION%\media\effects /S /C /I /Y
xcopy %DPHOME%\media\effects %DESTINATION%\media\effects /S /C /I /Y
xcopy %DPHOME%\dp\fx\xmd\res %DESTINATION%\dp\fx\xmd\res /S /C /I /Y
xcopy %DPHOME%\apps\Viewer\res %DESTINATION%\apps\Viewer\res /S /C /I /Y
xcopy platforms %DESTINATION%\platforms /S /C /I /Y
copy EffectLoaderXMD.cfg %DESTINATION%

