mkdir Benchmark
copy GLUTMinimal.exe  Benchmark\bench.exe
copy NBFLoader.nxm Benchmark
copy iltexloader.nxm Benchmark
copy HOOPSLoader.nxm Benchmark
copy RiXGl.rdr  Benchmark
copy RixCore.dll Benchmark
COPY RIXfx.dll Benchmark
copy DevIL.dll Benchmark
copy Qt*.dll Benchmark
copy cg*.dll Benchmark
copy Dpfx.dll Benchmark
copy dpsg*.dll Benchmark
copy dpmath.dll Benchmark
copy dputil.dll Benchmark
copy dpgl.dll Benchmark
copy dpui.dll Benchmark
copy dpcuda*.dll Benchmark
copy dpculling*.dll Benchmark
copy dpsg*.dll Benchmark
copy libmdl.dll Benchmark
copy glew32.dll Benchmark
copy freeglut.dll Benchmark

xcopy ptx Benchmark\ptx /S /C /I /Y
xcopy %DPHOME%\media\dpfx Benchmark\media\dpfx /S /C /I /Y
xcopy %DPHOME%\media\effects Benchmark\media\effects /S /C /I /Y

copy %DPHOME%\Apps\GLUTMinimal\Benchmark\benchmark.cmd Benchmark

REM HOOPS
for %%A in (
  A3DLIBS.dll
  catstep30.dll
  cgrstep30.dll
  cv5step30.dll
  cvstep30.dll
  dccstep30.dll
  iconv.dll
  Ideasstep30.dll
  igestep30.dll
  ImageMagick.dll
  Invstep30.dll
  JTstep30.dll
  prostep30.dll
  r3dxmlstep30.dll
  sdstep30.dll
  sestep30.dll
  slwstep30.dll
  stepstep30.dll
  stlstep30.dll
  TfFontMgr.dll
  TfKernel.dll
  TFKGEOM.dll
  TFUGEOM.dll
  u3dstep30.dll
  ugstep30.dll
  vdastep30.dll
  wiges.dll
  wrlstep30.dll
  wsat.dll
  wstl.dll
  wstp.dll
  wxt.dll
  xtstep30.dll
  Xvlstep30.dll
) DO COPY %%A Benchmark
