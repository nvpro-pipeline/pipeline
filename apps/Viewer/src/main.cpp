// Copyright NVIDIA Corporation 2009-2010
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#if defined( Q_WS_WIN )
// For the DebugBreak();
#include <windows.h>
#endif

#include <dp/sg/core/nvsg.h>

#include "Viewer.h"

// This is in res/ui and contains a QString appStyleSheet.
#include "StyleSheet.h"

// #include <crtdbg.h> // Track leaks on exit.

// This is a trick to get the shutdown sequence of Qt aligned with SceniX data stored at SceniXQGLWidgets.
// Widgets were deleted, after the nvsgTerminate was called otherwise.
int runApp(int argc, char *argv[] )
{
  Viewer app( argc, argv );

  // This could happen if SceniXQGLContext::create() did not get a 
  // suitable pixelformat inside RenderContextGLFormat::getPixelFormat().
  if ( !app.getGlobalShareGLWidget() || !app.getGlobalShareGLWidget()->getRenderContext() )
  {
    return -3;
  }

  app.addLibraryPath( app.applicationDirPath() );

  app.setStyleSheet(appStyleSheet);
  app.startup();

  return app.exec();
}

int main( int argc, char *argv[] )
{
  // _CrtSetBreakAlloc(/* Put CRT leak number here. */);
  dp::sg::core::nvsgInitialize();

#if !defined( NDEBUG )
  // Debug executable performance is faster without dp::sg::core::NVSG_DBG_LEAK_DETECTION.
  // Enable it if there is a memory leak assertion on program end.
  dp::sg::core::nvsgSetDebugFlags( dp::sg::core::NVSG_DBG_ASSERT /* | dp::sg::core::NVSG_DBG_LEAK_DETECTION */ );
#endif

  int result = runApp(argc, argv);

  dp::sg::core::nvsgTerminate();

  return result;
}
