// Copyright NVIDIA Corporation 2012
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


#define GLUT_DISABLE_ATEXIT_HACK

#include <test/rix/gl/framework/RiXGLBackend.h>
#include <dp/rix/gl/RiXGL.h>

#include <dp/util/File.h>
#include <dp/gl/RenderTargetFB.h>
#include <dp/gl/RenderTargetFBO.h>

#include <windows.h>
#include <GL/freeglut.h>

#include <dp/util/DPAssert.h>
#include <dp/util/SharedPtr.h>
#include <iostream>
#include <tchar.h>


namespace dp
{
  namespace rix
  {
    namespace gl
    {
      namespace test
      {
        namespace framework
        {

          using namespace core::test::framework;

          RiXGLBackendSharedPtr RiXGLBackend::create( char const* renderer, char const* options )
          {
            return( std::shared_ptr<RiXGLBackend>( new RiXGLBackend( renderer, options ) ) );
          }

          RiXGLBackend::RiXGLBackend(const char* renderer, const char* options)
            : RiXBackend(renderer, options)
          {
            char* dummyChar[] = {"nothing"};
            int dummyInt = 0;

            glutInit(&dummyInt, nullptr);
          }

          RiXGLBackend::~RiXGLBackend()
          {
            if ( m_windowId )
            {
              glutDestroyWindow( m_windowId );
              glutLeaveMainLoop();
              glutMainLoopEvent();
#if defined(DP_OS_WINDOWS)
              // As long as DPTRiXGL.bkd gets loaded and unloaded several times during test
              // execution this workaround is necessary to prevent freeglut from crashing.
              UnregisterClass( _T("FREEGLUT"), NULL);
#endif
            }
           m_context->makeNoncurrent();
          }

          dp::ui::RenderTargetSharedPtr RiXGLBackend::createDisplay( int width, int height, bool visible )
          {            
            glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA | GLUT_BORDERLESS );
            glutInitWindowSize( width, height );
            glutInitWindowPosition(0, 0);
            
            m_windowId = glutCreateWindow( "DPT" );

            glewInit();
            m_context = dp::gl::RenderContext::create( dp::gl::RenderContext::Attach() );

            dp::ui::RenderTargetSharedPtr displayTarget = createContextedRenderTarget<dp::gl::RenderTargetFB>(m_context);
            dp::util::shared_cast<dp::gl::RenderTargetFB>(displayTarget)->setSwapBuffersEnabled(true);
            dp::util::shared_cast<dp::gl::RenderTargetFB>(displayTarget)->setClearMask( dp::gl::TBM_COLOR_BUFFER | dp::gl::TBM_DEPTH_BUFFER );

            m_context->makeCurrent();

            DP_ASSERT( dynamic_cast<dp::rix::gl::RiXGL*>( getRenderer() ) );

            static_cast<dp::rix::gl::RiXGL*>( getRenderer() )->registerContext();

            return displayTarget;
          }

          dp::ui::RenderTargetSharedPtr RiXGLBackend::createAuxiliaryRenderTarget(int width, int height)
          {
            DP_ASSERT(!!m_context);
            dp::ui::RenderTargetSharedPtr fbo = dp::gl::RenderTargetFBO::create(m_context/*, width, height*/);
            fbo->setSize(width, height);
            return fbo;
          }

          void RiXGLBackend::finish()
          {
            glFinish();
          }

        } // namespace framework
      } // namespace gl
    } // namespace RiX
  } // namespace util
} // namespace dp
