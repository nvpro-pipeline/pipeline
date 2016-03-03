// Copyright (c) 2009-2016, NVIDIA CORPORATION. All rights reserved.
// Copyright NVIDIA Corporation 2009-2013
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


#include <dp/sg/ui/qt5/SceniXQGLSceneRendererWidget.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/gl/RenderTarget.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace qt5
      {

        SceniXQGLSceneRendererWidget::SceniXQGLSceneRendererWidget( QWidget *parent, const dp::gl::RenderContextFormat &format, SceniXQGLWidget *shareWidget )
        : SceniXQGLWidget( parent, format, shareWidget )
        , m_continuousUpdate( false )
        , m_timerID( -1 )
        {
          //
          // MMM - can we remove this stuf??
          //
          //if ( m_appTraverser->isSkinningProcessorSupported( AppTraverser::SKINNING_PROCESSOR_CUDA ) )
          //{
          //  m_appTraverser->setSkinningProcessor( AppTraverser::SKINNING_PROCESSOR_CUDA );
          //}
          //else if ( m_appTraverser->isSkinningProcessorSupported( AppTraverser::SKINNING_PROCESSOR_GPUG80GL ) )
          //{
          //  m_appTraverser->setSkinningProcessor( AppTraverser::SKINNING_PROCESSOR_GPUG80GL );
          //}

          m_todTimer.start();
          m_lastTime = m_todTimer.getTime();
        }

        SceniXQGLSceneRendererWidget::~SceniXQGLSceneRendererWidget()
        {
          if( m_timerID != -1 )
          {
            killTimer( m_timerID );
            m_timerID = -1;
          }

          getRenderContext()->makeCurrent();
        }

        void SceniXQGLSceneRendererWidget::initializeGL()
        {
          glewInit();

          glEnable( GL_DEPTH_TEST );

          std::static_pointer_cast<dp::gl::RenderTarget>(getRenderTarget())->setClearMask(dp::gl::TBM_COLOR_BUFFER | dp::gl::TBM_DEPTH_BUFFER);
          if ( m_manipulator )
          {
            m_manipulator->setRenderTarget( getRenderTarget() );
          }
        }

        void SceniXQGLSceneRendererWidget::paintGL()
        {
          if (m_renderer && m_viewState)
          {
            dp::sg::core::SceneSharedPtr sceneHandle = m_viewState->getScene();
            if (sceneHandle)
            {
              m_renderer->render( m_viewState, getRenderTarget() );
            }
          }
        }

        void SceniXQGLSceneRendererWidget::onTimer( float dt )
        {
          if ( m_manipulator )
          {
            m_manipulator->updateFrame( dt );
          }

          triggerRepaint();
        }

        float SceniXQGLSceneRendererWidget::getElapsedTime()
        {
          double curTime = m_todTimer.getTime();
          double diffTime = curTime - m_lastTime;

          m_lastTime = curTime;

          // gettime returns time in ms.  convert to s.
          diffTime /= 1000.0;

          return static_cast<float>( diffTime );
        }

        void SceniXQGLSceneRendererWidget::timerEvent( QTimerEvent * event )
        {
          onTimer( getElapsedTime() );
        }

        void SceniXQGLSceneRendererWidget::triggerRepaint()
        {
          update();
        }

        void SceniXQGLSceneRendererWidget::setContinuousUpdate( bool onOff )
        {
          if( onOff != m_continuousUpdate )
          {
            m_continuousUpdate = onOff;
            if( onOff )
            {
              // arg of 0 means signal every time there are no more window events to process
              m_timerID = startTimer(0);
            }
            else
            {
              DP_ASSERT( m_timerID != -1 );

              killTimer( m_timerID );
              m_timerID = -1;
            }
          }
        }

        void SceniXQGLSceneRendererWidget::hidNotify( dp::util::PropertyId property )
        {
          SceniXQGLWidget::hidNotify( property );
          if ( m_manipulator && m_manipulatorAutoRefresh )
          {
            if ( m_manipulator->updateFrame( getElapsedTime() ) )
            {
              update();
            }
          }
        }

        void SceniXQGLSceneRendererWidget::onManipulatorChanged( Manipulator *manipulator )
        {
          if ( manipulator )
          {
            manipulator->setRenderTarget( getRenderTarget() );
          }
        }

      } // namespace qt5
    } // namespace ui
  } // namespace sg
} // namespace dp
