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


#include <dp/sg/ui/glut/SceneRendererWidget.h>
#include <GL/freeglut.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace glut
      {

        BEGIN_REFLECTION_INFO( SceneRendererWidget )
          DERIVE_STATIC_PROPERTIES( SceneRendererWidget, Widget );
        END_REFLECTION_INFO

        SceneRendererWidget::SceneRendererWidget()
        {
        }

        SceneRendererWidget::~SceneRendererWidget()
        {
        }

        void SceneRendererWidget::paint()
        {
          dp::gl::RenderContextStack s;
          s.push(getRenderTarget()->getRenderContext());
          if ( m_renderer && m_viewState )
          {
            dp::sg::core::SceneSharedPtr const & sceneHandle = m_viewState->getScene();
            if (sceneHandle)
            {
              m_renderer->render( m_viewState, getRenderTarget() );
            }
          }
          glutSwapBuffers();

          Widget::paint();
          s.pop();
        }

        void SceneRendererWidget::cleanup()
        {
          m_renderer.reset();
          Widget::cleanup();
        }

        void SceneRendererWidget::triggerRepaint()
        {
          doTriggerRepaint();
        }

        void SceneRendererWidget::onHIDEvent( dp::util::PropertyId property )
        {
          if ( m_manipulator && m_manipulatorAutoRefresh )
          {
            if ( m_manipulator->updateFrame( float(m_updateTimer.getTime()) ) )
            {
              triggerRepaint();
            }
            m_updateTimer.restart();
          }
        }

        void SceneRendererWidget::onManipulatorChanged( Manipulator *manipulator )
        {
          if ( manipulator )
          {
            manipulator->setRenderTarget( getRenderTarget() );
          }
        }

      } // namespace glut
    } // namespace ui
  } // namespace sg
} // namespace dp
