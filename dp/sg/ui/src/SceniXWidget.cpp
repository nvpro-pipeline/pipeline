// Copyright NVIDIA Corporation 2010
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


#include <dp/sg/ui/SceniXWidget.h>
#include <dp/sg/ui/RendererOptions.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {

      SceniXSceneRendererWidget::SceniXSceneRendererWidget()
        : m_manipulatorAutoRefresh( true )
        , m_manipulator( 0 )
      {
      }

      SceniXSceneRendererWidget::~SceniXSceneRendererWidget()
      {
        if ( m_manipulator )
        {
          m_manipulator->setViewState( dp::sg::ui::ViewStateSharedPtr::null );
          m_manipulator->setRenderTarget( dp::ui::SmartRenderTarget::null );
        }
        if ( m_renderer )
        {
          m_renderer->setRenderTarget( dp::ui::SmartRenderTarget::null );
        }
      }

      void SceniXSceneRendererWidget::setViewState( dp::sg::ui::ViewStateSharedPtr const& viewState )
      {
        if ( viewState != m_viewState )
        {
          m_viewState = viewState;
          onViewStateChanged( m_viewState );
          triggerRepaint();
        }
      }

      dp::sg::ui::ViewStateSharedPtr const& SceniXSceneRendererWidget::getViewState() const
      {
        return m_viewState;
      }

      void SceniXSceneRendererWidget::setSceneRenderer( const dp::sg::ui::SmartSceneRenderer &sceneRenderer )
      {
        if ( m_renderer != sceneRenderer)
        {
          m_renderer = sceneRenderer;
          onSceneRendererChanged( m_renderer );
          triggerRepaint();
        }
      }

      dp::sg::ui::SmartSceneRenderer SceniXSceneRendererWidget::getSceneRenderer() const
      {
        return m_renderer;
      }

      void SceniXSceneRendererWidget::setManipulator( Manipulator *manipulator )
      {
        if ( m_manipulator != manipulator )
        {
          m_manipulator = manipulator;
          if ( m_manipulator )
          {
            m_manipulator->setViewState( m_viewState );
            m_manipulator->reset();
          }
          onManipulatorChanged( m_manipulator );
        }
      }

      Manipulator * SceniXSceneRendererWidget::getManipulator() const
      {
        return m_manipulator;
      }

      void SceniXSceneRendererWidget::onViewStateChanged( dp::sg::ui::ViewStateSharedPtr const& viewState )
      {
        if ( m_manipulator )
        {
          m_manipulator->setViewState( viewState );
        }
      }

    } // namespace ui
  } // namespace sg
} // namespace dp
