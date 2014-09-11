// Copyright NVIDIA Corporation 2010-2011
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


#include <dp/sg/ui/SceneRenderer.h>
#include <dp/sg/core/FrustumCamera.h>

using namespace dp::math;
using namespace dp::sg::core;

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      class FrustumStereoViewStateProvider : public SceneRenderer::StereoViewStateProvider
      {
      public:
        static dp::util::SmartPtr<FrustumStereoViewStateProvider> create();
      protected:
        FrustumStereoViewStateProvider();

        ViewStateSharedPtr calculateViewState( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTarget::StereoTarget eye );

        ViewStateWeakPtr   m_lastViewState;
        ViewStateSharedPtr m_viewStateLeft;
        ViewStateSharedPtr m_viewStateRight;
      };

      FrustumStereoViewStateProvider::FrustumStereoViewStateProvider()
        : m_lastViewState(0) // weakptr is not an object yet
      {
      }

      dp::util::SmartPtr<FrustumStereoViewStateProvider> FrustumStereoViewStateProvider::create()
      {
        return new FrustumStereoViewStateProvider();
      }

      dp::sg::ui::ViewStateSharedPtr FrustumStereoViewStateProvider::calculateViewState( dp::sg::ui::ViewStateSharedPtr const& viewStatePtr, dp::ui::RenderTarget::StereoTarget eye )
      {
        DP_ASSERT( viewStatePtr );

        ViewStateSharedPtr stereoViewState = ViewState::create();
        // make a flat copy of the original viewstate
        *stereoViewState = *viewStatePtr;
        {
          // and assign a cloned camera
          if ( stereoViewState->getCamera() && stereoViewState->getCamera().isPtrTo<FrustumCamera>() )
          {
            float direction = (eye == dp::ui::RenderTarget::LEFT ) ? -1.0f : 1.0f;
            float eyeDistance = stereoViewState->getStereoEyeDistance();
            //  do the left pass first
            FrustumCameraSharedPtr const& clonedCamera = stereoViewState->getCamera().clone().staticCast<FrustumCamera>();
            {
              Vec2f windowOffset = clonedCamera->getWindowOffset();
              clonedCamera->setWindowOffset( Vec2f( windowOffset[0] + 0.5f * eyeDistance * (-direction), windowOffset[1] ) );
              clonedCamera->move( Vec3f( 0.5f * eyeDistance * direction, 0.0f, 0.0f ) );

              stereoViewState->setCamera( clonedCamera );
            }
          }
        }
        return stereoViewState;
      }

      DEFINE_STATIC_PROPERTY( SceneRenderer, PreserveTexturesAfterUpload );
      DEFINE_STATIC_PROPERTY( SceneRenderer, TraversalMaskOverride );
  
      BEGIN_REFLECTION_INFO( SceneRenderer )
        DERIVE_STATIC_PROPERTIES( SceneRenderer, Renderer );

        INIT_STATIC_PROPERTY_RW_BOOL( SceneRenderer, PreserveTexturesAfterUpload,   bool, SEMANTIC_VALUE, value, value );
        INIT_STATIC_PROPERTY_RW     ( SceneRenderer, TraversalMaskOverride, unsigned int, SEMANTIC_VALUE, value, value );
      END_REFLECTION_INFO

      SceneRenderer::SceneRenderer( const dp::ui::SmartRenderTarget &renderTarget )
        : Renderer( renderTarget )
        , m_preserveTexturesAfterUpload( true )
        , m_traversalMaskOverride( 0 )
      {
      }

      void SceneRenderer::setViewState( const ViewStateSharedPtr & viewState )
      {
        m_viewState = viewState;
        if( m_viewState )
        {
          if( m_viewState->getRendererOptions().getWeakPtr() != m_rendererOptions )  
          {
            // update renderer options
            m_rendererOptions = m_viewState->getRendererOptions().getWeakPtr();
            addRendererOptions( m_viewState->getRendererOptions() );
          }
        }
      }

      const ViewStateSharedPtr & SceneRenderer::getViewState( ) const
      {
        return m_viewState;
      }

      void SceneRenderer::render( const ViewStateSharedPtr &viewState, const dp::ui::SmartRenderTarget &renderTarget, dp::ui::RenderTarget::StereoTarget stereoTarget )
      {
        dp::ui::SmartRenderTarget curRenderTarget = renderTarget ? renderTarget : getRenderTarget();

        DP_ASSERT( viewState );
        DP_ASSERT( curRenderTarget );

        {
          // FIXME for compability now. Later viewstate is the only parameter
          if ( viewState->getCamera() && viewState->getCamera().isPtrTo<FrustumCamera>() )
          {
            viewState->getCamera().staticCast<FrustumCamera>()->setAspectRatio( curRenderTarget->getAspectRatio() );
          }
        }

        if ( !m_stereoViewStateProvider )
        {
          m_stereoViewStateProvider = FrustumStereoViewStateProvider::create();
        }

        beginRendering( viewState, curRenderTarget );
        if ( curRenderTarget->isStereoEnabled() )
        {
          ViewStateSharedPtr stereoViewState;

          if ( stereoTarget == dp::ui::RenderTarget::LEFT || stereoTarget == dp::ui::RenderTarget::LEFT_AND_RIGHT )
          {
            curRenderTarget->setStereoTarget( dp::ui::RenderTarget::LEFT );
            stereoViewState = m_stereoViewStateProvider->getViewState( viewState, dp::ui::RenderTarget::LEFT );
            doRender( stereoViewState, curRenderTarget );
          }

          if ( stereoTarget == dp::ui::RenderTarget::RIGHT || stereoTarget == dp::ui::RenderTarget::LEFT_AND_RIGHT )
          {
            curRenderTarget->setStereoTarget( dp::ui::RenderTarget::RIGHT );
            stereoViewState = m_stereoViewStateProvider->getViewState( viewState, dp::ui::RenderTarget::RIGHT );
            doRender( stereoViewState, curRenderTarget );
          }
        }
        else
        {
          DP_ASSERT( stereoTarget == dp::ui::RenderTarget::LEFT  || stereoTarget == dp::ui::RenderTarget::LEFT_AND_RIGHT );
          doRender( viewState, curRenderTarget );
        }
        endRendering( viewState, curRenderTarget );
      }

      void SceneRenderer::doRender( const dp::ui::SmartRenderTarget &renderTarget )
      {
        DP_ASSERT( m_viewState );
        DP_ASSERT( renderTarget );

        render( m_viewState, renderTarget );
      }

      void SceneRenderer::beginRendering( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::SmartRenderTarget const& renderTarget )
      {
        DP_ASSERT( viewState );
        if ( viewState )
        {
          DP_ASSERT( viewState->getRendererOptions() );
          if( viewState->getRendererOptions() && viewState->getRendererOptions().getWeakPtr() != m_rendererOptions )  
          {
            // update renderer options
            m_rendererOptions = viewState->getRendererOptions().getWeakPtr();
            addRendererOptions( viewState->getRendererOptions() );
          }
        }
      }

      void SceneRenderer::endRendering( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::SmartRenderTarget const& renderTarget )
      {
      }

      ViewStateSharedPtr SceneRenderer::StereoViewStateProvider::getViewState( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTarget::StereoTarget eye )
      {
        return calculateViewState( viewState, eye );
      }

      void SceneRenderer::addRendererOptions( const dp::sg::ui::RendererOptionsSharedPtr &rendererOptions )
      {
      }

      void SceneRenderer::setEnvironmentSampler( const dp::sg::core::SamplerSharedPtr & sampler )
      {
        if ( m_environmentSampler != sampler )
        {
          m_environmentSampler = sampler;
          onEnvironmentSamplerChanged();
        }
      }

      void SceneRenderer::onEnvironmentSamplerChanged()
      {
      }

    } // namespace ui
  } // namespace sg
} // namespace dp
