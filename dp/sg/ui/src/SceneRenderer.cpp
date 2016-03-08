// Copyright (c) 2010-2016, NVIDIA CORPORATION. All rights reserved.
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
      DEFINE_PTR_TYPES( FrustumStereoViewStateProvider );

      class FrustumStereoViewStateProvider : public SceneRenderer::StereoViewStateProvider
      {
      public:
        static FrustumStereoViewStateProviderSharedPtr create();
      protected:
        FrustumStereoViewStateProvider();

        ViewStateSharedPtr calculateViewState( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTarget::StereoTarget eye );

        ViewStateWeakPtr   m_lastViewState;
        ViewStateSharedPtr m_viewStateLeft;
        ViewStateSharedPtr m_viewStateRight;
      };

      FrustumStereoViewStateProvider::FrustumStereoViewStateProvider()
      {
      }

      FrustumStereoViewStateProviderSharedPtr FrustumStereoViewStateProvider::create()
      {
        return( std::shared_ptr<FrustumStereoViewStateProvider>( new FrustumStereoViewStateProvider() ) );
      }

      dp::sg::ui::ViewStateSharedPtr FrustumStereoViewStateProvider::calculateViewState( dp::sg::ui::ViewStateSharedPtr const& viewStatePtr, dp::ui::RenderTarget::StereoTarget eye )
      {
        DP_ASSERT( viewStatePtr );

        ViewStateSharedPtr stereoViewState = ViewState::create();
        // make a flat copy of the original viewstate
        *stereoViewState = *viewStatePtr;
        {
          // and assign a cloned camera
          if ( stereoViewState->getCamera() && std::dynamic_pointer_cast<FrustumCamera>(stereoViewState->getCamera()) )
          {
            float direction = (eye == dp::ui::RenderTarget::StereoTarget::LEFT ) ? -1.0f : 1.0f;
            float eyeDistance = stereoViewState->getStereoEyeDistance();
            //  do the left pass first
            FrustumCameraSharedPtr const& clonedCamera = std::static_pointer_cast<FrustumCamera>(stereoViewState->getCamera()->clone());
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

        INIT_STATIC_PROPERTY_RW_BOOL( SceneRenderer, PreserveTexturesAfterUpload,   bool, Semantic::VALUE, value, value );
        INIT_STATIC_PROPERTY_RW     ( SceneRenderer, TraversalMaskOverride, unsigned int, Semantic::VALUE, value, value );
      END_REFLECTION_INFO

      SceneRenderer::SceneRenderer( const dp::ui::RenderTargetSharedPtr &renderTarget )
        : Renderer( renderTarget )
        , m_environmentRenderingEnabled( false )
        , m_preserveTexturesAfterUpload( true )
        , m_traversalMaskOverride( 0 )
      {
      }

      void SceneRenderer::setViewState( const ViewStateSharedPtr & viewState )
      {
        m_viewState = viewState;
        if( m_viewState )
        {
          if( m_viewState->getRendererOptions() != m_rendererOptions.lock() )
          {
            // update renderer options
            m_rendererOptions = m_viewState->getRendererOptions();
            addRendererOptions( m_viewState->getRendererOptions() );
          }
        }
      }

      const ViewStateSharedPtr & SceneRenderer::getViewState( ) const
      {
        return m_viewState;
      }

      void SceneRenderer::render( const ViewStateSharedPtr &viewState, const dp::ui::RenderTargetSharedPtr &renderTarget, dp::ui::RenderTarget::StereoTarget stereoTarget )
      {
        dp::ui::RenderTargetSharedPtr curRenderTarget = renderTarget ? renderTarget : getRenderTarget();

        DP_ASSERT( viewState );
        DP_ASSERT( curRenderTarget );

        {
          // FIXME for compability now. Later viewstate is the only parameter
          if ( viewState->getCamera() && std::dynamic_pointer_cast<FrustumCamera>(viewState->getCamera()) )
          {
            std::static_pointer_cast<FrustumCamera>(viewState->getCamera())->setAspectRatio( curRenderTarget->getAspectRatio() );
          }
        }

        if ( !m_stereoViewStateProvider )
        {
          m_stereoViewStateProvider = FrustumStereoViewStateProvider::create();
        }

        if ( curRenderTarget->isStereoEnabled() )
        {
          ViewStateSharedPtr viewStateLeft = m_stereoViewStateProvider->getViewState(viewState, dp::ui::RenderTarget::StereoTarget::LEFT);
          ViewStateSharedPtr viewStateRight = m_stereoViewStateProvider->getViewState(viewState, dp::ui::RenderTarget::StereoTarget::RIGHT);
          if (renderTarget->isMulticastEnabled() && stereoTarget == dp::ui::RenderTarget::StereoTarget::LEFT_AND_RIGHT)
          {
            curRenderTarget->setStereoTarget(dp::ui::RenderTarget::StereoTarget::LEFT_AND_RIGHT);
            beginRendering(viewState, curRenderTarget);
            doRender(viewState, curRenderTarget, { viewStateLeft->getCamera(), viewStateRight->getCamera() });
            endRendering(viewState, curRenderTarget);
          }
          else
          {
            if (stereoTarget == dp::ui::RenderTarget::StereoTarget::LEFT || stereoTarget == dp::ui::RenderTarget::StereoTarget::LEFT_AND_RIGHT)
            {
              curRenderTarget->setStereoTarget(dp::ui::RenderTarget::StereoTarget::LEFT);
              beginRendering(viewState, curRenderTarget);
              doRender(viewStateLeft, curRenderTarget, { viewStateLeft->getCamera() });
              endRendering(viewState, curRenderTarget);
            }

            if (stereoTarget == dp::ui::RenderTarget::StereoTarget::RIGHT || stereoTarget == dp::ui::RenderTarget::StereoTarget::LEFT_AND_RIGHT)
            {
              curRenderTarget->setStereoTarget(dp::ui::RenderTarget::StereoTarget::RIGHT);
              beginRendering(viewState, curRenderTarget);
              doRender(viewStateRight, curRenderTarget, { viewStateRight->getCamera() });
              endRendering(viewState, curRenderTarget);
            }
          }

        }
        else
        {
          DP_ASSERT( stereoTarget == dp::ui::RenderTarget::StereoTarget::LEFT  || stereoTarget == dp::ui::RenderTarget::StereoTarget::LEFT_AND_RIGHT );
          curRenderTarget->setStereoTarget(dp::ui::RenderTarget::StereoTarget::LEFT);
          beginRendering(viewState, curRenderTarget);
          doRender( viewState, curRenderTarget, {viewState->getCamera()} );
          endRendering(viewState, curRenderTarget);
        }
      }

      void SceneRenderer::doRender( const dp::ui::RenderTargetSharedPtr &renderTarget )
      {
        DP_ASSERT( m_viewState );
        DP_ASSERT( renderTarget );

        render( m_viewState, renderTarget );
      }

      void SceneRenderer::beginRendering( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTargetSharedPtr const& renderTarget )
      {
        DP_ASSERT( viewState );
        if ( viewState )
        {
          DP_ASSERT( viewState->getRendererOptions() );
          if( viewState->getRendererOptions() && viewState->getRendererOptions() != m_rendererOptions.lock() )
          {
            // update renderer options
            m_rendererOptions = viewState->getRendererOptions();
            addRendererOptions( viewState->getRendererOptions() );
          }
        }
      }

      void SceneRenderer::endRendering( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTargetSharedPtr const& renderTarget )
      {
      }

      ViewStateSharedPtr SceneRenderer::StereoViewStateProvider::getViewState( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTarget::StereoTarget eye )
      {
        return calculateViewState( viewState, eye );
      }

      void SceneRenderer::addRendererOptions( const dp::sg::ui::RendererOptionsSharedPtr &rendererOptions )
      {
      }

      void SceneRenderer::onEnvironmentRenderingEnabledChanged()
      {
      }

      void SceneRenderer::setEnvironmentRenderingEnabled( bool enabled )
      {
        if ( m_environmentRenderingEnabled != enabled )
        {
          m_environmentRenderingEnabled = enabled;
          onEnvironmentRenderingEnabledChanged();
        }
      }

      bool SceneRenderer::getEnvironmentRenderingEnabled() const
      {
        return( m_environmentRenderingEnabled );
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
