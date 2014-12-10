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


#pragma once

#include <dp/fx/EffectDefs.h>
#include <dp/rix/core/RiX.h>
#include <dp/sg/ui/Renderer.h>
#include <dp/culling/Manager.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      DEFINE_PTR_TYPES( SceneRenderer );

      /** \brief SceneRenderer is the base class for all Renderers which visualize a scene. On a stereo
                 dp::ui::RenderTarget stereo rendering is turned on automatically. SceniX supports a default implementation
                 of dp::sg::ui::SceneRenderer::StereoViewStateProvider which supports dp::sg::core::PerspectiveCamera. For other
                 cameras it is possible to specify the camera matrices for the left and right eye by implementing 
                 a custom dp::sg::ui::SceneRenderer::StereoViewStateProvider.
      **/
      class SceneRenderer : public dp::sg::ui::Renderer
      {
      public:
        /** \brief This class generates an dp::sg::ui::ViewState for the left and right eye based on a
                   monoscopic dp::sg::ui::ViewState. 
        **/
        DEFINE_PTR_TYPES( StereoViewStateProvider );
        class StereoViewStateProvider
        {
        public:
          /** \brief Get an dp::sg::ui::ViewState for the given eye. Implement calculateViewState for custom behaviour.
              \param viewState Monoscopic dp::sg::ui::ViewState used as base for the stereo dp::sg::ui::ViewState.
              \param eye dp::ui::RenderTarget::LEFT or dp::ui::RenderTarget::RIGHT for the left or right eye, respectively.
              \return dp::sg::ui::ViewState for the given eye.
          **/
          DP_SG_UI_API dp::sg::ui::ViewStateSharedPtr getViewState( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTarget::StereoTarget eye );

        protected:
          /** \brief Calculate an dp::sg::ui::ViewState for the given eye. Override this function to implement custom behaviour.
              \param viewState Monoscopic dp::sg::ui::ViewState used as base for the stereo dp::sg::ui::ViewState.
              \param eye dp::ui::RenderTarget::LEFT or dp::ui::RenderTarget::RIGHT for the left or right eye.
              \return dp::sg::ui::ViewState for the given eye.
          **/
          DP_SG_UI_API virtual dp::sg::ui::ViewStateSharedPtr calculateViewState( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTarget::StereoTarget eye ) = 0;
        };

        /*! \brief Set an dp::sg::ui::ViewState to be used with this SceneRenderer.
         *  \param viewState The dp::sg::ui::ViewState to be used.
         *  \note The dp::sg::ui::ViewState's TraversalMask will be used in conjunction with any TraversalMaskOverride to direct
         *  scene traversal and rendering. See setTraversalMaskOverride for more info.
         *  \sa getViewState, setTraversalMaskOverride */
        DP_SG_UI_API void setViewState( dp::sg::ui::ViewStateSharedPtr const& viewState );

        /** \brief Get the default dp::sg::ui::ViewState used by render calls
            \return Default dp::sg::ui::ViewState used by render calls
        **/
        DP_SG_UI_API dp::sg::ui::ViewStateSharedPtr const& getViewState( ) const;

        /*! \brief Set the TraversalMask Override to be used with this SceneRenderer.
         *  \param mask The mask to be used.
         *  \remarks This method provides a way to set the traversal mask override for this SceneRenderer.
         *  \note \li The dp::sg::ui::ViewState's TraversalMask is used in conjuction with the OverrideTraversalMask and every scene graph node's 
         *  TraversalMask to determine whether nodes (and therefore possibly their entire subgraph) are rendered.  
         *  The scene renderer's override traversal mask is OR'd with the node's traversal mask and that result is ANDed with 
         *  the dp::sg::ui::ViewState's traversal mask. If the result is nonzero then the node is traversed/rendered, otherwise it is ignored.  IE:
         *  If ( ( (SceneRenderer::TraversalMaskOverride | Object::TraversalMask) & dp::sg::ui::ViewState::TraversalMask ) != 0 ) is true the node is
         *  rendered.
         *  \li Setting the dp::sg::ui::ViewState's TraversalMask to 0 will cause no nodes to be traversed.  Setting the dp::sg::ui::ViewState's TraversalMask 
         *  to ~0 and the TraversalMaskOverride to ~0 will cause all nodes to be traversed regardless of the Object::TraversalMask.
         *  \li The default traversal mask override is 0 so that it does not affect traversal/rendering.
         *  \sa getTraversalMaskOverride, dp::sg::ui::ViewState::setTraversalMask, Object::setTraversalMask */
        DP_SG_UI_API void setTraversalMaskOverride( unsigned int mask );

        /*! \brief Get the current TraversalMask override
         *  \return mask The mask in use.
         *  \sa setTraversalMaskOverride */
        DP_SG_UI_API unsigned int getTraversalMaskOverride() const;

        /**\brief Set the StereoViewState provider which should be used for stereo ViewState calculation.
           \param viewStateProvider A ViewStateProvider instance with desired behaviour.
        **/
        DP_SG_UI_API void setStereoViewStateProvider( StereoViewStateProviderSharedPtr const& viewStateProvider );
        DP_SG_UI_API StereoViewStateProviderSharedPtr const& getStereoViewStateProvider() const;

        DP_SG_UI_API void setEnvironmentSampler( const dp::sg::core::SamplerSharedPtr & sampler );
        DP_SG_UI_API const dp::sg::core::SamplerSharedPtr & getEnvironmentSampler() const;

        DP_SG_UI_API virtual std::map<dp::fx::Domain,std::string> getShaderSources( const dp::sg::core::GeoNodeSharedPtr & geoNode, bool depthPass ) const = 0;

        /** \brief Render one frame with the given dp::sg::ui::ViewState and renderTarget. 
            \param viewState dp::sg::ui::ViewState to use to render the frame.
            \param renderTarget If renderTarget is valid it temporarily overrides the default set by Renderer::setRenderTarget.
        **/
        DP_SG_UI_API void render( dp::sg::ui::ViewStateSharedPtr const& viewState
                                , dp::ui::RenderTargetSharedPtr const& renderTarget = dp::ui::RenderTargetSharedPtr()
                                , dp::ui::RenderTarget::StereoTarget stereoTarget = dp::ui::RenderTarget::LEFT_AND_RIGHT );

        /** \brief Add all renderer options required by this renderer to the given dp::sg::ui::RendererOptions object. It is possible
        *          to pass the same options object to multiple SceneRenderers as long as there are no property name collisions.
            \param rendererOptions Object which holds renderer options.
        **/
        DP_SG_UI_API virtual void addRendererOptions( const dp::sg::ui::RendererOptionsSharedPtr &rendererOptions );

        DP_SG_UI_API virtual void setCullingEnabled( bool enabled ) = 0;
        DP_SG_UI_API virtual bool isCullingEnabled() const = 0;

        DP_SG_UI_API virtual void setCullingMode( dp::culling::Mode mode ) = 0;
        DP_SG_UI_API virtual dp::culling::Mode getCullingMode( ) const = 0;

        DP_SG_UI_API virtual void setShaderManager( dp::fx::Manager shaderManager ) = 0;
        DP_SG_UI_API virtual dp::fx::Manager getShaderManager() const = 0;


        REFLECTION_INFO_API( DP_SG_UI_API, SceneRenderer );
        BEGIN_DECLARE_STATIC_PROPERTIES
          DP_SG_UI_API DECLARE_STATIC_PROPERTY( PreserveTexturesAfterUpload );
          DP_SG_UI_API DECLARE_STATIC_PROPERTY( TraversalMaskOverride );
        END_DECLARE_STATIC_PROPERTIES

        using Renderer::render;

      protected:
        /** \brief Constructor used by create **/
        DP_SG_UI_API SceneRenderer( const dp::ui::RenderTargetSharedPtr &renderTarget = dp::ui::RenderTargetSharedPtr() );

        // preserve interface should be set using Reflection
        DP_SG_UI_API bool isPreserveTexturesAfterUpload() const;
        DP_SG_UI_API void setPreserveTexturesAfterUpload( bool onOff );

        DP_SG_UI_API virtual void beginRendering( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTargetSharedPtr const& renderTarget );
        DP_SG_UI_API virtual void endRendering( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTargetSharedPtr const& renderTarget );

        DP_SG_UI_API virtual void doRender( const dp::ui::RenderTargetSharedPtr &renderTarget );

        /** \brief Interface for the actual rendering algorithm.
            \param viewState The dp::sg::ui::ViewState to use to render the frame.
            \param renderTarget The RenderTarget to use to render the frame.
        **/
        DP_SG_UI_API virtual void doRender( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTargetSharedPtr const& renderTarget ) = 0;

        DP_SG_UI_API virtual void onEnvironmentSamplerChanged();

      protected:
        dp::sg::core::SceneSharedPtr        m_scene;
        dp::sg::ui::ViewStateSharedPtr      m_viewState;
        bool                                m_preserveTexturesAfterUpload;
        dp::sg::ui::RendererOptionsWeakPtr  m_rendererOptions;

      private:
        StereoViewStateProviderSharedPtr  m_stereoViewStateProvider;
        unsigned int                      m_traversalMaskOverride;
        dp::sg::core::SamplerSharedPtr    m_environmentSampler;
      };

      inline bool SceneRenderer::isPreserveTexturesAfterUpload() const
      {
        return m_preserveTexturesAfterUpload;
      }

      inline void SceneRenderer::setPreserveTexturesAfterUpload( bool onOff )
      {
        if ( m_preserveTexturesAfterUpload != onOff )
        {
          m_preserveTexturesAfterUpload = onOff;
          notify( PropertyEvent( this, PID_PreserveTexturesAfterUpload ) );
        }
      }

      inline unsigned int SceneRenderer::getTraversalMaskOverride() const
      {
        return m_traversalMaskOverride;
      }

      inline void SceneRenderer::setTraversalMaskOverride( unsigned int mask )
      {
        if ( m_traversalMaskOverride != mask )
        {
          m_traversalMaskOverride = mask;
          notify( PropertyEvent( this, PID_TraversalMaskOverride ) );
        }
      }

      inline void SceneRenderer::setStereoViewStateProvider( SceneRenderer::StereoViewStateProviderSharedPtr const& viewStateProvider )
      {
        m_stereoViewStateProvider = viewStateProvider;
      }

      inline SceneRenderer::StereoViewStateProviderSharedPtr const& SceneRenderer::getStereoViewStateProvider( ) const
      {
        return m_stereoViewStateProvider;
      }

      inline const dp::sg::core::SamplerSharedPtr & SceneRenderer::getEnvironmentSampler() const
      {
        return( m_environmentSampler );
      }

    } // namespace ui
  } // namespace sg
} // namespace dp
