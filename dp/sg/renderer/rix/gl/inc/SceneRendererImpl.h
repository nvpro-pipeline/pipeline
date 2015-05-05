// Copyright NVIDIA Corporation 2010-2015
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

#include <dp/sg/renderer/rix/gl/SceneRenderer.h>
#include <dp/sg/renderer/rix/gl/FSQRenderer.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceManager.h>
#include <dp/sg/xbar/SceneTree.h>
#include <dp/sg/xbar/DrawableManager.h>

#include <dp/sg/ui/SceneRenderer.h>
#include <dp/sg/renderer/rix/gl/TransparencyManager.h>
#include <dp/gl/RenderTarget.h>
#include <dp/util/Reflection.h>
#include <dp/util/DynamicLibrary.h>

namespace dp
{
  namespace sg
  {
    namespace renderer
    {
      namespace rix
      {
        namespace gl
        {

          /** \brief SceneRendererGL is an OpenGL 3.0 based renderer. The OpenGL context used during the first render call
                    must not be changed for successive calls. It is necessary to remove all references to this object before
                    the attached OpenGL context gets destroyed. Deleting this object after destroying the corresponding OpenGL
                    context will result in a crash during resource cleanup. OpenGL resources like VBOs, display lists and 
                    textures will be shared between multiple SceneRendererImpl objects if the corresponding OpenGL contexts
                    are being shared.
          **/
          class SceneRendererImpl : public SceneRenderer
          {
          protected:
            SceneRendererImpl( const char *renderEngine
                             , dp::fx::Manager shaderManagerType
                             , dp::culling::Mode cullingMode
                             , TransparencyMode transparencyMode
                             , const dp::gl::RenderTargetSharedPtr &renderTarget );

          public:
            /** \brief Create an instance of SceneRendererImpl
                \param renderTarget Default RenderTarget to use.
                \return An instance of a SceneRendererImpl object.
            **/
            static SceneRendererSharedPtr create( const char *renderEngine
                                                , dp::fx::Manager shaderManagerType
                                                , dp::culling::Mode culling
                                                , TransparencyMode transparencyMode
                                                , const dp::gl::RenderTargetSharedPtr &renderTarget );
            virtual ~SceneRendererImpl();

            /** \brief Add all supported options to the RendererOptions container.
                \param rendererOptions A container for RendererOptions 
            **/
            virtual void addRendererOptions( const dp::sg::ui::RendererOptionsSharedPtr &rendererOptions );

            // HACK HACK HACK
            void setDepthPass( bool depthPass );
            bool getDepthPass( ) const { return m_depthPass; }

            virtual std::map<dp::fx::Domain,std::string> getShaderSources( const dp::sg::core::GeoNodeSharedPtr & geoNode, bool depthPass ) const;
            virtual dp::sg::renderer::rix::gl::TransparencyManagerSharedPtr const & getTransparencyManager() const;

          protected:
            /** \brief Delete all primitive caches. Call this function only if an OpenGL context is active since resources need
            to be deleted.
            **/
            void deletePrimitiveCaches();

            virtual void beginRendering( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTargetSharedPtr const& renderTarget );
            virtual void endRendering( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTargetSharedPtr const& renderTarget );

            virtual void doRender( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTargetSharedPtr const& renderTarget );

            virtual void doRenderDrawables( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::gl::RenderTargetSharedPtr const& renderTarget );
            virtual dp::sg::xbar::DrawableManager *createDrawableManager( const ResourceManagerSharedPtr &resourceManager ) const;
            dp::sg::xbar::DrawableManager* getDrawableManager() const { return m_drawableManager; }

            virtual void onEnvironmentRenderingEnabledChanged();
            virtual void onEnvironmentSamplerChanged();

            virtual void setCullingEnabled( bool enabled );
            virtual bool isCullingEnabled() const;

            virtual void setCullingMode( dp::culling::Mode mode );
            virtual dp::culling::Mode getCullingMode( ) const;

            virtual void setShaderManager( dp::fx::Manager shaderManager );
            virtual dp::fx::Manager getShaderManager() const;

            virtual void setRenderEngine( std::string const& renderEngine );
            virtual std::string const& getRenderEngine() const;

            virtual dp::sg::renderer::rix::gl::TransparencyMode getTransparencyMode() const;
            virtual void setTransparencyMode( dp::sg::renderer::rix::gl::TransparencyMode mode );

          private:
            void doRenderEnvironmentMap( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::gl::RenderTargetSharedPtr const& renderTarget );

          protected:
            bool initializeRenderer();
            void shutdownRenderer();

            dp::util::DynamicLibrarySharedPtr        m_rix; // keep m_rix always before m_renderer to ensure proper destruction order
            std::unique_ptr<dp::rix::core::Renderer> m_renderer;
            dp::sg::xbar::SceneTreeSharedPtr         m_sceneTree;
            dp::sg::xbar::DrawableManager*           m_drawableManager;
            ResourceManagerSharedPtr                 m_resourceManager;
            dp::fx::Manager                          m_shaderManager;
            std::string                              m_renderEngineOptions;

            bool                                     m_contextRegistered;
            bool                                     m_rendererInitialized;

            dp::gl::RenderContextSharedPtr           m_userRenderContext;     // RenderContext provided by the user in the first render call
            dp::culling::Mode                        m_cullingMode;

            bool                                     m_cullingEnabled;

            dp::math::Vec2ui                         m_viewportSize;

          private:
            dp::sg::renderer::rix::gl::FSQRendererSharedPtr         m_environmentRenderer;
            dp::sg::renderer::rix::gl::TransparencyManagerSharedPtr m_transparencyManager;
          };

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
