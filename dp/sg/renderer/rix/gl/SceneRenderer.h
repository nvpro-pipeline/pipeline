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

#include <dp/sg/renderer/rix/gl/Config.h>
#include <dp/sg/renderer/rix/gl/TransparencyManager.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceManager.h>
#include <dp/gl/RenderTarget.h>
#include <dp/sg/xbar/SceneTree.h>
#include <dp/sg/xbar/DrawableManager.h>
#include <dp/culling/Manager.h>

#include <dp/sg/ui/SceneRenderer.h>

#include <dp/util/Reflection.h>
#include <dp/util/SmartPtr.h>

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

          class ResourceManager;
          typedef dp::util::SmartPtr<ResourceManager> SmartResourceManager;

          /** \brief SceneRenderer is an OpenGL 3.0 based renderer. The OpenGL context used during the first render call
                    must not be changed for successive calls. It is necessary to remove all references to this object before
                    the attached OpenGL context gets destroyed. Deleting this object after destroying the corresponding OpenGL
                    context will result in a crash during resource cleanup. OpenGL resources like VBOs, display lists and 
                    textures will be shared between multiple SceneRenderer objects if the corresponding OpenGL contexts
                    are being shared.
          **/
          class SceneRenderer : public dp::sg::ui::SceneRenderer
          {    
          protected:
            DP_SG_RDR_RIX_GL_API SceneRenderer( const dp::gl::SharedRenderTarget &renderTarget = dp::gl::SharedRenderTarget() );
            DP_SG_RDR_RIX_GL_API virtual ~SceneRenderer();

          public:
            /** \brief Create an instance of SceneRenderer
                \param renderTarget Default RenderTarget to use.
                \return An instance of a SceneRenderer object.
            **/
            static DP_SG_RDR_RIX_GL_API dp::util::SmartPtr<SceneRenderer> create( const char *renderEngine = 0,
                                                                                  dp::fx::Manager shaderManagerType = dp::fx::MANAGER_SHADERBUFFER,
                                                                                  dp::culling::Mode cullingMode = dp::culling::MODE_AUTO,
                                                                                  TransparencyMode transparencyMode = TM_ORDER_INDEPENDENT_CLOSEST_LIST,
                                                                                  const dp::gl::SharedRenderTarget &renderTarget = dp::gl::SharedRenderTarget() );

            // HACK HACK HACK
            void setDepthPass( bool depthPass ) { m_depthPass = depthPass; }
            bool getDepthPass( ) const { return m_depthPass; }

            virtual void setRenderEngine( std::string const& renderEngine ) = 0;
            virtual std::string const& getRenderEngine() const = 0;

            virtual dp::sg::renderer::rix::gl::TransparencyMode getTransparencyMode() const = 0;
            virtual void setTransparencyMode( dp::sg::renderer::rix::gl::TransparencyMode mode ) = 0;
            virtual dp::sg::renderer::rix::gl::SmartTransparencyManager const & getTransparencyManager() const = 0;

          protected:
            /** \brief Delete all primitive caches. Call this function only if an OpenGL context is active since resources need
            to be deleted.
            **/
            DP_SG_RDR_RIX_GL_API virtual void deletePrimitiveCaches() = 0;

            DP_SG_RDR_RIX_GL_API virtual dp::sg::xbar::DrawableManager *createDrawableManager( const SmartResourceManager &resourceManager ) const = 0;
            DP_SG_RDR_RIX_GL_API virtual dp::sg::xbar::DrawableManager* getDrawableManager() const = 0;

          protected:
            bool m_depthPass;
          };

          typedef dp::util::SmartPtr<SceneRenderer> SmartSceneRenderer;

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
