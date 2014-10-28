// Copyright NVIDIA Corporation 2011
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

#include <dp/gl/Program.h>
#include <dp/gl/Texture.h>
#include <dp/sg/renderer/rix/gl/inc/ShaderManager.h>
#include <dp/sg/renderer/rix/gl/TransparencyManager.h>
#include <GL/glew.h>

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
          SMART_TYPES( TransparencyManagerOITAll );

          class TransparencyManagerOITAll : public TransparencyManager
          {
            public:
              DP_SG_RDR_RIX_GL_API static SmartTransparencyManagerOITAll create( dp::math::Vec2ui const & size );
              DP_SG_RDR_RIX_GL_API virtual ~TransparencyManagerOITAll();

            public:
              DP_SG_RDR_RIX_GL_API virtual void addFragmentCodeSnippets( bool transparent, bool depth, std::vector<std::string> & snippets );
              DP_SG_RDR_RIX_GL_API virtual void beginTransparentPass( dp::rix::core::Renderer * renderer );
              DP_SG_RDR_RIX_GL_API virtual void resolveDepthPass();
              DP_SG_RDR_RIX_GL_API virtual bool endTransparentPass();
              DP_SG_RDR_RIX_GL_API virtual void initializeParameterContainer( dp::rix::core::Renderer * renderer, dp::math::Vec2ui const & viewportSize );
              DP_SG_RDR_RIX_GL_API virtual void useParameterContainer( dp::rix::core::Renderer * renderer, dp::rix::core::RenderGroupSharedHandle const & transparentRenderGroup );
              DP_SG_RDR_RIX_GL_API virtual bool needsSortedRendering() const;
              DP_SG_RDR_RIX_GL_API virtual bool supportsDepthPass() const;

            protected:
              DP_SG_RDR_RIX_GL_API TransparencyManagerOITAll( dp::math::Vec2ui const & size );

              DP_SG_RDR_RIX_GL_API virtual void viewportSizeChanged();

            private:
              void drawQuad();

            private:
              dp::rix::core::ContainerSharedHandle              m_parameterContainer;
              dp::rix::core::ContainerDescriptorSharedHandle    m_parameterContainerDescriptor;
              dp::rix::core::TextureSharedHandle                m_perFragmentCountTexture;
              dp::rix::core::TextureSharedHandle                m_perFragmentOffsetTexture;
              dp::rix::core::TextureSharedHandle                m_samplesTexture;

              dp::gl::SharedProgram       m_clearProgram;
              dp::gl::SharedTexture1D     m_counterAccuTexture;
              dp::gl::SharedBuffer        m_fullScreenQuad;
              bool                        m_initializedBuffers;
              bool                        m_initializedHandles;
              dp::gl::SharedTexture2D     m_perFragmentCountTextureGL;
              dp::gl::SharedTexture2D     m_perFragmentOffsetTextureGL;
              dp::gl::SharedProgram       m_resolveCountersProgram;
              dp::gl::SharedProgram       m_resolveSamplesProgram;
              dp::gl::SharedTextureBuffer m_samplesTextureGL;
              GLuint                      m_samplesPassedQuery;
          };

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
