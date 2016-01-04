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

#include <dp/fx/ParameterSpec.h>
#include <dp/rix/core/RiX.h>
#include <dp/sg/renderer/rix/gl/Config.h>

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
          DEFINE_PTR_TYPES( TransparencyManager );

          class ShaderManager;

          enum class TransparencyMode
          {
            UNKNOWN
          , NONE
          , ORDER_INDEPENDENT_ALL
          , ORDER_INDEPENDENT_CLOSEST_ARRAY
          , ORDER_INDEPENDENT_CLOSEST_LIST
          , SORTED_BLENDED
          };

          class TransparencyManager
          {
            public:
              DP_SG_RDR_RIX_GL_API unsigned int getLayersCount() const;
              DP_SG_RDR_RIX_GL_API void setLayersCount( unsigned int count );
              DP_SG_RDR_RIX_GL_API ShaderManager * getShaderManager() const;
              DP_SG_RDR_RIX_GL_API void setShaderManager( ShaderManager * shaderManager );
              DP_SG_RDR_RIX_GL_API TransparencyMode getTransparencyMode() const;
              DP_SG_RDR_RIX_GL_API dp::math::Vec2ui const& getViewportSize() const;
              DP_SG_RDR_RIX_GL_API void setViewportSize( dp::math::Vec2ui const & size );

              DP_SG_RDR_RIX_GL_API virtual void addFragmentCodeSnippets( bool transparent, bool depth, std::vector<std::string> & snippets ) = 0;
              DP_SG_RDR_RIX_GL_API virtual void addFragmentParameters( std::vector<dp::rix::core::ProgramParameter> & parameters );
              DP_SG_RDR_RIX_GL_API virtual void addFragmentParameterSpecs( std::vector<dp::fx::ParameterSpec> & specs );
              DP_SG_RDR_RIX_GL_API virtual void updateFragmentParameters();
              DP_SG_RDR_RIX_GL_API virtual void beginTransparentPass( dp::rix::core::Renderer * renderer );
              DP_SG_RDR_RIX_GL_API virtual bool endTransparentPass();
              DP_SG_RDR_RIX_GL_API virtual void initializeParameterContainer( dp::rix::core::Renderer * renderer, dp::math::Vec2ui const & viewportSize ) = 0;
              DP_SG_RDR_RIX_GL_API virtual void useParameterContainer( dp::rix::core::Renderer * renderer, dp::rix::core::RenderGroupSharedHandle const & transparentRenderGroup ) = 0;
              DP_SG_RDR_RIX_GL_API virtual bool needsSortedRendering() const = 0;
              DP_SG_RDR_RIX_GL_API virtual bool supportsDepthPass() const;
              DP_SG_RDR_RIX_GL_API virtual void resolveDepthPass();

            protected:
              DP_SG_RDR_RIX_GL_API TransparencyManager( TransparencyMode transparencyMode );

              DP_SG_RDR_RIX_GL_API virtual void layersCountChanged();
              DP_SG_RDR_RIX_GL_API virtual void viewportSizeChanged();

            private:
              unsigned int        m_layersCount;
              ShaderManager     * m_shaderManager;
              TransparencyMode    m_transparencyMode;
              bool                m_transparentPass;
              dp::math::Vec2ui    m_viewportSize;
          };

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
