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
#include <dp/sg/renderer/rix/gl/TransparencyManager.h>

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
          SMART_TYPES( TransparencyManagerSB );

          class DrawableManagerDefault;
          class ShaderManager;

          class TransparencyManagerSB : public TransparencyManager
          {
            public:
              DP_SG_RDR_RIX_GL_API static SmartTransparencyManagerSB create();
              DP_SG_RDR_RIX_GL_API virtual ~TransparencyManagerSB();

            public:
              DP_SG_RDR_RIX_GL_API virtual void addFragmentCodeSnippets( bool transparent, bool depth, std::vector<std::string> & snippets );
              DP_SG_RDR_RIX_GL_API virtual void beginTransparentPass( dp::rix::core::Renderer * renderer );
              DP_SG_RDR_RIX_GL_API virtual bool endTransparentPass();
              DP_SG_RDR_RIX_GL_API virtual void initializeParameterContainer( dp::rix::core::Renderer * renderer, dp::math::Vec2ui const & viewportSize );
              DP_SG_RDR_RIX_GL_API virtual void useParameterContainer( dp::rix::core::Renderer * renderer, dp::rix::core::RenderGroupSharedHandle const & transparentRenderGroup );
              DP_SG_RDR_RIX_GL_API virtual bool needsSortedRendering() const;

            protected:
              TransparencyManagerSB();
          };

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
