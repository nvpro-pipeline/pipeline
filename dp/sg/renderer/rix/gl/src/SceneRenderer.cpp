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


#include <dp/sg/renderer/rix/gl/SceneRenderer.h>
#include <dp/sg/renderer/rix/gl/inc/SceneRendererImpl.h>
#include <dp/sg/renderer/rix/gl/inc/DrawableManagerDefault.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/rix/gl/RiXGL.h>

#define ENABLE_PROFILING 0
#include <dp/util/Profile.h>

#include <iostream>

using namespace dp::math;
using namespace dp::sg::xbar;

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

          SceneRenderer::SceneRenderer( const dp::gl::SmartRenderTarget &renderTarget )
            : dp::sg::ui::SceneRenderer( renderTarget )
            , m_depthPass( false )

          {
          }

          SceneRenderer::~SceneRenderer()
          {
          }

          SmartSceneRenderer SceneRenderer::create( const char *renderEngine, dp::fx::Manager shaderManagerType
                                                  , dp::culling::Mode cullingMode, TransparencyMode transparencyMode
                                                  , const dp::gl::SmartRenderTarget &renderTarget )
          {
            return SceneRendererImpl::create( renderEngine, shaderManagerType, cullingMode, transparencyMode, renderTarget );
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
