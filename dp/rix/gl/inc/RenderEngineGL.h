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

#include <dp/rix/core/RiX.h>
#include <RenderGroupGL.h>


namespace dp
{
  namespace rix
  {
    namespace gl
    {

      typedef RenderEngineGL*(*RenderEngineCreator)(std::map<std::string, std::string> const & options);

      typedef std::map<std::string, RenderEngineCreator> RenderEngineMap;

      RenderEngineMap &getRenderEngineMap();
      RenderEngineGL* getRenderEngine( const char *name );
      bool registerRenderEngine( const char *renderEngine, RenderEngineCreator creator );

      class RenderEngineGL 
      {
      public:
        virtual ~RenderEngineGL() {};

        virtual void beginRender() = 0;
        virtual void render( RenderGroupGLSharedHandle const & groupHandle, dp::rix::core::RenderOptions const & renderOptions = dp::rix::core::RenderOptions() ) = 0;
        virtual void render( RenderGroupGLSharedHandle const & groupHandle, dp::rix::core::GeometryInstanceSharedHandle const * gis, size_t numGIs, dp::rix::core::RenderOptions const & renderOptions = dp::rix::core::RenderOptions() ) = 0;
        virtual void endRender() = 0;
        virtual RenderGroupGL::SmartCache createCache( RenderGroupGLSharedHandle const & renderGroupGL, ProgramPipelineGLSharedHandle const & programPipeline ) = 0;

      };
    } // namespace gl
  } // namespace rix
} // namespace dp
 
