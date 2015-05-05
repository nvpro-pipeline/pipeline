// Copyright NVIDIA Corporation 2011-2015
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

#include <dp/util/Config.h>

#if defined(_WIN32)
#  ifdef RIX_GL_EXPORTS
#    define RIX_GL_API __declspec(dllexport)
#  else
#    define RIX_GL_API __declspec(dllimport)
#  endif
#else
#  define RIX_GL_API
#endif


// Controls whether RiXGL is compiled to use the OpenGL sampler objects (GL_ARB_sampler_objects)
#define RIX_GL_SAMPLEROBJECT_SUPPORT 1

// Controls whether RiXGL is compiled to support OpenGL separate shader objects (GL_ARB_separate_shader_objects)
#define RIX_GL_SEPARATE_SHADER_OBJECTS_SUPPORT 0

// Enable bindless texture within the Bindless RenderEngine if available
#define RIX_GL_ENABLE_BINDLESS_TEXTURE 0

// Define for how many attributes RiXGL is compiled
#define RIX_GL_MAX_ATTRIBUTES 16

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      // UBO parameter technique to switch between parameters
      enum BufferMode
      {
        BM_BIND_BUFFER_RANGE,         // put parameters in a big UBO, use glBindBufferRange to switch between parameters
        BM_BUFFER_SUBDATA,            // create one UBO for each binding, use glBufferSubData to switch between parameters
        BM_PERSISTENT_BUFFER_MAPPING  // put parameters in a big persistently mapped UBO, use glBindBufferRange to switch between parameters
      };

    }
  }
}
