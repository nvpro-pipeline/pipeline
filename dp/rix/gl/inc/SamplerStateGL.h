// Copyright NVIDIA Corporation 2012
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

#include <dp/rix/gl/RiXGL.h>
#include <GL/glew.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      class SamplerStateGL : public dp::rix::core::SamplerState
      {
      protected:
        SamplerStateGL();

      public:
        ~SamplerStateGL();

        static dp::rix::core::SamplerStateHandle create( const dp::rix::core::SamplerStateData& samplerStateData );

  #if RIX_GL_SAMPLEROBJECT_SUPPORT
        void updateSamplerObject();

        GLuint m_id;
  #endif

        SamplerBorderColorDataType m_borderColorDataType;
        union
        {
          float        f[4];
          unsigned int ui[4];
          int          i[4];
        } m_borderColor;

        GLenum m_minFilterModeGL;
        GLenum m_magFilterModeGL;
        GLenum m_wrapSModeGL;
        GLenum m_wrapTModeGL;
        GLenum m_wrapRModeGL;    

        float        m_minLOD;
        float        m_maxLOD;
        float        m_LODBias;

        unsigned int m_compareModeGL;
        unsigned int m_compareFuncGL;

        float        m_maxAnisotropy;
      };
    } // namespace gl
  } // namespace rix
} // namespace dp
