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


#include <cstring>
#include <dp/rix/gl/inc/SamplerStateGL.h>
#include <dp/rix/gl/inc/DataTypeConversionGL.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      using namespace dp::rix::core;

      SamplerStateGL::SamplerStateGL()
      {
        m_borderColorDataType = SBCDT_FLOAT;
        m_borderColor.f[0]  = 0.0f;
        m_borderColor.f[1]  = 0.0f;
        m_borderColor.f[2]  = 0.0f;
        m_borderColor.f[3]  = 0.0f;

        m_minFilterModeGL = GL_NEAREST;
        m_magFilterModeGL = GL_NEAREST;
        m_wrapSModeGL     = GL_CLAMP_TO_EDGE;
        m_wrapTModeGL     = GL_CLAMP_TO_EDGE;
        m_wrapRModeGL     = GL_CLAMP_TO_EDGE;

        m_minLOD          = -1000.0f;
        m_maxLOD          =  1000.0f;
        m_LODBias         =  0.0f;

        m_compareModeGL   = GL_NONE;
        m_compareFuncGL   = GL_LEQUAL;

        m_maxAnisotropy   = 1.0f;

  #if RIX_GL_SAMPLEROBJECT_SUPPORT
        glGenSamplers( 1, &m_id );
        assert( m_id && "Couldn't create OpenGL sampler object" );
        updateSamplerObject();
  #endif
      }

      SamplerStateGL::~SamplerStateGL()
      {
  #if RIX_GL_SAMPLEROBJECT_SUPPORT
        if( m_id )
        {
          glDeleteSamplers( 1, &m_id );
        }
  #else
        // nothing to clean up
  #endif
      }

      SamplerStateHandle SamplerStateGL::create( const SamplerStateData& samplerStateData )
      {
        SamplerStateGLHandle samplerStateGL = new SamplerStateGL;

        switch( samplerStateData.getSamplerStateDataType() )
        {
        case dp::rix::core::SSDT_COMMON:
          {
            assert( dynamic_cast<const dp::rix::core::SamplerStateDataCommon*>(&samplerStateData) );
            const SamplerStateDataCommon& samplerStateDataCommon = static_cast<const SamplerStateDataCommon&>(samplerStateData);

            samplerStateGL->m_minFilterModeGL = getGLFilterMode( samplerStateDataCommon.m_minFilterMode );
            samplerStateGL->m_magFilterModeGL = getGLFilterMode( samplerStateDataCommon.m_magFilterMode );
            samplerStateGL->m_wrapSModeGL     = getGLWrapMode( samplerStateDataCommon.m_wrapSMode );
            samplerStateGL->m_wrapTModeGL     = getGLWrapMode( samplerStateDataCommon.m_wrapTMode );
            samplerStateGL->m_wrapRModeGL     = getGLWrapMode( samplerStateDataCommon.m_wrapRMode );
            samplerStateGL->m_compareModeGL   = getGLCompareMode( samplerStateDataCommon.m_compareMode );
          }
          break;

        case dp::rix::core::SSDT_NATIVE:
          {          
            assert( dynamic_cast<const SamplerStateDataGL*>(&samplerStateData) );
            const SamplerStateDataGL& samplerStateDataGL = static_cast<const SamplerStateDataGL&>(samplerStateData);

            samplerStateGL->m_borderColorDataType = samplerStateDataGL.m_borderColorDataType;

            // either this or a switch case copying all data one by one
            memcpy( &samplerStateGL->m_borderColor, &samplerStateDataGL.m_borderColor, sizeof(samplerStateGL->m_borderColor ) );

            samplerStateGL->m_minFilterModeGL = samplerStateDataGL.m_minFilterModeGL;
            samplerStateGL->m_magFilterModeGL = samplerStateDataGL.m_magFilterModeGL;
            samplerStateGL->m_wrapSModeGL     = samplerStateDataGL.m_wrapSModeGL;
            samplerStateGL->m_wrapTModeGL     = samplerStateDataGL.m_wrapTModeGL;
            samplerStateGL->m_wrapRModeGL     = samplerStateDataGL.m_wrapRModeGL;

            samplerStateGL->m_minLOD          = samplerStateDataGL.m_minLOD;
            samplerStateGL->m_maxLOD          = samplerStateDataGL.m_maxLOD;
            samplerStateGL->m_LODBias         = samplerStateDataGL.m_LODBias;

            samplerStateGL->m_compareModeGL   = samplerStateDataGL.m_compareModeGL;
            samplerStateGL->m_compareFuncGL   = samplerStateDataGL.m_compareFuncGL;

            samplerStateGL->m_maxAnisotropy   = samplerStateDataGL.m_maxAnisotropy;
          }
          break;

        default:
          {
            assert( !"unsupported sampler state data type" );
            delete samplerStateGL;
            return nullptr;
          }
          break;
        }

  #if RIX_GL_SAMPLEROBJECT_SUPPORT
        samplerStateGL->updateSamplerObject();
  #endif

        return handleCast<SamplerState>(samplerStateGL);
      }

  #if RIX_GL_SAMPLEROBJECT_SUPPORT
      void SamplerStateGL::updateSamplerObject()
      {
        switch( m_borderColorDataType )
        {
        case SBCDT_FLOAT:
          glSamplerParameterfv( m_id, GL_TEXTURE_BORDER_COLOR, m_borderColor.f );
          break;
        case SBCDT_UINT:
          glSamplerParameterIuiv( m_id, GL_TEXTURE_BORDER_COLOR, m_borderColor.ui );
          break;
        case SBCDT_INT:
          glSamplerParameterIiv( m_id, GL_TEXTURE_BORDER_COLOR, m_borderColor.i );
          break;
        default:
          assert( !"unknown sampler border color data type" );
          break;
        }

        glSamplerParameteri( m_id, GL_TEXTURE_MIN_FILTER, m_minFilterModeGL );
        glSamplerParameteri( m_id, GL_TEXTURE_MAG_FILTER, m_magFilterModeGL );
        glSamplerParameteri( m_id, GL_TEXTURE_WRAP_S, m_wrapSModeGL );
        glSamplerParameteri( m_id, GL_TEXTURE_WRAP_T, m_wrapTModeGL );
        glSamplerParameteri( m_id, GL_TEXTURE_WRAP_R, m_wrapRModeGL );

        glSamplerParameterf( m_id, GL_TEXTURE_MIN_LOD, m_minLOD );
        glSamplerParameterf( m_id, GL_TEXTURE_MAX_LOD, m_maxLOD );
        glSamplerParameterf( m_id, GL_TEXTURE_LOD_BIAS, m_LODBias );

        glSamplerParameteri( m_id, GL_TEXTURE_COMPARE_MODE, m_compareModeGL );
        glSamplerParameteri( m_id, GL_TEXTURE_COMPARE_FUNC, m_compareFuncGL );

        glSamplerParameterf( m_id, GL_TEXTURE_MAX_ANISOTROPY_EXT, m_maxAnisotropy );
      }
  #endif

    } // namespace gl
  } // namespace rix
} // namespace dp
