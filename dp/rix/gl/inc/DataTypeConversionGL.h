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

#include <GL/glew.h>
#include <dp/rix/core/RiX.h>
#include <dp/rix/gl/RiXGL.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      inline GLenum getGLCompareMode( dp::rix::core::SamplerStateCompareMode compareMode )
      {
        switch( compareMode )
        {
        case dp::rix::core::SSCM_NONE :
          return GL_NONE;
        case dp::rix::core::SSCM_R_TO_TEXTURE :
          return GL_COMPARE_R_TO_TEXTURE;
        default :
          DP_ASSERT( !"Unknown RiX SamplerStateCompareMode" );
          return GL_NONE;
        };
      }

      inline GLenum getGLDataType( dp::DataType dataType )
      {
        switch ( dataType )
        {
        case dp::DT_UNSIGNED_INT_8:
          return GL_UNSIGNED_BYTE;
        case dp::DT_UNSIGNED_INT_16:
          return GL_UNSIGNED_SHORT;
        case dp::DT_UNSIGNED_INT_32:
          return GL_UNSIGNED_INT;
        case dp::DT_INT_8:
          return GL_BYTE;
        case dp::DT_INT_16:
          return GL_SHORT;
        case dp::DT_INT_32:
          return GL_INT;
        case dp::DT_FLOAT_16:
          return GL_HALF_FLOAT;
        case dp::DT_FLOAT_32:
          return GL_FLOAT;
        case dp::DT_FLOAT_64:
          return GL_DOUBLE;
        default:
          DP_ASSERT( !"Unknwon RiX DataType" );
          return GL_FALSE;
        };
      }

      inline dp::DataType getDPDataType( GLenum dataType )
      {
        switch( dataType )
        {
          case GL_UNSIGNED_BYTE   : return( dp::DT_UNSIGNED_INT_8 );
          case GL_UNSIGNED_SHORT  : return( dp::DT_UNSIGNED_INT_16 );
          case GL_UNSIGNED_INT    : return( dp::DT_UNSIGNED_INT_32 );
          case GL_BYTE            : return( dp::DT_UNSIGNED_INT_8 );
          case GL_SHORT           : return( dp::DT_INT_16 );
          case GL_INT             : return( dp::DT_INT_32 );
          case GL_HALF_FLOAT      : return( dp::DT_FLOAT_16 );
          case GL_FLOAT           : return( dp::DT_FLOAT_32 );
          case GL_DOUBLE          : return( dp::DT_FLOAT_64 );
          default :
              DP_ASSERT( !"Unknown GL Data Type!" );
              return( dp::DT_UNKNOWN );
        }
      }

      inline GLenum getGLPixelFormat( dp::PixelFormat pixelFormat, GLenum internalFormat )
      {
        // handle integer formats differently
        switch (internalFormat) 
        {
        case GL_R8I:
        case GL_R8UI:
        case GL_R16I:
        case GL_R16UI:
        case GL_R32I:
        case GL_R32UI:
        case GL_RG8I:
        case GL_RG8UI:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_RG32I:
        case GL_RG32UI:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_RGB16I:
        case GL_RGB16UI:
        case GL_RGB32I:
        case GL_RGB32UI:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGBA16I:
        case GL_RGBA16UI:
        case GL_RGBA32I:
        case GL_RGBA32UI:
          switch ( pixelFormat )
          {
          case dp::PF_R:
            return GL_RED_INTEGER;
          case dp::PF_RG:
            return GL_RG_INTEGER;
          case dp::PF_RGB:
            return GL_RGB_INTEGER;
          case dp::PF_RGBA:
            return GL_RGBA_INTEGER;
          case dp::PF_BGR:
            return GL_BGR_INTEGER;
          case dp::PF_BGRA:
            return GL_BGRA_INTEGER;
          case dp::PF_LUMINANCE:
            return GL_LUMINANCE_INTEGER_EXT;
          case dp::PF_ALPHA:
            return GL_ALPHA_INTEGER;
          case dp::PF_LUMINANCE_ALPHA:
            return GL_LUMINANCE_ALPHA_INTEGER_EXT;
            // TODO
            /*
            case dp::util::PF_NATIVE:
            {
            DP_ASSERT( dynamic_cast<const *>(&) );
            const &  = static_cast<const &>();
            return ;
            }
            */
          default:
            DP_ASSERT( !"Unknown RiX PixelFormat");
            return GL_FALSE;
          }

        default:
          switch ( pixelFormat )
          {
          case dp::PF_R:
            return GL_RED;
          case dp::PF_RG:
            return GL_RG;
          case dp::PF_RGB:
            return GL_RGB;
          case dp::PF_RGBA:
            return GL_RGBA;
          case dp::PF_BGR:
            return GL_BGR;
          case dp::PF_BGRA:
            return GL_BGRA;
          case dp::PF_LUMINANCE:
            return GL_LUMINANCE;
          case dp::PF_ALPHA:
            return GL_ALPHA;
          case dp::PF_LUMINANCE_ALPHA:
            return GL_LUMINANCE_ALPHA;
            // TODO
            /*
            case dp::util::PF_NATIVE:
            {
            DP_ASSERT( dynamic_cast<const *>(&) );
            const &  = static_cast<const &>();
            return ;
            }
            */
          default:
            DP_ASSERT( !"Unknown RiX PixelFormat");
            return GL_FALSE;
          }
        }
      }

      inline dp::PixelFormat getDPPixelFormat( GLenum pixelFormat )
      {
        switch( pixelFormat )
        {
          case GL_RED             : return( dp::PF_R );
          case GL_RG              : return( dp::PF_RG );
          case GL_RGB             : return( dp::PF_RGB );
          case GL_RGBA            : return( dp::PF_RGBA );
          case GL_BGR             : return( dp::PF_BGR );
          case GL_BGRA            : return( dp::PF_BGRA );
          case GL_LUMINANCE       : return( dp::PF_LUMINANCE );
          case GL_ALPHA           : return( dp::PF_ALPHA );
          case GL_LUMINANCE_ALPHA : return( dp::PF_LUMINANCE_ALPHA );
          default :
            DP_ASSERT( !"Unknown GL Pixel Format!" );
            return( dp::PF_UNKNOWN );
        }
      }

      inline GLenum getGLInternalFormat( const dp::rix::core::TextureDescription& description )
      {
        switch ( description.m_internalFormat )
        {
        case dp::rix::core::ITF_R8:
          return GL_R8;
        case dp::rix::core::ITF_R16:
          return GL_R16;
        case dp::rix::core::ITF_RG8:
          return GL_RG8;
        case dp::rix::core::ITF_RG16:
          return GL_RG16;
        case dp::rix::core::ITF_RGB8:
          return GL_RGB8;
        case dp::rix::core::ITF_RGB16:
          return GL_RGB16;
        case dp::rix::core::ITF_RGBA8:
          return GL_RGBA8;
        case dp::rix::core::ITF_RGBA16:
          return GL_RGBA16;
        case dp::rix::core::ITF_R16F:
          return GL_R16F;
        case dp::rix::core::ITF_RG16F:
          return GL_RG16F;
        case dp::rix::core::ITF_RGB16F:
          return GL_RGB16F;
        case dp::rix::core::ITF_RGBA16F:
          return GL_RGBA16F;
        case dp::rix::core::ITF_R32F:
          return GL_R32F;
        case dp::rix::core::ITF_RG32F:
          return GL_RG32F;
        case dp::rix::core::ITF_RGB32F:
          return GL_RGB32F;
        case dp::rix::core::ITF_RGBA32F:
          return GL_RGBA32F;
        case dp::rix::core::ITF_R8I:
          return GL_R8I;
        case dp::rix::core::ITF_R8UI:
          return GL_R8UI;
        case dp::rix::core::ITF_R16I:
          return GL_R16I;
        case dp::rix::core::ITF_R16UI:
          return GL_R16UI;
        case dp::rix::core::ITF_R32I:
          return GL_R32I;
        case dp::rix::core::ITF_R32UI:
          return GL_R32UI;
        case dp::rix::core::ITF_RG8I:
          return GL_RG8I;
        case dp::rix::core::ITF_RG8UI:
          return GL_RG8UI;
        case dp::rix::core::ITF_RG16I:
          return GL_RG16I;
        case dp::rix::core::ITF_RG16UI:
          return GL_RG16UI;
        case dp::rix::core::ITF_RG32I:
          return GL_RG32I;
        case dp::rix::core::ITF_RG32UI:
          return GL_RG32UI;
        case dp::rix::core::ITF_RGB8I:
          return GL_RGB8I;
        case dp::rix::core::ITF_RGB8UI:
          return GL_RGB8UI;
        case dp::rix::core::ITF_RGB16I:
          return GL_RGB16I;
        case dp::rix::core::ITF_RGB16UI:
          return GL_RGB16UI;
        case dp::rix::core::ITF_RGB32I:
          return GL_RGB32I;
        case dp::rix::core::ITF_RGB32UI:
          return GL_RGB32UI;
        case dp::rix::core::ITF_RGBA8I:
          return GL_RGBA8I;
        case dp::rix::core::ITF_RGBA8UI:
          return GL_RGBA8UI;
        case dp::rix::core::ITF_RGBA16I:
          return GL_RGBA16I;
        case dp::rix::core::ITF_RGBA16UI:
          return GL_RGBA16UI;
        case dp::rix::core::ITF_RGBA32I:
          return GL_RGBA32I;
        case dp::rix::core::ITF_RGBA32UI:
          return GL_RGBA32UI;
        case dp::rix::core::ITF_COMPRESSED_R:
          return GL_COMPRESSED_RED;
        case dp::rix::core::ITF_COMPRESSED_RG:
          return GL_COMPRESSED_RG;
        case dp::rix::core::ITF_COMPRESSED_RGB:
          return GL_COMPRESSED_RGB;
        case dp::rix::core::ITF_COMPRESSED_RGBA:
          return GL_COMPRESSED_RGBA;
        case dp::rix::core::ITF_COMPRESSED_SRGB:
          return GL_COMPRESSED_SRGB;
        case dp::rix::core::ITF_COMPRESSED_SRGB_ALPHA:
          return GL_COMPRESSED_SRGB_ALPHA;
        case dp::rix::core::ITF_ALPHA:
          return GL_ALPHA;
        case dp::rix::core::ITF_LUMINANCE:
          return GL_LUMINANCE;
        case dp::rix::core::ITF_LUMINANCE_ALPHA:
          return GL_LUMINANCE_ALPHA;
        case dp::rix::core::ITF_RGB:
          return GL_RGB;
        case dp::rix::core::ITF_RGBA:
          return GL_RGBA;
        case dp::rix::core::ITF_NATIVE:
          {
            DP_ASSERT( dynamic_cast<const TextureDescriptionGL*>(&description) );
            const TextureDescriptionGL& descriptionGL = static_cast<const TextureDescriptionGL&>(description);
            return descriptionGL.m_internalFormatGL;
          }
        default:
          DP_ASSERT( !"Unknown RiX internal texture format" );
          return GL_FALSE;
        }
      }

      inline GLenum getGLPrimitiveType( GeometryPrimitiveType primitiveType )
      {
        switch ( primitiveType )
        {
        case GPT_POINTS:
          return GL_POINTS;
        case GPT_LINE_STRIP:
          return GL_LINE_STRIP;
        case GPT_LINE_LOOP:
          return GL_LINE_LOOP;
        case GPT_LINES:
          return GL_LINES;
        case GPT_TRIANGLE_STRIP:
          return GL_TRIANGLE_STRIP;
        case GPT_TRIANGLE_FAN:
          return GL_TRIANGLE_FAN;
        case GPT_TRIANGLES:
          return GL_TRIANGLES;
        case GPT_QUAD_STRIP:
          return GL_QUAD_STRIP;
        case GPT_QUADS:
          return GL_QUADS;
        case GPT_POLYGON:
          return GL_POLYGON;
        case GPT_PATCHES:
          return GL_PATCHES;
          // TODO
          /*
        case GPT_NATIVE:
          {
            DP_ASSERT( dynamic_cast<const *>(&) );
            const &  = static_cast<const &>();
            return ;
          }
          */
        default:
          DP_ASSERT( !"Unknown RiX GeometryPrimitiveType" );
          return GL_FALSE;
        }
      }

      inline GLenum getGLFilterMode( dp::rix::core::SamplerStateFilterMode filterMode )
      {
        switch ( filterMode )
        {
        case dp::rix::core::SSFM_NEAREST:
           return GL_NEAREST;
        case dp::rix::core::SSFM_LINEAR:
           return GL_LINEAR;
        case dp::rix::core::SSFM_NEAREST_MIPMAP_NEAREST:
           return GL_NEAREST_MIPMAP_NEAREST;
        case dp::rix::core::SSFM_LINEAR_MIPMAP_NEAREST:
           return GL_LINEAR_MIPMAP_NEAREST;
        case dp::rix::core::SSFM_NEAREST_MIPMAP_LINEAR:
           return GL_NEAREST_MIPMAP_LINEAR;
        case dp::rix::core::SSFM_LINEAR_MIPMAP_LINEAR:
           return GL_LINEAR_MIPMAP_LINEAR;
        default:
          DP_ASSERT( !"Unknown RiX SamplerStateFilterMode" );
          return GL_FALSE;
        }
      }

      inline GLenum getGLWrapMode( dp::rix::core::SamplerStateWrapMode wrapMode )
      {
        switch ( wrapMode )
        {
        case dp::rix::core::SSWM_CLAMP:
           return GL_CLAMP;
        case dp::rix::core::SSWM_CLAMP_TO_BORDER:
           return GL_CLAMP_TO_BORDER;
        case dp::rix::core::SSWM_CLAMP_TO_EDGE:
           return GL_CLAMP_TO_EDGE;
        case dp::rix::core::SSWM_MIRRORED_REPEAT:
           return GL_MIRRORED_REPEAT;
        case dp::rix::core::SSWM_REPEAT:
           return GL_REPEAT;
        default:
          DP_ASSERT( !"Unknown RiX SamplerStateWrapMode" );
          return GL_FALSE;
        }
      }

      inline GLenum getGLAccessMode( dp::rix::core::AccessType access )
      {
        switch ( access )
        {
        case dp::rix::core::AT_READ_ONLY:
          return GL_READ_ONLY;
        case dp::rix::core::AT_WRITE_ONLY:
          return GL_WRITE_ONLY;
        case dp::rix::core::AT_READ_WRITE:
          return GL_READ_WRITE;
        default:
          DP_ASSERT( !"Unexpected RiX AccessType" );
          return GL_FALSE;
        }
      }

      
      inline GLenum getGLProgramDomain( dp::rix::core::ShaderType shaderType )
      {
        switch ( shaderType )
        {
          case dp::rix::core::ST_VERTEX_SHADER:
            return GL_VERTEX_SHADER;
          case dp::rix::core::ST_TESS_CONTROL_SHADER:
            return GL_TESS_CONTROL_SHADER;
          case dp::rix::core::ST_TESS_EVALUATION_SHADER:
            return GL_TESS_EVALUATION_SHADER;
          case dp::rix::core::ST_GEOMETRY_SHADER:
            return GL_GEOMETRY_SHADER;
          case dp::rix::core::ST_FRAGMENT_SHADER:
            return GL_FRAGMENT_SHADER;
          default:
            DP_ASSERT( !"Unknown RiX ShaderType");
            return GL_FALSE;
        }
      }

      inline dp::rix::core::ShaderType getShaderType( GLenum type )
      {
        switch( type )
        {
          case GL_VERTEX_SHADER           : return( dp::rix::core::ST_VERTEX_SHADER );
          case GL_TESS_CONTROL_SHADER     : return( dp::rix::core::ST_TESS_CONTROL_SHADER );
          case GL_TESS_EVALUATION_SHADER  : return( dp::rix::core::ST_TESS_EVALUATION_SHADER );
          case GL_GEOMETRY_SHADER         : return( dp::rix::core::ST_GEOMETRY_SHADER );
          case GL_FRAGMENT_SHADER         : return( dp::rix::core::ST_FRAGMENT_SHADER );
          default :
            DP_ASSERT( !"Invalid GL Shader Type" );
            return( dp::rix::core::ST_NUM_SHADERTYPES );
        }
      }

      inline GLbitfield getGLShaderBits( dp::rix::core::ShaderType shaderType )
      {
        switch ( shaderType )
        {
        case dp::rix::core::ST_VERTEX_SHADER:
          return GL_VERTEX_SHADER_BIT;
        case dp::rix::core::ST_TESS_CONTROL_SHADER:
          return GL_TESS_CONTROL_SHADER_BIT;
        case dp::rix::core::ST_TESS_EVALUATION_SHADER:
          return GL_TESS_EVALUATION_SHADER_BIT;
        case dp::rix::core::ST_GEOMETRY_SHADER:
          return GL_GEOMETRY_SHADER_BIT;
        case dp::rix::core::ST_FRAGMENT_SHADER:
          return GL_FRAGMENT_SHADER_BIT;
        default:
          DP_ASSERT( !"Unknown RiX ShaderType");
          return GL_FALSE;
        }
      }
    } // namespace gl
  } // namespace rix
} // namespace dp
