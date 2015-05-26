// Copyright NVIDIA Corporation 2014-2015
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


#include <dp/rix/gl/inc/TextureGL.h>
#include <dp/rix/gl/inc/DataTypeConversionGL.h>

#include <dp/rix/gl/inc/BufferGL.h>
#include <dp/rix/gl/inc/SamplerStateGL.h>
#include <dp/rix/gl/inc/TextureGLBuffer.h>
#include <dp/rix/gl/inc/TextureGL1D.h>
#include <dp/rix/gl/inc/TextureGL1DArray.h>
#include <dp/rix/gl/inc/TextureGL2D.h>
#include <dp/rix/gl/inc/TextureGL2DArray.h>
#include <dp/rix/gl/inc/TextureGL2DRectangle.h>
#include <dp/rix/gl/inc/TextureGL3D.h>
#include <dp/rix/gl/inc/TextureGLCubeMap.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      using namespace dp::rix::core;

      TextureGL::TextureGL( dp::gl::TextureSharedPtr const& texture, bool mipMapped )
        : m_hasMipmaps( mipMapped )
        , m_texture( texture )
      {
        DP_ASSERT( m_texture );

        SamplerStateDataCommon samplerStateDataCommon;
  #if RIX_GL_SAMPLEROBJECT_SUPPORT
        // here we don't need a default sampler state, default is just set via apply
        // and an opengl sampler object can override this state without changing the texture

        if( m_texture->getTarget() != GL_TEXTURE_BUFFER )
        {
          // apply default sampler state to texture (but dont store it as an object)
          SamplerStateGLSharedHandle tmpSamplerState = static_cast<dp::rix::gl::SamplerStateGLHandle>(SamplerStateGL::create( samplerStateDataCommon ));
          applySamplerState( tmpSamplerState.get() );
        }
  #else
        if( m_texture->getTarget() != GL_TEXTURE_BUFFER )
        {
          m_defaultSamplerStateHandle = static_cast<dp::rix::gl::SamplerStateGLHandle>(SamplerStateGL::create( samplerStateDataCommon ));
          applySamplerState( m_defaultSamplerStateHandle.get() );
        }
  #endif
      }

      TextureGL::~TextureGL()
      {
      }

      TextureHandle TextureGL::create( const dp::rix::core::TextureDescription& textureDescription )
      {
        // TODO replace by factory where objects automatically get registered
        switch ( textureDescription.m_type )
        {
        case TT_BUFFER:
          return TextureGLBuffer::create( textureDescription );
        case TT_1D:
          return TextureGL1D::create( textureDescription );
        case TT_1D_ARRAY:
          return TextureGL1DArray::create( textureDescription );
        case TT_2D:
          return TextureGL2D::create( textureDescription );
        case TT_2D_ARRAY:
          return TextureGL2DArray::create( textureDescription );
        case TT_2D_RECTANGLE:
          return TextureGL2DRectangle::create( textureDescription );
        case TT_3D:
          return TextureGL3D::create( textureDescription );
        case TT_CUBEMAP:
          return TextureGLCubeMap::create( textureDescription );
        case TT_NATIVE:
          {
            DP_ASSERT( dynamic_cast<const TextureDescriptionGL*>(&textureDescription) );
            DP_ASSERT(!"native texture types not yet implemented");
          }
          return nullptr;
        default:
          DP_ASSERT( 0 && "unsupported texture type" );
          return nullptr;
        }
      }

      bool TextureGL::setData( const TextureData& textureData )
      {
        switch( textureData.getTextureDataType() )
        {
        case dp::rix::core::TDT_NATIVE:
          {
            DP_ASSERT( dynamic_cast<const TextureDataGLTexture*>(&textureData) );
            const TextureDataGLTexture& textureDataGLTexture = static_cast<const TextureDataGLTexture&>(textureData);
            setTexture( textureDataGLTexture.m_texture );
          }
          break;

        case dp::rix::core::TDT_BUFFER:
          {
            DP_ASSERT(!"this function is implemented in TextureGLBuffer.h");
            return false;
          }
          break;

        case dp::rix::core::TDT_POINTER:
          {
            DP_ASSERT( m_texture->getTarget() != GL_TEXTURE_BUFFER );
            DP_ASSERT( dynamic_cast<const dp::rix::core::TextureDataPtr*>(&textureData) );
            const TextureDataPtr& dataPtr = static_cast<const TextureDataPtr&>(textureData);

            DP_ASSERT( dataPtr.m_pixelFormat < dp::PF_NUM_PIXELFORMATS );
            DP_ASSERT( dataPtr.m_pixelDataType < dp::DT_NUM_DATATYPES );
            DP_ASSERT( (!m_hasMipmaps && dataPtr.m_numMipMapLevels == 0) || m_hasMipmaps );

            GLenum glFormat   = getGLPixelFormat( dataPtr.m_pixelFormat, m_texture->getInternalFormat() );
            GLenum glDataType = getGLDataType( dataPtr.m_pixelDataType );

            // update storage for the texture
            if ( m_texture->getFormat() != glFormat || m_texture->getType() != glDataType )
            {
              m_texture->setFormat( glFormat );
              m_texture->setType( glDataType );
            }

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // E.g. GL_RGB GL_UNSIGNED_BYTE with odd widths requires this!

            unsigned int numMipMapLevels = dataPtr.m_numMipMapLevels ? dataPtr.m_numMipMapLevels : 1;

            unsigned int idx = 0;
            for ( unsigned int layer = 0; layer < dataPtr.m_numLayers; ++layer )
            {
              for ( unsigned int mipMapLevel = 0; mipMapLevel < numMipMapLevels; ++mipMapLevel )
              {
                upload( mipMapLevel, layer, dataPtr.m_data[idx++] );
              }
            }
            if( m_hasMipmaps && !dataPtr.m_numMipMapLevels )
            {
              m_texture->generateMipMap();
            }
          }
          break;

        default:
          DP_ASSERT( 0 && "unsupported texture data type" );
          break;
        }

        return true;
      }

      void TextureGL::applySamplerState( SamplerStateGLHandle samplerState )
      {
        // this function expects the texture to be already bound!
        // TODO: here we could DP_ASSERT on the right texture binding via glGetIntegerv(GL_TEXTURE_BINDING_*, id )

        DP_ASSERT( samplerState );
        DP_ASSERT( m_texture->getTarget() != GL_TEXTURE_BUFFER ); // texture buffers should not have a sampler state

  #if RIX_GL_SAMPLEROBJECT_SUPPORT
  #else
        m_lastSamplerStateHandle = samplerState;
  #endif

        switch( samplerState->m_borderColorDataType )
        {
        case SBCDT_FLOAT:
          m_texture->setBorderColor( samplerState->m_borderColor.f );
          break;
        case SBCDT_UINT:
          m_texture->setBorderColor( samplerState->m_borderColor.ui );
          break;
        case SBCDT_INT:
          m_texture->setBorderColor( samplerState->m_borderColor.i );
          break;
        default:
          DP_ASSERT( !"unknown sampler border color data type" );
          break;
        }

        m_texture->setFilterParameters( samplerState->m_minFilterModeGL, samplerState->m_magFilterModeGL );
        m_texture->setWrapParameters( samplerState->m_wrapSModeGL, samplerState->m_wrapTModeGL, samplerState->m_wrapRModeGL );
        m_texture->setLODParameters( samplerState->m_minLOD, samplerState->m_maxLOD, samplerState->m_LODBias );
        m_texture->setCompareParameters( samplerState->m_compareModeGL, samplerState->m_compareFuncGL );
        m_texture->setMaxAnisotropy( samplerState->m_maxAnisotropy );
      }

      void TextureGL::applyDefaultSamplerState()
      {
  #if RIX_GL_SAMPLEROBJECT_SUPPORT
  #else
        DP_ASSERT( m_defaultSamplerStateHandle );
        applySamplerState( m_defaultSamplerStateHandle.get() );
  #endif
      }

      void TextureGL::setDefaultSamplerState( SamplerStateGLHandle samplerState )
      {
        DP_ASSERT( samplerState );

        applySamplerState( samplerState );
  #if RIX_GL_SAMPLEROBJECT_SUPPORT
        // dont store the default sampler state
  #else
        m_defaultSamplerStateHandle = samplerState;
  #endif
      }

      void TextureGL::upload( unsigned int /*mipMapLevel*/, unsigned int /*layer*/, const void* /*data*/ )
      {
        DP_ASSERT( 0 && "generic update function not implemented");
      }

      unsigned int TextureGL::getNumberOfMipMapLevels() const
      {
        DP_ASSERT( m_texture );
        return( m_texture->getMaxLevel() );
      }

      unsigned int TextureGL::getMipMapSize( unsigned int size, unsigned int mipMapLevel )
      {
        unsigned int newSize = size / (1 << mipMapLevel);
        return newSize == 0 ? 1 : newSize;
      }

      TextureGL::BindlessReferenceHandle TextureGL::getBindlessTextureHandle( SamplerStateGLHandle samplerState )
      {
        GLuint sampler_id = 0;
        if (samplerState)
        {
#if RIX_GL_SAMPLEROBJECT_SUPPORT == 0
          DP_ASSERT(0 && "BindlessTexture with explicit samplerState is not supported without samplerobject support");
#else
          sampler_id = samplerState->m_id;
#endif
        }

        // TODO might need to cache this
        return new BindlessReference(m_texture->getGLId(), sampler_id);
      }

      TextureGL::BindlessReferenceHandle TextureGL::getBindlessTextureHandle( GLint level, GLboolean layered, GLint layer, GLint format, GLenum access )
      {
        return new BindlessReference(m_texture->getGLId(), level, layered, layer, format, access);
      }

      TextureGL::BindlessReference::BindlessReference( GLuint tex, GLuint sampler )
      {
        DP_ASSERT( tex );
        if (!sampler)
        {
          m_handle = glGetTextureHandleNV( tex );
        }
        else
        {
          m_handle = glGetTextureSamplerHandleNV( tex, sampler );
        }
        m_isImage = false;
        if ( !glIsTextureHandleResidentNV( m_handle ) )
        {
          glMakeTextureHandleResidentNV( m_handle );
        }
      }

      TextureGL::BindlessReference::BindlessReference( GLuint tex, GLint level, GLboolean layered, GLint layer, GLint format, GLenum access )
      {
        m_isImage = true;
        m_handle = glGetImageHandleNV( tex, level, layered, layer, format);
        if ( !glIsImageHandleResidentNV( m_handle ) )
        {
          glMakeImageHandleResidentNV( m_handle, access );
        }
      }

      TextureGL::BindlessReference::~BindlessReference()
      {
        // For one texture/sampler tuple the driver creates unique
        // handles. We need to refcount the tuples to determine if
        // it's still in use somewhere. With this feature it would
        // be possible to hold only textures required in gpu memory.
        // Till then a texture will always be resident until it gets
        // deleted.
  #if 0
        if (m_isImage)
        {
          glMakeImageHandleNonResidentNV( m_handle );
        }
        else
        {
          glMakeTextureHandleNonResidentNV( m_handle );
        }
  #endif
      }

      void TextureGL::setTexture( dp::gl::TextureSharedPtr const& texture )
      {
        DP_ASSERT( texture && glIsTexture( texture->getGLId() ) );
        m_texture = texture;
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
