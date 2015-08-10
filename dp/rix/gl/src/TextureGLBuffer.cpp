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


#include <dp/rix/gl/inc/BufferGL.h>
#include <dp/rix/gl/inc/DataTypeConversionGL.h>
#include <dp/rix/gl/inc/TextureGLBuffer.h>
#include <dp/rix/gl/inc/SamplerStateGL.h>
#include <cmath>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      using namespace dp::rix::core;

      dp::rix::core::TextureHandle TextureGLBuffer::create( const TextureDescription &textureDescription )
      {
        return new TextureGLBuffer( textureDescription );
      }

      bool TextureGLBuffer::setData( const TextureData& textureData )
      {
        DP_ASSERT( getTexture()->getTarget() == GL_TEXTURE_BUFFER );

        switch ( textureData.getTextureDataType() )
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
            DP_ASSERT( dynamic_cast<const TextureDataBuffer*>(&textureData) );
            const TextureDataBuffer& dataBuffer = static_cast<const TextureDataBuffer&>(textureData);

            DP_ASSERT( handleIsTypeOf<BufferGL>( dataBuffer.m_buffer ) );
            BufferGLHandle buffer = handleCast<BufferGL>( dataBuffer.m_buffer.get() );

            getTexture().inplaceCast<dp::gl::TextureBuffer>()->setBuffer( buffer->getBuffer() );

            m_textureBuffer = buffer;
          }
          break;
        case dp::rix::core::TDT_POINTER:
        case dp::rix::core::TDT_NUM_TEXTUREDATATYPES:
          DP_ASSERT( !"unsupported" );
          break;
        }

        return true;
      }

      TextureGLBuffer::TextureGLBuffer( const dp::rix::core::TextureDescription& description )
        : TextureGL( dp::gl::TextureBuffer::create( getGLInternalFormat( description ) ), description.m_mipmaps )
        , m_textureBuffer( nullptr )
      {
        DP_ASSERT( description.m_width  == 0 );
        DP_ASSERT( description.m_height == 0 );
        DP_ASSERT( description.m_depth  == 0 );
        DP_ASSERT( description.m_layers == 0 );
        
  #if RIX_GL_SAMPLEROBJECT_SUPPORT
  #else
        // delete the defaulte sampler state for texture buffers, they should not have a sampler state
        m_defaultSamplerStateHandle.reset();
  #endif
      }

      void TextureGLBuffer::upload( unsigned int /*mipMapLevel*/, unsigned int /*layer*/, const void* /*data*/ )
      {
        DP_ASSERT(!"don't call this");
      }

      void TextureGLBuffer::applySamplerState( SamplerStateGLHandle samplerState )
      {
        DP_ASSERT( samplerState == 0 );
        if ( samplerState != 0 )
        {
          /* TODO throw exception */
        }
      }

      void TextureGLBuffer::applyDefaultSamplerState()
      {
      }

      void TextureGLBuffer::setDefaultSamplerState( SamplerStateGLHandle samplerState )
      {
        DP_ASSERT( samplerState == 0 );
        if ( samplerState != 0 )
        {
          /* TODO throw exception */
        }
      }

    } // namespace gl
  } // namespace rix
} // namespace dp


