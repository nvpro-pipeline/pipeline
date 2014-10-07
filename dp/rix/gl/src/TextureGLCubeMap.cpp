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


#include "TextureGLCubeMap.h"
#include "DataTypeConversionGL.h"
#include <dp/util/SharedPtr.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      using namespace dp::rix::core;

      TextureHandle TextureGLCubeMap::create( const TextureDescription &textureDescription )
      {
        return new TextureGLCubeMap( textureDescription );
      }

      TextureGLCubeMap::TextureGLCubeMap( const TextureDescription& description )
        : TextureGL( dp::gl::TextureCubemap::create( getGLInternalFormat( description )
                                                   , getGLPixelFormat( description.m_pixelFormat, getGLInternalFormat( description ) )
                                                   , getGLDataType( description.m_dataType )
                                                   , static_cast<GLsizei>(description.m_width)
                                                   , static_cast<GLsizei>(description.m_height) )
                   , description.m_mipmaps )
      {
        DP_ASSERT( description.m_layers == 0 );
      }

      bool TextureGLCubeMap::setData( const TextureData& textureData )
      {
        bool ok = false;
        switch( textureData.getTextureDataType() )
        {
        case dp::rix::core::TDT_NATIVE:
          {
            ok = TextureGL::setData( textureData );
          }
          break;

        case dp::rix::core::TDT_POINTER:
          {
            DP_ASSERT( dynamic_cast<const dp::rix::core::TextureDataPtr*>(&textureData) );
            const TextureDataPtr& dataPtr = static_cast<const TextureDataPtr&>(textureData);

            // make sure TextureDataPtr has exactly 6 layers (one for each cube face)
            if( dataPtr.m_numLayers != 6 )
            {
              DP_ASSERT( !"TextureDataPtr for cube maps needs exactly 6 layers" );
              return false;
            }

            // call base class setData, which validates data and calls upload( mipmaplevel, layer, data )
            ok = TextureGL::setData( textureData );
          }
          break;

        default:
          {
            DP_ASSERT( 0 && "unsupported texture data type" );
            ok = false;
          }
          break;

        }
        return ok;
      }

      void TextureGLCubeMap::upload( unsigned int mipMapLevel, unsigned int layer, const void *data )
      {
        DP_ASSERT( layer < 6 );

        // compute target from layer in order +x,-x,+y,-y,+z,-z
        dp::util::shared_cast<dp::gl::TextureCubemap>( getTexture() )->setData( data, layer, mipMapLevel );
      }

    } // namespace gl
  } // namespace rix
} // namespace dp


