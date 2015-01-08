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


#include <dp/sg/xbar/SceneTree.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceTexture.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceManager.h>
#include <dp/sg/renderer/rix/gl/inc/Utility.h>
#include <dp/rix/gl/inc/DataTypeConversionGL.h>
#include <dp/rix/gl/RiXGL.h>
#include <dp/sg/gl/TextureGL.h>
#include <dp/sg/core/Buffer.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/sg/core/TextureFile.h>
#include <dp/sg/io/IO.h>

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

          ResourceTextureSharedPtr ResourceTexture::get( const dp::sg::core::TextureSharedPtr& texture, const ResourceManagerSharedPtr& resourceManager )
          {
            DP_ASSERT( texture );
            DP_ASSERT( !!resourceManager );

            ResourceTextureSharedPtr resourceTexture = resourceManager->getResource<ResourceTexture>( reinterpret_cast<size_t>(texture.getWeakPtr()) );
            if ( !resourceTexture )
            {
              resourceTexture = std::shared_ptr<ResourceTexture>( new ResourceTexture( texture, resourceManager ) );
            }
    
            return resourceTexture;
          }

          inline dp::util::PixelFormat getRiXPixelFormat( dp::sg::core::Image::PixelFormat pixelFormat )
          {
            switch ( pixelFormat )
            {
            case dp::sg::core::Image::IMG_RGB:
              return dp::util::PF_RGB;
            case dp::sg::core::Image::IMG_RGBA:
              return dp::util::PF_RGBA;
            case dp::sg::core::Image::IMG_BGR:
              return dp::util::PF_BGR;
            case dp::sg::core::Image::IMG_BGRA:
              return dp::util::PF_BGRA;
            case dp::sg::core::Image::IMG_LUMINANCE:
              return dp::util::PF_LUMINANCE;
            case dp::sg::core::Image::IMG_ALPHA:
              return dp::util::PF_ALPHA;
            case dp::sg::core::Image::IMG_LUMINANCE_ALPHA:
              return dp::util::PF_LUMINANCE_ALPHA;
            default:
              DP_ASSERT( !"unknown pixel format, assuming IMG_RGB");
              return dp::util::PF_RGB;
            }
          }

          dp::rix::core::TextureType getRiXTextureType( dp::sg::core::TextureSharedPtr const& texture )
          {
            dp::rix::core::TextureType tt;
            switch ( texture->getTextureTarget() )
            {
            case dp::sg::core::TT_TEXTURE_1D:
              tt = dp::rix::core::TT_1D;
              break;

            case dp::sg::core::TT_TEXTURE_1D_ARRAY:
              tt =dp::rix::core::TT_1D_ARRAY;
              break;

            case dp::sg::core::TT_TEXTURE_2D:
              tt =dp::rix::core::TT_2D;
              break;

            case dp::sg::core::TT_TEXTURE_RECTANGLE:
              tt =dp::rix::core::TT_2D_RECTANGLE;
              break;

            case dp::sg::core::TT_TEXTURE_2D_ARRAY:
              tt =dp::rix::core::TT_2D_ARRAY;
              break;

            case dp::sg::core::TT_TEXTURE_3D:
              tt =dp::rix::core::TT_3D;
              break;

            case dp::sg::core::TT_TEXTURE_CUBE:
              tt =dp::rix::core::TT_CUBEMAP;
              break;

            case dp::sg::core::TT_TEXTURE_CUBE_ARRAY:
              tt =dp::rix::core::TT_CUBEMAP_ARRAY;
              break;

            case dp::sg::core::TT_TEXTURE_BUFFER:
              DP_ASSERT( !"texture type not yet supported" );
              return dp::rix::core::TT_NUM_TEXTURETYPES;

            default:
              DP_ASSERT( !"unknwon texture type");
              return dp::rix::core::TT_NUM_TEXTURETYPES;
            }

            return tt;
          }

          dp::rix::core::TextureSharedHandle ResourceTexture::getRiXTexture( dp::sg::core::TextureHostSharedPtr const& texture )
          {
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();

            // TODO assumes that all images have the same datatype
      
            dp::rix::core::TextureType tt;
            size_t width = 0;
            size_t height = 0;
            size_t depth = 0;
            size_t layers = 0;

            switch( texture->getTextureTarget() )
            {
            case dp::sg::core::TT_TEXTURE_1D:
              tt = dp::rix::core::TT_1D;
              width  = texture->getWidth();
              break;

            case dp::sg::core::TT_TEXTURE_1D_ARRAY:
              tt = dp::rix::core::TT_1D_ARRAY;
              width  = texture->getWidth();
              layers = texture->getNumberOfImages();
              break;

            case dp::sg::core::TT_TEXTURE_2D:
              tt = dp::rix::core::TT_2D;
              width  = texture->getWidth();
              height = texture->getHeight();
              break;

            case dp::sg::core::TT_TEXTURE_RECTANGLE:
              tt = dp::rix::core::TT_2D_RECTANGLE;
              width  = texture->getWidth();
              height = texture->getHeight();
              break;

            case dp::sg::core::TT_TEXTURE_2D_ARRAY:
              tt = dp::rix::core::TT_2D_ARRAY;
              width  = texture->getWidth();
              height = texture->getHeight();
              layers = texture->getNumberOfImages();
              break;

            case dp::sg::core::TT_TEXTURE_3D:
              tt = dp::rix::core::TT_3D;
              width  = texture->getWidth();
              height = texture->getHeight();
              depth  = texture->getDepth();
              break;

            case dp::sg::core::TT_TEXTURE_CUBE:
              tt = dp::rix::core::TT_CUBEMAP;
              width  = texture->getWidth();
              height = texture->getHeight();
              break;

            case dp::sg::core::TT_TEXTURE_CUBE_ARRAY:
              tt = dp::rix::core::TT_CUBEMAP_ARRAY;
              width  = texture->getWidth();
              height = texture->getHeight();
              layers = texture->getNumberOfImages() / 6;  // !
              break;

            case dp::sg::core::TT_TEXTURE_BUFFER:
              DP_ASSERT( !"texture type not yet supported" );
              return nullptr;

            default:
              DP_ASSERT( !"unknwon texture type");
              return nullptr;
            }
      
            // use SceniX capability to determine the best internal GL format and pass it in as GL format  
            dp::sg::gl::GLTexImageFmt texFormat;
            // get the user-provided requested GPU format
            dp::sg::core::TextureHost::TextureGPUFormat gpuFormat = texture->getTextureGPUFormat();
            // receive the GL texture format description here
            if ( !dp::sg::gl::TextureGL::getTexImageFmt(texFormat, texture->getFormat(), texture->getType(), gpuFormat) )
            {
              DP_ASSERT( !"ERROR: could not fetch valid texture format!\n");
              return nullptr;
            }

            bool mipMaps = texture->isMipmapRequired();
            dp::rix::gl::TextureDescriptionGL td( tt, dp::rix::core::ITF_NATIVE, getRiXPixelFormat( texture->getFormat() ), getRiXDataType( texture->getType() ), width, height, depth, layers, mipMaps );
            td.m_internalFormatGL = texFormat.intFmt;

            return renderer->textureCreate( td );
          }

          void ResourceTexture::updateRiXTexture( const dp::rix::core::TextureSharedHandle& rixTexture, dp::sg::core::TextureHostSharedPtr const& texture )
          {
            std::vector<dp::sg::core::Buffer::DataReadLock> buffers; // Keep these locks until textureSetData() is done with the input pointers.
            std::vector<void const *> data;

            dp::util::PixelFormat pixelFormat = getRiXPixelFormat( texture->getFormat() );
            dp::util::DataType pixelDataType = getRiXDataType( texture->getType() );

            for ( unsigned int img = 0; img < texture->getNumberOfImages(); ++img )
            {
              for ( unsigned int lod = 0; lod < texture->getNumberOfMipmaps() + 1; ++lod ) // + 1 for LOD 0.
              {
                buffers.push_back( dp::sg::core::Buffer::DataReadLock( texture->getPixels(img, lod) ) );
                data.push_back( buffers.back().getPtr() );
              }
            }

            // getNumberOfMipmaps() does NOT include LOD 0 in SceniX!
            // RiX generates mipmaps when numMipMapLevels == 0 and the Texture 
            unsigned int numMipMapLevels = texture->getNumberOfMipmaps();
            // If mipmaps are provided include the LOD 0 in the count for RiX.
            if (numMipMapLevels)
            {
              numMipMapLevels++;
            }

            dp::rix::core::TextureDataPtr dataPtr( &data[0], numMipMapLevels, texture->getNumberOfImages(), pixelFormat, pixelDataType );
            m_resourceManager->getRenderer()->textureSetData( m_textureHandle, dataPtr );
          }

          ResourceTexture::ResourceTexture( const dp::sg::core::TextureSharedPtr& texture, const ResourceManagerSharedPtr& resourceManager )
            : ResourceManager::Resource( reinterpret_cast<size_t>( texture.getWeakPtr() ), resourceManager )
            , m_texture( texture )
            , m_isNativeTexture( false )
          {
            resourceManager->subscribe( this );
            update();
          }

          ResourceTexture::~ResourceTexture()
          {
            if ( m_resourceManager )
            {
              m_resourceManager->unsubscribe( this );
            }
          }

          const dp::sg::core::HandledObjectSharedPtr& ResourceTexture::getHandledObject() const
          {
            return m_texture.inplaceCast<dp::sg::core::HandledObject>();
          }

          void ResourceTexture::update()
          {
            if( m_isNativeTexture )
            {
              return;
            }

            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();

            if ( m_texture.isPtrTo<dp::sg::core::TextureFile>() )
            {
              // It's a file texture. Generate a TextureHost out of the TextureFile and upload it.
              dp::sg::core::TextureFileSharedPtr const& textureFile = m_texture.staticCast<dp::sg::core::TextureFile>();
              dp::sg::core::TextureHostSharedPtr textureHost = dp::sg::io::loadTextureHost( textureFile->getFilename() );
              if ( textureHost )
              {
                textureHost->setTextureTarget( textureFile->getTextureTarget() );

                // If TextureFile required mipmaps increase the mipmap count of the loaded TextureHost so that it
                // requires mipmaps too.
                if ( textureFile->isMipmapRequired() )
                {
                  textureHost->incrementMipmapUseCount();
                }

                m_textureHandle = getRiXTexture( textureHost );
                updateRiXTexture( m_textureHandle, textureHost );
              }
            }

            // TODO buffers are not being observed
            else if ( m_texture.isPtrTo<dp::sg::core::TextureHost>() )
            {
              dp::sg::core::TextureHostSharedPtr const& textureHost = m_texture.staticCast<dp::sg::core::TextureHost>();
              m_textureHandle = getRiXTexture( textureHost );
              updateRiXTexture( m_textureHandle, textureHost );
            }
            else if ( m_texture.isPtrTo<dp::sg::gl::TextureGL>() )
            {
              dp::sg::gl::TextureGLSharedPtr const& textureGL = m_texture.staticCast<dp::sg::gl::TextureGL>();
              dp::rix::gl::TextureDescriptionGL td( getRiXTextureType(m_texture), dp::rix::core::ITF_NATIVE, dp::rix::gl::getDPPixelFormat( textureGL->getTexture()->getFormat() )
                                                  , dp::rix::gl::getDPDataType( textureGL->getTexture()->getType() ) );
              m_textureHandle = renderer->textureCreate( td );
              renderer->textureSetData( m_textureHandle, dp::rix::gl::TextureDataGLTexture( textureGL->getTexture() ) );
            }
            else
            {
              DP_ASSERT( 0 && "unknown texture type");
            }

            // TODO: extend for TextureSharedPtr support
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp

