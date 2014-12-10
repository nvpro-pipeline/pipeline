// Copyright NVIDIA Corporation 2014
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


#include <dp/gl/Texture.h>
#include <dp/gl/inc/TextureBind.h>
#include <dp/gl/inc/TextureDSA.h>

namespace dp
{
  namespace gl
  {
    namespace
    {

      inline ITexture * getGLInterface()
      {
        static TextureBind  glInterfaceBind;
        static TextureDSA   glInterfaceDSA;
        static ITexture* glInterface = GLEW_EXT_direct_state_access ? static_cast<ITexture*>(&glInterfaceDSA) : static_cast<ITexture*>(&glInterfaceBind);
        return( glInterface );
      }

      // Wrapper class for nD uploads for textures.
      // It can handle TexSubImage and TexImage as well as compressed data
      class TexGLTransfer
      {

      public:
        TexGLTransfer( GLenum texture, GLenum target, GLint level, GLenum internalFormat, GLenum dataFormat, GLenum dataType
                     , const void  *dataPtr, GLsizei width, GLsizei height = 1,  GLsizei depth = 1, GLsizei dataSize = 0 );

        typedef void (TexGLTransfer::*PNFTEXGLUPLOAD)();


        GLenum m_target;
        GLint  m_level;
        GLenum m_internalFormat;
        GLenum m_dataFormat;
        GLenum m_dataType;
        const void  *m_dataPtr;

        GLsizei m_width;
        GLsizei m_height;
        GLsizei m_depth;

        size_t  m_dataSize;
    
        GLsizei m_xOffset;
        GLsizei m_yOffset;
        GLsizei m_zOffset;


        void Create1D();
        void Create2D();
        void Create3D();
        void Update1D();
        void Update2D();
        void Update3D();

        void Create1DCompressed();
        void Create2DCompressed();
        void Create3DCompressed();
        void Update1DCompressed();
        void Update2DCompressed();
        void Update3DCompressed();

        // uses "compressed" path when m_dataSize is != 0
        void doTransfer(int dimension, bool update );

        static PNFTEXGLUPLOAD getTransferFunction(int dimension, bool update, bool compressed );

      private:
        GLenum  m_texture;
      };
      typedef TexGLTransfer::PNFTEXGLUPLOAD TexGLUploadFunc;
  

      TexGLTransfer::TexGLTransfer( GLenum texture, GLenum target, GLint level, GLenum internalFormat, GLenum dataFormat, GLenum dataType,
                                    const void *dataPtr,
                                    GLsizei width, GLsizei height,  GLsizei depth, GLsizei dataSize ) 
       : m_texture(texture)
       , m_target(target)
       , m_level(level)
       , m_internalFormat(internalFormat)
       , m_dataFormat(dataFormat)
       , m_dataType(dataType)
       , m_dataPtr(dataPtr)
       , m_width(width)
       , m_height(height)
       , m_depth(depth)
       , m_dataSize(dataSize)
       , m_xOffset(0)
       , m_yOffset(0)
       , m_zOffset(0)
      {
      }

      void TexGLTransfer::Create1D()
      {
        getGLInterface()->setImage1D( m_texture, m_target, m_level, m_internalFormat, m_width, 0, m_dataFormat, m_dataType, m_dataPtr );
      }

      void TexGLTransfer::Create2D()
      {
        getGLInterface()->setImage2D( m_texture, m_target, m_level, m_internalFormat, m_width, m_height, 0, m_dataFormat, m_dataType, m_dataPtr );
      }

      void TexGLTransfer::Create3D()
      {
        getGLInterface()->setImage3D( m_texture, m_target, m_level, m_internalFormat, m_width, m_height, m_depth, 0, m_dataFormat, m_dataType, m_dataPtr );
      }

      void TexGLTransfer::Update1D()
      {
        getGLInterface()->setSubImage1D( m_texture, m_target, m_level, m_xOffset, m_width, m_dataFormat, m_dataType, m_dataPtr );
      }

      void TexGLTransfer::Update2D()
      {
        getGLInterface()->setSubImage2D( m_texture, m_target, m_level, m_xOffset, m_yOffset, m_width, m_height, m_dataFormat, m_dataType, m_dataPtr );
      }

      void TexGLTransfer::Update3D()
      {
        getGLInterface()->setSubImage3D( m_texture, m_target, m_level, m_xOffset, m_yOffset, m_zOffset, m_width, m_height, m_depth, m_dataFormat, m_dataType, m_dataPtr );
      }

      void TexGLTransfer::Create1DCompressed()
      {
        getGLInterface()->setCompressedImage1D( m_texture, m_target, m_level, m_internalFormat, m_width, 0, (GLsizei)m_dataSize, m_dataPtr );
      }

      void TexGLTransfer::Create2DCompressed()
      {
        getGLInterface()->setCompressedImage2D( m_texture, m_target, m_level, m_internalFormat, m_width, m_height, 0, (GLsizei)m_dataSize, m_dataPtr );
      }

      void TexGLTransfer::Create3DCompressed()
      {
        getGLInterface()->setCompressedImage3D( m_texture, m_target, m_level, m_internalFormat, m_width, m_height, m_depth, 0, (GLsizei)m_dataSize, m_dataPtr );
      }

      void TexGLTransfer::Update1DCompressed()
      {
        getGLInterface()->setCompressedSubImage1D( m_texture, m_target, m_level, m_xOffset, m_width, m_internalFormat, (GLsizei)m_dataSize, m_dataPtr );
      }

      void TexGLTransfer::Update2DCompressed()
      {
        getGLInterface()->setCompressedSubImage2D( m_texture, m_target, m_level, m_xOffset, m_yOffset, m_width, m_height, m_internalFormat, (GLsizei)m_dataSize, m_dataPtr );
      }

      void TexGLTransfer::Update3DCompressed()
      {
        getGLInterface()->setCompressedSubImage3D( m_texture, m_target, m_level, m_xOffset, m_yOffset, m_zOffset, m_width, m_height, m_depth, m_internalFormat, (GLsizei)m_dataSize, m_dataPtr );
      }

      TexGLTransfer::PNFTEXGLUPLOAD TexGLTransfer::getTransferFunction( int dimension, bool update, bool compressed )
      {
        DP_ASSERT( 1 <= dimension && dimension <= 3 );
        static const TexGLTransfer::PNFTEXGLUPLOAD funcs[] = {
          &TexGLTransfer::Create1D,
          &TexGLTransfer::Create2D,
          &TexGLTransfer::Create3D,
          &TexGLTransfer::Update1D,
          &TexGLTransfer::Update2D,
          &TexGLTransfer::Update3D,
          &TexGLTransfer::Create1DCompressed,
          &TexGLTransfer::Create2DCompressed,
          &TexGLTransfer::Create3DCompressed,
          &TexGLTransfer::Update1DCompressed,
          &TexGLTransfer::Update2DCompressed,
          &TexGLTransfer::Update3DCompressed,
        };

        return funcs[ (compressed ? 6 : 0)  + (update ? 3 : 0) + (dimension - 1) ];
      }

      inline void TexGLTransfer::doTransfer( int dimension, bool update )
      {
        ((*this).*getTransferFunction( dimension, update, m_dataSize != 0 ))();
      }

      inline unsigned int numberOfMipmaps(unsigned int w, unsigned int h, unsigned int d)
      {
        unsigned int bits = dp::math::max( w, h, d );
        unsigned int i=1;
        while (bits >>= 1) 
        {
          ++i;
        }
        return i;
      }

      bool isCompressedSizeKnown( GLenum format )
      {
        switch (format)
        {
        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:

        case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT:
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT:
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:

        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_ARB:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT_ARB:
        case GL_COMPRESSED_RGBA_BPTC_UNORM_ARB:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM_ARB:

        case GL_COMPRESSED_RED_RGTC1:
        case GL_COMPRESSED_SIGNED_RED_RGTC1:
        case GL_COMPRESSED_RG_RGTC2:
        case GL_COMPRESSED_SIGNED_RG_RGTC2:

        case GL_COMPRESSED_LUMINANCE_LATC1_EXT:
        case GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT:
        case GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT:
        case GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT:
          return true;
        }
        return false;
      }

      void getTexImage( const Texture *texGL, GLenum target, void *data, GLuint mipLevel )
      {
        DP_ASSERT( texGL );
        DP_ASSERT( texGL->getInternalFormat() != ~0 );

        if ( isCompressedFormat( texGL->getFormat() ) )
        {
          getGLInterface()->getCompressedImage( texGL->getGLId(), target, mipLevel, data );
        }
        else
        {
          getGLInterface()->getImage( texGL->getGLId(), target, mipLevel, texGL->getFormat(), texGL->getType(), data );
        }
      }

      GLsizei getMipMapSize( GLsizei sz, GLuint level )
      {
        sz = (sz >> level);
        return sz ? sz : 1;
      }

      bool sizeValid( GLenum target, GLsizei w, GLsizei h, GLsizei d, GLsizei layers )
      {
        // init to zero so comparison at bottom will fail on unknown texture
        GLsizei maxw = 0;
        GLsizei maxh = 0;
        GLsizei maxd = 0;
        GLsizei maxl = 0;
        bool powerOfTwoCheck = true;

        switch( target )
        {
          case GL_TEXTURE_1D: 
            maxw = Texture1D::getMaximumSize();
            maxh = 1;
            maxd = 1;
            maxl = 0;
            break;

          case GL_TEXTURE_2D: 
            maxw = Texture2D::getMaximumSize();
            maxh = maxw;
            maxd = 1;
            maxl = 0;
            break;

          case GL_TEXTURE_RECTANGLE_ARB:
            maxw = TextureRectangle::getMaximumSize();
            maxh = maxw;
            maxd = 1;
            powerOfTwoCheck = false;
            maxl = 0;
            break;

          case GL_TEXTURE_3D:
            maxw = Texture3D::getMaximumSize();
            maxh = maxw;
            maxd = maxh;
            maxl = 0;
            break;

          case GL_TEXTURE_CUBE_MAP: 
            maxw = TextureCubemap::getMaximumSize();
            maxh = maxw;
            maxd = 1;
            maxl = 0;
            break;

          case GL_TEXTURE_1D_ARRAY_EXT:
            maxw = Texture1DArray::getMaximumSize();
            maxh = 1;
            maxd = 1;
            maxl = Texture1DArray::getMaximumLayers();
            if (!layers)
            {
              return false;
            }
            break;

          case GL_TEXTURE_2D_ARRAY_EXT: 
            maxw = Texture2DArray::getMaximumSize();
            maxh = maxw;
            maxd = 1;
            maxl = Texture2DArray::getMaximumLayers();
            if (!layers)
            {
              return false;
            }
            break;

          case GL_TEXTURE_CUBE_MAP_ARRAY: 
            maxw = TextureCubemapArray::getMaximumSize();
            maxh = maxw;
            maxd = 1;
            maxl = TextureCubemapArray::getMaximumLayers();
            if (!layers || (layers % 6 != 0))
            {
              return false;
            }
            break;

          case GL_TEXTURE_BUFFER:
            maxw = TextureBuffer::getMaximumSize();
            maxh = 1;
            maxd = 1;
            maxl = 0;
            break;

          case GL_TEXTURE_2D_MULTISAMPLE:
            maxw = Texture2DMultisample::getMaximumSize();
            maxh = maxw;
            maxd = 1;
            maxl = 0;
            break;

          case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            maxw = Texture2DMultisampleArray::getMaximumSize();
            maxh = maxw;
            maxd = 1;
            maxl = Texture2DMultisampleArray::getMaximumLayers();
            break;

          default:
            DP_ASSERT( !"texture target unknown!" );
            break;
        }

        // either we dont require a check, or we are only valid
        // when extension exists, or we are power of two
        powerOfTwoCheck = powerOfTwoCheck ? 
          (!!GLEW_ARB_texture_non_power_of_two || 
           (dp::math::isPowerOfTwo(w) && dp::math::isPowerOfTwo(h) && dp::math::isPowerOfTwo(d))) : true;


        return ( w <= maxw && h <= maxh && d <= maxd && layers <= maxl) && powerOfTwoCheck;
      }
    } // namespace

    size_t getCompressedSize( GLenum format, GLsizei w, GLsizei h, GLsizei d, GLsizei layers /*= 0 */ )
    {
      // http://www.opengl.org/registry/specs/NV/texture_compression_vtc.txt
      // blocks = ceil(w/4) * ceil(h/4) * d;

      layers = std::max( 1, layers );

      size_t blocks = ( (w + 3) / 4) * ( (h + 3) / 4 ) * d;

      // written against OpenGL 4.1 compatibility profile

      size_t bytesPerBlock = 0;
      switch ( format )
      {
        case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:

        case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT:

        case GL_COMPRESSED_LUMINANCE_LATC1_EXT:
        case GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT:
        case GL_COMPRESSED_RED_RGTC1:
        case GL_COMPRESSED_SIGNED_RED_RGTC1:
          // these formats use 8 bytes per block
          bytesPerBlock = 8;
          break;

        case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT:
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:

        case GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT:
        case GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT:
        case GL_COMPRESSED_RG_RGTC2:
        case GL_COMPRESSED_SIGNED_RG_RGTC2:

        case GL_COMPRESSED_RGBA_BPTC_UNORM_ARB:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM_ARB:

        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_ARB:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT_ARB:
          // these formats use 16 bytes per block
          bytesPerBlock = 16;
          break;
      }

      return( bytesPerBlock * blocks * layers );
    }

    size_t getImageDataSize( GLenum format, GLenum type, GLsizei w, GLsizei h, GLsizei d, GLsizei layers /*= 0 */ )
    {
      size_t size = getCompressedSize( format, w, h, d, layers );
      if ( size )
      {
        return size;
      }

      // written against OpenGL 4.1 compatibility profile

      size_t formatSize = 0;
      bool isIntegerFormat = false;
      switch ( format )
      {
        case GL_RED_INTEGER:
        case GL_GREEN_INTEGER:
        case GL_BLUE_INTEGER:
        case GL_ALPHA_INTEGER:
          isIntegerFormat = true;
        case GL_RED:
        case GL_GREEN:
        case GL_BLUE:
        case GL_ALPHA:
        case GL_LUMINANCE:
          formatSize = 1;
          break;

        case GL_RG_INTEGER:
          isIntegerFormat = true;
        case GL_RG:
        case GL_LUMINANCE_ALPHA:
        case GL_DEPTH_STENCIL:
          formatSize = 2;
          break;

        case GL_RGB_INTEGER:
        case GL_BGR_INTEGER:
          isIntegerFormat = true;
        case GL_RGB:
        case GL_BGR:
          formatSize = 3;
          break;

        case GL_RGBA_INTEGER:
        case GL_BGRA_INTEGER:
          isIntegerFormat = true;
        case GL_RGBA:
        case GL_BGRA:
          formatSize = 4;
          break;
      }

      size_t typeSize = 0;
      switch( type )
      {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
          typeSize = 1;
          break;

        case GL_HALF_FLOAT:
          DP_ASSERT( !isIntegerFormat );
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
          typeSize = 2;
          break;

        case GL_FLOAT:
          DP_ASSERT( !isIntegerFormat );
        case GL_UNSIGNED_INT:
        case GL_INT:
          typeSize = 4;
          break;

        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
          DP_ASSERT( format == GL_RGB || format == GL_RGB_INTEGER );
          typeSize = 1;
          formatSize = 1;
          break;

        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
          DP_ASSERT( formatSize == 4 );
          typeSize = 2;
          formatSize = 1;
          break;

        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
          DP_ASSERT( formatSize == 4 );
          typeSize = 4;
          formatSize = 1;
          break;

        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
          DP_ASSERT( format == GL_RGB );
          DP_ASSERT( !isIntegerFormat );
          typeSize = 4;
          formatSize = 1;
          break;

        case GL_UNSIGNED_INT_24_8:
          DP_ASSERT( format == GL_DEPTH_STENCIL );
          typeSize = 4;
          formatSize = 1;
          break;

        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
          DP_ASSERT( format == GL_DEPTH_STENCIL );
          typeSize = 8;
          formatSize = 1;
          break;
      }

      layers = std::max( 1, layers );

      return( w * h * d * layers * formatSize * typeSize );
    }

    GLenum getTargetForSamplerType( GLenum samplerType )
    {
      switch ( samplerType )
      {
        case GL_SAMPLER_1D:
        case GL_SAMPLER_1D_SHADOW:
        case GL_INT_SAMPLER_1D:
        case GL_UNSIGNED_INT_SAMPLER_1D:
          return GL_TEXTURE_1D;

        case GL_SAMPLER_2D:
        case GL_SAMPLER_2D_SHADOW:
        case GL_INT_SAMPLER_2D:
        case GL_UNSIGNED_INT_SAMPLER_2D:
          return GL_TEXTURE_2D;

        case GL_SAMPLER_3D:
        case GL_INT_SAMPLER_3D:
        case GL_UNSIGNED_INT_SAMPLER_3D:
          return GL_TEXTURE_3D;

        case GL_SAMPLER_CUBE:
        case GL_SAMPLER_CUBE_SHADOW:
        case GL_UNSIGNED_INT_SAMPLER_CUBE:
        case GL_INT_SAMPLER_CUBE:
          return GL_TEXTURE_CUBE_MAP;

        case GL_SAMPLER_1D_ARRAY:
        case GL_SAMPLER_1D_ARRAY_SHADOW:
        case GL_INT_SAMPLER_1D_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:
          return GL_TEXTURE_1D_ARRAY;

        case GL_SAMPLER_2D_ARRAY:
        case GL_SAMPLER_2D_ARRAY_SHADOW:
        case GL_INT_SAMPLER_2D_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
          return GL_TEXTURE_2D_ARRAY;

        case GL_SAMPLER_2D_MULTISAMPLE:
        case GL_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
          return GL_TEXTURE_2D_MULTISAMPLE;

        case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
          return GL_TEXTURE_2D_MULTISAMPLE_ARRAY;

        case GL_SAMPLER_BUFFER:
        case GL_INT_SAMPLER_BUFFER:
        case GL_UNSIGNED_INT_SAMPLER_BUFFER:
          return GL_TEXTURE_BUFFER;

        case GL_SAMPLER_2D_RECT:
        case GL_SAMPLER_2D_RECT_SHADOW :
        case GL_INT_SAMPLER_2D_RECT:
        case GL_UNSIGNED_INT_SAMPLER_2D_RECT:
          return GL_TEXTURE_RECTANGLE;

        default:
          DP_ASSERT(!"please support me");
      }
      return GL_NONE;
    }

    bool isCompressedFormat( GLenum format )
    {
      bool ok = false;
      switch ( format )
      {
        case GL_COMPRESSED_RG :
        case GL_COMPRESSED_RGB :
        case GL_COMPRESSED_RGBA :
        case GL_COMPRESSED_ALPHA :
        case GL_COMPRESSED_LUMINANCE :
        case GL_COMPRESSED_LUMINANCE_ALPHA :
        case GL_COMPRESSED_INTENSITY :
        case GL_COMPRESSED_SLUMINANCE :
        case GL_COMPRESSED_SLUMINANCE_ALPHA :
        case GL_COMPRESSED_SRGB :
        case GL_COMPRESSED_SRGB_ALPHA :

        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT :
        case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT :
        case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT :
        case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT :

        case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT :
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT :
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT :
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT :

        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_ARB :
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT_ARB :
        case GL_COMPRESSED_RGBA_BPTC_UNORM_ARB :
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM_ARB :

        case GL_COMPRESSED_RED_RGTC1 :
        case GL_COMPRESSED_SIGNED_RED_RGTC1 :
        case GL_COMPRESSED_RG_RGTC2 :
        case GL_COMPRESSED_SIGNED_RG_RGTC2 :

        case GL_COMPRESSED_LUMINANCE_LATC1_EXT :
        case GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT :
        case GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT :
        case GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT :
          ok = true;
          break;
      }
      return( ok );
    }

    bool isIntegerInternalFormat( GLenum format )
    {
      bool ok = false;
      switch (format)
      {
        // http://www.opengl.org/registry/specs/ARB/texture_rg.txt
        case GL_R8I    :
        case GL_R8UI   :
        case GL_R16I   :
        case GL_R16UI  :
        case GL_R32I   :
        case GL_R32UI  :
        case GL_RG8I   :
        case GL_RG8UI  :
        case GL_RG16I  :
        case GL_RG16UI :
        case GL_RG32I  :
        case GL_RG32UI :

        // http://www.opengl.org/registry/specs/EXT/texture_integer.txt
        case GL_RGBA32UI_EXT           :
        case GL_RGB32UI_EXT            :
        case GL_ALPHA32UI_EXT          :
        case GL_INTENSITY32UI_EXT      :
        case GL_LUMINANCE32UI_EXT      :
        case GL_LUMINANCE_ALPHA32UI_EXT:

        case GL_RGBA16UI_EXT           :
        case GL_RGB16UI_EXT            :
        case GL_ALPHA16UI_EXT          :
        case GL_INTENSITY16UI_EXT      :
        case GL_LUMINANCE16UI_EXT      :
        case GL_LUMINANCE_ALPHA16UI_EXT:

        case GL_RGBA8UI_EXT            :
        case GL_RGB8UI_EXT             :
        case GL_ALPHA8UI_EXT           :
        case GL_INTENSITY8UI_EXT       :
        case GL_LUMINANCE8UI_EXT       :
        case GL_LUMINANCE_ALPHA8UI_EXT :

        case GL_RGBA32I_EXT            :
        case GL_RGB32I_EXT             :
        case GL_ALPHA32I_EXT           :
        case GL_INTENSITY32I_EXT       :
        case GL_LUMINANCE32I_EXT       :
        case GL_LUMINANCE_ALPHA32I_EXT :

        case GL_RGBA16I_EXT            :
        case GL_RGB16I_EXT             :
        case GL_ALPHA16I_EXT           :
        case GL_INTENSITY16I_EXT       :
        case GL_LUMINANCE16I_EXT       :
        case GL_LUMINANCE_ALPHA16I_EXT :

        case GL_RGBA8I_EXT             :
        case GL_RGB8I_EXT              :
        case GL_ALPHA8I_EXT            :
        case GL_INTENSITY8I_EXT        :
        case GL_LUMINANCE8I_EXT        :
        case GL_LUMINANCE_ALPHA8I_EXT  :
          ok = true;
          break;
      }
      return( ok );
    }

    bool isValidInternalFormat( GLenum format )
    {
      bool ok = false;
      switch( format )
      {
        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
          ok = !!GLEW_ARB_texture_buffer_object_rgb32;
          break;

        case GL_R8:
        case GL_R16:
        case GL_R16F:
        case GL_R32F:
        case GL_R8I:
        case GL_R16I:
        case GL_R32I:
        case GL_R8UI:
        case GL_R16UI:
        case GL_R32UI:

        case GL_RG8:
        case GL_RG16:
        case GL_RG16F:
        case GL_RG32F:
        case GL_RG8I:
        case GL_RG16I:
        case GL_RG32I:
        case GL_RG8UI:
        case GL_RG16UI:
        case GL_RG32UI:
          ok = !!GLEW_ARB_texture_rg;
          break;

        case GL_ALPHA8:
        case GL_ALPHA16:
        case GL_ALPHA16F_ARB:
        case GL_ALPHA32F_ARB:
        case GL_ALPHA8I_EXT:
        case GL_ALPHA16I_EXT:
        case GL_ALPHA32I_EXT:
        case GL_ALPHA8UI_EXT:
        case GL_ALPHA16UI_EXT:
        case GL_ALPHA32UI_EXT:

        case GL_LUMINANCE8:
        case GL_LUMINANCE16:
        case GL_LUMINANCE16F_ARB:
        case GL_LUMINANCE32F_ARB:
        case GL_LUMINANCE8I_EXT:
        case GL_LUMINANCE16I_EXT:
        case GL_LUMINANCE32I_EXT:
        case GL_LUMINANCE8UI_EXT:
        case GL_LUMINANCE16UI_EXT:
        case GL_LUMINANCE32UI_EXT:

        case GL_LUMINANCE8_ALPHA8:
        case GL_LUMINANCE16_ALPHA16:
        case GL_LUMINANCE_ALPHA16F_ARB:
        case GL_LUMINANCE_ALPHA32F_ARB:
        case GL_LUMINANCE_ALPHA8I_EXT:
        case GL_LUMINANCE_ALPHA16I_EXT:
        case GL_LUMINANCE_ALPHA32I_EXT:
        case GL_LUMINANCE_ALPHA8UI_EXT:
        case GL_LUMINANCE_ALPHA16UI_EXT:
        case GL_LUMINANCE_ALPHA32UI_EXT:

        case GL_INTENSITY8:
        case GL_INTENSITY16:
        case GL_INTENSITY16F_ARB:
        case GL_INTENSITY32F_ARB:
        case GL_INTENSITY8I_EXT:
        case GL_INTENSITY16I_EXT:
        case GL_INTENSITY32I_EXT:
        case GL_INTENSITY8UI_EXT:
        case GL_INTENSITY16UI_EXT:
        case GL_INTENSITY32UI_EXT:

        case GL_RGBA8:
        case GL_RGBA16:
        case GL_RGBA16F_ARB:
        case GL_RGBA32F_ARB:
        case GL_RGBA8I_EXT:
        case GL_RGBA16I_EXT:
        case GL_RGBA32I_EXT:
        case GL_RGBA8UI_EXT:
        case GL_RGBA16UI_EXT:
        case GL_RGBA32UI_EXT:
          ok = true;
          break;

        default:
          DP_ASSERT( !"unknown internal Format encountered!" );
          break;
      }
      return( ok );
    }

    bool isLayeredTarget( GLenum target )
    {
      bool ok = false;
      switch( target )
      {
        case GL_TEXTURE_1D_ARRAY_EXT :
        case GL_TEXTURE_2D_ARRAY_EXT :
        case GL_TEXTURE_CUBE_MAP_ARRAY :
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY :
          ok = true;
          break;
      }
      return( ok );
    }

    bool isImageType( GLenum type )
    {
      bool ok = false;
      switch( type )
      {
        case GL_IMAGE_1D :
        case GL_IMAGE_2D :
        case GL_IMAGE_3D :
        case GL_IMAGE_2D_RECT :
        case GL_IMAGE_CUBE :
        case GL_IMAGE_BUFFER :
        case GL_IMAGE_1D_ARRAY :
        case GL_IMAGE_2D_ARRAY :
        case GL_IMAGE_CUBE_MAP_ARRAY :
        case GL_IMAGE_2D_MULTISAMPLE :
        case GL_IMAGE_2D_MULTISAMPLE_ARRAY :

        case GL_INT_IMAGE_1D :
        case GL_INT_IMAGE_2D :
        case GL_INT_IMAGE_3D :
        case GL_INT_IMAGE_2D_RECT :
        case GL_INT_IMAGE_CUBE :
        case GL_INT_IMAGE_BUFFER :
        case GL_INT_IMAGE_1D_ARRAY :
        case GL_INT_IMAGE_2D_ARRAY :
        case GL_INT_IMAGE_CUBE_MAP_ARRAY :
        case GL_INT_IMAGE_2D_MULTISAMPLE :
        case GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY :

        case GL_UNSIGNED_INT_IMAGE_1D :
        case GL_UNSIGNED_INT_IMAGE_2D :
        case GL_UNSIGNED_INT_IMAGE_3D :
        case GL_UNSIGNED_INT_IMAGE_2D_RECT :
        case GL_UNSIGNED_INT_IMAGE_CUBE :
        case GL_UNSIGNED_INT_IMAGE_BUFFER :
        case GL_UNSIGNED_INT_IMAGE_1D_ARRAY :
        case GL_UNSIGNED_INT_IMAGE_2D_ARRAY :
        case GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY :
        case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE :
        case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY :
          ok = true;
          break;
      }
      return( ok );
    }

    bool isSamplerType( GLenum type )
    {
      bool ok = false;
      switch( type )
      {
        case GL_SAMPLER_1D:
        case GL_SAMPLER_2D:
        case GL_SAMPLER_3D:
        case GL_SAMPLER_CUBE:
        case GL_SAMPLER_2D_RECT:
        case GL_SAMPLER_2D_MULTISAMPLE:
        case GL_SAMPLER_1D_ARRAY:
        case GL_SAMPLER_2D_ARRAY:
        case GL_SAMPLER_CUBE_MAP_ARRAY:
        case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_SAMPLER_BUFFER:

        case GL_INT_SAMPLER_1D:
        case GL_INT_SAMPLER_2D:
        case GL_INT_SAMPLER_3D:
        case GL_INT_SAMPLER_CUBE:
        case GL_INT_SAMPLER_2D_RECT:
        case GL_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_INT_SAMPLER_1D_ARRAY:
        case GL_INT_SAMPLER_2D_ARRAY:
        case GL_INT_SAMPLER_CUBE_MAP_ARRAY:
        case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_INT_SAMPLER_BUFFER:

        case GL_UNSIGNED_INT_SAMPLER_1D:
        case GL_UNSIGNED_INT_SAMPLER_2D:
        case GL_UNSIGNED_INT_SAMPLER_3D:
        case GL_UNSIGNED_INT_SAMPLER_CUBE:
        case GL_UNSIGNED_INT_SAMPLER_2D_RECT:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_BUFFER:

        case GL_SAMPLER_1D_SHADOW:
        case GL_SAMPLER_2D_SHADOW:
        case GL_SAMPLER_CUBE_SHADOW:
        case GL_SAMPLER_2D_RECT_SHADOW:
        case GL_SAMPLER_1D_ARRAY_SHADOW:
        case GL_SAMPLER_2D_ARRAY_SHADOW:
        case GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW:
          ok = true;
          break;
      }
      return( ok );
    }


    /***************/
    /* Texture     */
    /***************/

    Texture::Texture( GLenum target, GLenum internalFormat, GLenum format, GLenum type, GLsizei border )
      : m_target( target )
      , m_internalFormat( internalFormat )
      , m_format( format )
      , m_type( type )
      , m_definedLevels( 0 )
      , m_maxLevel( 0 )
    {
      GLuint id;
      glGenTextures( 1, &id );
      setGLId( id );

      GLenum filter = isIntegerInternalFormat( m_internalFormat ) ? GL_NEAREST : GL_LINEAR;
      switch (m_target)
      {
        case GL_TEXTURE_1D:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_3D:
        case GL_TEXTURE_RECTANGLE:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
          // default to linear/nearest filtering for those
          // as it will mean texture is always complete independent
          // of current mipmap state and internal format
          setFilterParameters( filter, filter );
          break;
        default:
          break;
      }
    }

    Texture::~Texture( )
    {
      if ( getGLId() )
      {
        if ( getShareGroup() )
        {
          DEFINE_PTR_TYPES( CleanupTask );
          class CleanupTask : public ShareGroupTask
          {
            public:
              static CleanupTaskSharedPtr create( GLuint id )
              {
                return( std::shared_ptr<CleanupTask>( new CleanupTask( id ) ) );
              }

              virtual void execute() { glDeleteTextures( 1, &m_id ); }

            protected:
              CleanupTask( GLuint id ) : m_id( id ) {}

            private:
              GLuint m_id;
          };

          // make destructor exception safe
          try
          {
            getShareGroup()->executeTask( CleanupTask::create( getGLId() ) );
          } catch (...) {}
        }
        else
        {
          GLuint id = getGLId();
          glDeleteTextures(1, &id );
        }
      }
    }

    void Texture::bind() const
    {
      glBindTexture( m_target, getGLId() );
    }

    void Texture::unbind() const
    {
      glBindTexture( m_target, 0 );
    }

    void Texture::generateMipMap()
    {
      DP_ASSERT( m_maxLevel );

      getGLInterface()->generateMipMap( getGLId(), m_target );
      m_definedLevels = (1 << (m_maxLevel + 1)) - 1;
    }

    void Texture::setBorderColor( float color[4] )
    {
      getGLInterface()->setParameter( getGLId(), m_target, GL_TEXTURE_BORDER_COLOR, color );
    }

    void Texture::setBorderColor( int color[4] )
    {
      getGLInterface()->setParameterUnmodified( getGLId(), m_target, GL_TEXTURE_BORDER_COLOR, color );
    }

    void Texture::setBorderColor( unsigned int color[4] )
    {
      getGLInterface()->setParameterUnmodified( getGLId(), m_target, GL_TEXTURE_BORDER_COLOR, color );
    }

    void Texture::setCompareParameters( GLenum mode, GLenum func )
    {
      getGLInterface()->setCompareParameters( getGLId(), m_target, mode, func );
    }

    void Texture::setFilterParameters( GLenum minFilter, GLenum magFilter )
    {
      getGLInterface()->setFilterParameters( getGLId(), m_target, minFilter, magFilter );
    }

    void Texture::setFormat( GLenum format )
    {
      if ( m_format != format )
      {
        m_format = format;
        m_definedLevels = 0;
      }
    }

    void Texture::setLODParameters( float minLOD, float maxLOD, float LODBias )
    {
      // must not set lod parameters on a texture rectangle (invalid operation)
      if ( m_target != GL_TEXTURE_RECTANGLE )
      {
        getGLInterface()->setLODParameters( getGLId(), m_target, minLOD, maxLOD, LODBias );
      }
    }

    void Texture::setMaxAnisotropy( float anisotropy )
    {
      getGLInterface()->setParameter( getGLId(), m_target, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy );
    }

    void Texture::setType( GLenum type )
    {
      if ( m_type != type )
      {
        m_type = type;
        m_definedLevels = 0;
      }
    }

    void Texture::setWrapParameters( GLenum wrapS, GLenum wrapT, GLenum wrapR )
    {
      getGLInterface()->setWrapParameters( getGLId(), m_target, wrapS, wrapT, wrapR );
    }

    void Texture::addDefinedLevel( GLuint level )
    {
      m_definedLevels |= ( 1 << level );
    }

    void Texture::resetDefinedLevels()
    {
      m_definedLevels = 1;
    }

    void Texture::setMaxLevel( GLuint level )
    {
      m_maxLevel = level;
    }


    /***************/
    /* Texture1D   */
    /***************/

    Texture1DSharedPtr Texture1D::create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width )
    {
      return( std::shared_ptr<Texture1D>( new Texture1D( internalFormat, format, type, width ) ) );
    }

    Texture1D::Texture1D( GLenum internalFormat, GLenum format, GLenum type, GLsizei width )
      : Texture( GL_TEXTURE_1D, internalFormat, format, type )
      , m_width( 0 )
    {
      resize( width );
    }

    void Texture1D::setData( const void *data, GLuint mipLevel /*= 0 */ )
    {
      DP_ASSERT( isMipMapLevelValid( mipLevel) );

      TexGLTransfer upload( getGLId(), getTarget(), mipLevel, getInternalFormat(), getFormat(), getType(), data, getMipMapSize(m_width, mipLevel));
      // set m_dataSize for compressed format
      upload.m_dataSize = getCompressedSize( getFormat(), upload.m_width, 1, 1 ); 
      // format must match if compressed
      DP_ASSERT( !upload.m_dataSize || getFormat() == getInternalFormat() );

      upload.doTransfer( 1, isMipMapLevelDefined(mipLevel) );

      addDefinedLevel( mipLevel );
    }

    void Texture1D::getData( void *data, GLuint mipLevel /*= 0 */ ) const
    {
      DP_ASSERT( isMipMapLevelDefined( mipLevel ) );
      DP_ASSERT( data );
      getTexImage( this, getTarget(), data, mipLevel );
    }

    void Texture1D::resize( GLsizei width )
    {
      if ( m_width != width )
      {
        DP_ASSERT( width <= getMaximumSize() );
        m_width = width;

        getGLInterface()->setImage1D( getGLId(), getTarget(), 0, getInternalFormat(), width, 0, getFormat(), getType(), nullptr );

        resetDefinedLevels();
        setMaxLevel( numberOfMipmaps( getWidth(), 1 , 1 ) );
      }
    }

    GLsizei Texture1D::getMaximumSize()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_TEXTURE_SIZE, &size );
      return size;
    }


    /********************/
    /* Texture1DArray   */
    /********************/

    Texture1DArraySharedPtr Texture1DArray::create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei layers )
    {
      return( std::shared_ptr<Texture1DArray>( new Texture1DArray( internalFormat, format, type, width, layers ) ) );
    }

    Texture1DArray::Texture1DArray( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei layers )
      : Texture( GL_TEXTURE_1D_ARRAY_EXT, internalFormat, format, type )
      , m_width( 0 )
      , m_layers( 0 )
    {
      resize( width, layers );
    }

    void Texture1DArray::setData( const void *data, GLint layer, GLuint mipLevel /*= 0 */ )
    {
      DP_ASSERT( isMipMapLevelValid( mipLevel) );
      DP_ASSERT( layer < getLayers() );

      TexGLTransfer upload( getGLId(), getTarget(), mipLevel, getInternalFormat(), getFormat(), getType(), data, getMipMapSize(m_width, mipLevel), 1);
      // set m_dataSize for compressed format
      upload.m_dataSize = getCompressedSize( getFormat(), upload.m_width, 1, 1 ); 
      // format must match if compressed
      DP_ASSERT( !upload.m_dataSize || getFormat() == getInternalFormat() ); 

      if ( !isMipMapLevelDefined(mipLevel) )
      {
        TexGLTransfer uploadCreate = upload;
        dp::gl::bind( GL_PIXEL_UNPACK_BUFFER, BufferSharedPtr::null );    // make sure, GL_PIXEL_UNPACK_BUFFER is unbound !
        uploadCreate.m_dataSize *= m_layers;
        uploadCreate.m_height = m_layers;
        uploadCreate.m_dataPtr = nullptr;
        uploadCreate.doTransfer( 2, false );
        addDefinedLevel( mipLevel );
      }

      upload.m_yOffset = layer;
      upload.doTransfer( 2, true );
    }

    void Texture1DArray::getData( void *data, GLuint mipLevel /*= 0 */ ) const
    {
      DP_ASSERT( isMipMapLevelDefined( mipLevel ) );
      DP_ASSERT( data );
      getTexImage( this, getTarget(), data, mipLevel );
    }

    void Texture1DArray::resize( GLsizei width, GLsizei layers )
    {
      if ( m_width != width || m_layers != layers)
      {
        DP_ASSERT( width <= getMaximumSize() && layers <= getMaximumLayers() );
        m_width = width;
        m_layers = layers;

        getGLInterface()->setImage2D( getGLId(), getTarget(), 0, getInternalFormat(), width, layers, 0, getFormat(), getType(), 0 ); 

        resetDefinedLevels();
        setMaxLevel( numberOfMipmaps( getWidth(), 1 , 1 ) );
      }
    }

    GLsizei Texture1DArray::getMaximumSize()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_TEXTURE_SIZE, &size );
      return size;
    }

    GLsizei Texture1DArray::getMaximumLayers()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_ARRAY_TEXTURE_LAYERS_EXT, &size );
      return size;
    }

    bool Texture1DArray::isSupported()
    {
      return !!GLEW_EXT_texture_array;
    }


    /***************/
    /* Texture2D   */
    /***************/

    Texture2DSharedPtr Texture2D::create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height )
    {
      return( std::shared_ptr<Texture2D>( new Texture2D( internalFormat, format, type, width, height ) ) );
    }

    Texture2D::Texture2D( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height )
      : Texture( GL_TEXTURE_2D, internalFormat, format, type )
      , m_width( 0 )
      , m_height( 0 )
    {
      resize( width, height );
    }

    void Texture2D::setData( const void *data, GLuint mipLevel /*= 0 */ )
    {
      DP_ASSERT( isMipMapLevelValid( mipLevel) );

      TexGLTransfer upload( getGLId(), getTarget(), mipLevel, getInternalFormat(), getFormat(), getType(), data, 
                            getMipMapSize( m_width, mipLevel ), getMipMapSize( m_height, mipLevel ) );
      // set m_dataSize for compressed format
      upload.m_dataSize = getCompressedSize( getFormat(), upload.m_width, upload.m_height, 1 );
      // format must match if compressed
      DP_ASSERT( !upload.m_dataSize || getFormat() == getInternalFormat() );

      upload.doTransfer( 2, isMipMapLevelDefined( mipLevel ) );

      addDefinedLevel( mipLevel );
    }

    void Texture2D::getData( void *data, GLuint mipLevel /*= 0 */ ) const
    {
      DP_ASSERT( isMipMapLevelDefined( mipLevel ) );
      DP_ASSERT( data );
      getTexImage( this, getTarget(), data, mipLevel );
    }

    void Texture2D::resize( GLsizei width, GLsizei height )
    {
      if ( m_width != width || m_height != height)
      {
        DP_ASSERT( width <= getMaximumSize() && height <= getMaximumSize() );
        m_width = width;
        m_height = height;

        getGLInterface()->setImage2D( getGLId(), getTarget(), 0, getInternalFormat(), width, height, 0, getFormat(), getType(), nullptr ); 

        resetDefinedLevels();
        setMaxLevel( numberOfMipmaps( getWidth(), getWidth() , 1 ) );
      }
    }

    GLsizei Texture2D::getMaximumSize()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_TEXTURE_SIZE, &size );
      return size;
    }


    /************************/
    /* TextureRectangle     */
    /************************/

    TextureRectangleSharedPtr TextureRectangle::create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height )
    {
      return( std::shared_ptr<TextureRectangle>( new TextureRectangle( internalFormat, format, type, width, height ) ) );
    }

    TextureRectangle::TextureRectangle( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height )
      : Texture( GL_TEXTURE_RECTANGLE, internalFormat, format, type )
      , m_width( 0 )
      , m_height( 0 )
    {
      resize( width, height );
    }

    void TextureRectangle::setData( const void *data )
    {
      TexGLTransfer upload( getGLId(), getTarget(), 0, getInternalFormat(), getFormat(), getType(), data, 
                            m_width, m_height );
      // set m_dataSize for compressed format
      upload.m_dataSize = getCompressedSize( getFormat(), upload.m_width, upload.m_height, 1 );
      // format must match if compressed
      DP_ASSERT( !upload.m_dataSize || getFormat() == getInternalFormat() );

      upload.doTransfer( 2, isMipMapLevelDefined( 0 ) );

      resetDefinedLevels();
    }

    void TextureRectangle::getData( void *data ) const
    {
      DP_ASSERT( data );
      getTexImage( this, getTarget(), data, 0 );
    }

    void TextureRectangle::resize( GLsizei width, GLsizei height )
    {
      if ( m_width != width || m_height != height)
      {
        DP_ASSERT( width <= getMaximumSize() && height <= getMaximumSize() );
        m_width = width;
        m_height = height;

        getGLInterface()->setImage2D( getGLId(), getTarget(), 0, getInternalFormat(), width, height, 0, getFormat(), getType(), 0 ); 

        resetDefinedLevels();
        setMaxLevel( 0 ); // rectangle textures must not have mipmaps
      }
    }

    GLsizei TextureRectangle::getMaximumSize()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_RECTANGLE_TEXTURE_SIZE_ARB, &size );
      return size;
    }


    /*********************/
    /* Texture2DArray    */
    /*********************/

    Texture2DArraySharedPtr Texture2DArray::create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height, GLsizei layers )
    {
      return( std::shared_ptr<Texture2DArray>( new Texture2DArray( internalFormat, format, type, width, height, layers ) ) );
    }

    Texture2DArray::Texture2DArray( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height, GLsizei layers )
      : Texture( GL_TEXTURE_2D_ARRAY, internalFormat, format, type )
      , m_width( 0 )
      , m_height( 0 )
      , m_layers( 0 )
    {
      resize( width, height, layers );
    }

    void Texture2DArray::setData( const void *data, GLint layer, GLuint mipLevel /*= 0 */ )
    {
      DP_ASSERT( isMipMapLevelValid( mipLevel) );
      DP_ASSERT( layer < getLayers() );

      TexGLTransfer upload( getGLId(), getTarget(), mipLevel, getInternalFormat(), getFormat(), getType(), data, 
                            getMipMapSize( m_width, mipLevel ), getMipMapSize( m_height, mipLevel ) );
      // set m_dataSize for compressed format
      upload.m_dataSize = getCompressedSize( getFormat(), upload.m_width, upload.m_height, 1 );
      // format must match if compressed
      DP_ASSERT( !upload.m_dataSize || getFormat() == getInternalFormat() );

      if ( !isMipMapLevelDefined(mipLevel) )
      {
        TexGLTransfer uploadCreate = upload;
        dp::gl::bind( GL_PIXEL_UNPACK_BUFFER, BufferSharedPtr::null );    // make sure, GL_PIXEL_UNPACK_BUFFER is unbound !
        uploadCreate.m_dataSize *= m_layers;
        uploadCreate.m_depth = m_layers;
        uploadCreate.m_dataPtr = nullptr;
        uploadCreate.doTransfer( 3, false );
        addDefinedLevel( mipLevel );
      }
   
      upload.m_zOffset = layer;
      upload.doTransfer( 3, true );

      addDefinedLevel( mipLevel );
    }

    void Texture2DArray::getData( void *data, GLuint mipLevel /*= 0 */ ) const
    {
      DP_ASSERT( isMipMapLevelDefined( mipLevel ) );
      DP_ASSERT( data );
      getTexImage( this, getTarget(), data, mipLevel );
    }

    void Texture2DArray::resize( GLsizei width, GLsizei height, GLsizei layers )
    {
      if ( m_width != width || m_height != height || m_layers != layers)
      {
        DP_ASSERT( width <= getMaximumSize() && height <= getMaximumSize() && layers <= getMaximumLayers() );
        m_width = width;
        m_height = height;
        m_layers = layers;

        getGLInterface()->setImage3D( getGLId(), getTarget(), 0, getInternalFormat(), width, height, layers, 0, getFormat(), getType(), 0 ); 

        resetDefinedLevels();
        setMaxLevel( numberOfMipmaps( getWidth(), getWidth() , 1 ) );
      }
    }

    GLsizei Texture2DArray::getMaximumSize()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_TEXTURE_SIZE, &size );
      return size;
    }

    GLsizei Texture2DArray::getMaximumLayers()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_ARRAY_TEXTURE_LAYERS_EXT, &size );
      return size;
    }
  
    bool Texture2DArray::isSupported()
    {
      return !!GLEW_EXT_texture_array;
    }


    /***************/
    /* Texture3D   */
    /***************/

    Texture3DSharedPtr Texture3D::create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height, GLsizei depth )
    {
      return( std::shared_ptr<Texture3D>( new Texture3D( internalFormat, format, type, width, height, depth ) ) );
    }

    Texture3D::Texture3D( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height, GLsizei depth )
      : Texture( GL_TEXTURE_3D, internalFormat, format, type )
      , m_width( 0 )
      , m_height( 0 )
      , m_depth( 0 )
    {
      resize( width, height, depth );
    }

    void Texture3D::setData( const void *data, GLuint mipLevel /*= 0 */ )
    {
      DP_ASSERT( isMipMapLevelValid( mipLevel) );

      TexGLTransfer upload( getGLId(), getTarget(), mipLevel, getInternalFormat(), getFormat(), getType(), data, 
                            getMipMapSize( m_width, mipLevel ), getMipMapSize( m_height, mipLevel ), getMipMapSize( m_depth, mipLevel ) );
      // set m_dataSize for compressed format
      upload.m_dataSize = getCompressedSize( getFormat(), upload.m_width, upload.m_height, upload.m_depth );
      // format must match if compressed
      DP_ASSERT( !upload.m_dataSize || getFormat() == getInternalFormat() );

      upload.doTransfer( 3, isMipMapLevelDefined(mipLevel) );

      addDefinedLevel( mipLevel );
    }

    void Texture3D::getData( void *data, GLuint mipLevel /*= 0 */ ) const
    {
      DP_ASSERT( isMipMapLevelDefined( mipLevel ) );
      DP_ASSERT( data );
      getTexImage( this, getTarget(), data, mipLevel );
    }

    void Texture3D::resize( GLsizei width, GLsizei height, GLsizei depth )
    {
      if ( m_width != width || m_height != height || m_depth != depth)
      {
        DP_ASSERT( width <= getMaximumSize() && height <= getMaximumSize() && depth <= getMaximumSize() );
        m_width = width;
        m_height = height;
        m_depth = depth;

        getGLInterface()->setImage3D( getGLId(), getTarget(), 0, getInternalFormat(), width, height, depth, 0, getFormat(), getType(), 0 ); 

        resetDefinedLevels();
        setMaxLevel( numberOfMipmaps( getWidth(), getWidth(), getDepth() ) );
      }
    }

    GLsizei Texture3D::getMaximumSize()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_3D_TEXTURE_SIZE, &size );
      return size;
    }


    /********************/
    /* TextureCubemap   */
    /********************/

    TextureCubemapSharedPtr TextureCubemap::create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height )
    {
      return( std::shared_ptr<TextureCubemap>( new TextureCubemap( internalFormat, format, type, width, height ) ) );
    }

    TextureCubemap::TextureCubemap( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height )
      : Texture( GL_TEXTURE_CUBE_MAP, internalFormat, format, type )
      , m_width( 0 )
      , m_height( 0 )
    {
      resize( width, height );
    }

    void TextureCubemap::setData( const void *data, int face, GLuint mipLevel /*= 0 */ )
    {
      DP_ASSERT( 0 <= face && face <= 6);
      DP_ASSERT( isMipMapLevelValid( mipLevel ) );

      TexGLTransfer upload( getGLId(), GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, mipLevel, getInternalFormat(), getFormat(), getType(), data,
                            getMipMapSize( m_width, mipLevel ), getMipMapSize( m_height, mipLevel ) );
      // set m_dataSize for compressed format
      upload.m_dataSize = getCompressedSize( getFormat(), upload.m_width, upload.m_height, 1 );
      // format must match if compressed
      DP_ASSERT( !upload.m_dataSize || getFormat() == getInternalFormat() );

      if ( !isMipMapLevelDefined(mipLevel) )
      {
        TexGLTransfer uploadCreate = upload;
        dp::gl::bind( GL_PIXEL_UNPACK_BUFFER, BufferSharedPtr::null );    // make sure, GL_PIXEL_UNPACK_BUFFER is unbound !
        uploadCreate.m_dataPtr = nullptr;
        for (unsigned int f = 0; f < 6; ++f)
        {
          uploadCreate.m_target = GL_TEXTURE_CUBE_MAP_POSITIVE_X + f;
          uploadCreate.doTransfer( 2, false );
        }
        addDefinedLevel( mipLevel );
      }

      upload.doTransfer( 2, true );
    }

    void TextureCubemap::getData( void *data, int face, GLuint mipLevel /*= 0 */ ) const
    {
      DP_ASSERT( 0 <= face && face <= 6);
      DP_ASSERT( isMipMapLevelDefined( mipLevel ) );
      DP_ASSERT( data );
      getTexImage( this, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, data, mipLevel );
    }

    void TextureCubemap::resize( GLsizei width, GLsizei height )
    {
      if ( m_width != width || m_height != height)
      {
        DP_ASSERT( width <= getMaximumSize() );
        DP_ASSERT( width == height );
        m_width = width;
        m_height = height;

        for ( unsigned int face = 0;face < 6;++face )
        {
          getGLInterface()->setImage2D( getGLId(), GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, getInternalFormat(), width, height, 0, getFormat(), getType(), nullptr ); 
        }

        resetDefinedLevels();
        setMaxLevel( numberOfMipmaps( getWidth(), getWidth(), 1 ) );
      }
    }

    GLsizei TextureCubemap::getMaximumSize()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_CUBE_MAP_TEXTURE_SIZE, &size );
      return size;
    }


    /**************************/
    /* TextureCubemapArray    */
    /**************************/

    TextureCubemapArraySharedPtr TextureCubemapArray::create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height, GLsizei layers )
    {
      return( std::shared_ptr<TextureCubemapArray>( new TextureCubemapArray( internalFormat, format, type, width, height, layers ) ) );
    }

    TextureCubemapArray::TextureCubemapArray( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height, GLsizei layers )
      : Texture( GL_TEXTURE_CUBE_MAP_ARRAY, internalFormat, format, type )
      , m_width( 0 )
      , m_height( 0 )
      , m_layers( 0 )
    {
      resize( width, height, layers );
    }

    void TextureCubemapArray::setData( const void *data, GLint layer, GLuint mipLevel /*= 0 */ )
    {
      DP_ASSERT( isMipMapLevelValid( mipLevel) );
      DP_ASSERT( layer < getLayers() );

      TexGLTransfer upload( getGLId(), getTarget(), mipLevel, getInternalFormat(), getFormat(), getType(), data, 
                            getMipMapSize( m_width, mipLevel ), getMipMapSize( m_height, mipLevel ) );
      // set m_dataSize for compressed format
      upload.m_dataSize = getCompressedSize( getFormat(), upload.m_width, upload.m_height, 1 );
      // format must match if compressed
      DP_ASSERT( !upload.m_dataSize || getFormat() == getInternalFormat() );

      if ( !isMipMapLevelDefined(mipLevel) )
      {
        TexGLTransfer uploadCreate = upload;
        dp::gl::bind( GL_PIXEL_UNPACK_BUFFER, BufferSharedPtr::null );    // make sure, GL_PIXEL_UNPACK_BUFFER is unbound !
        uploadCreate.m_dataSize *= m_layers;
        uploadCreate.m_depth = m_layers;
        uploadCreate.m_dataPtr = nullptr;
        uploadCreate.doTransfer( 3, false );
        addDefinedLevel( mipLevel );
      }

      upload.m_zOffset = layer;
      upload.doTransfer( 3, true );
    }

    void TextureCubemapArray::getData( void *data, GLuint mipLevel /*= 0 */ ) const
    {
      DP_ASSERT( isMipMapLevelDefined( mipLevel ) );
      DP_ASSERT( data );
      getTexImage( this, getTarget(), data, mipLevel );
    }

    void TextureCubemapArray::resize( GLsizei width, GLsizei height, GLsizei layers )
    {
      if ( m_width != width || m_height != height || m_layers != layers)
      {
        DP_ASSERT( width <= getMaximumSize() && layers <= getMaximumLayers() && layers%6 == 0);
        DP_ASSERT( width == height );
        m_width = width;
        m_height = height;
        m_layers = layers;

        getGLInterface()->setImage3D( getGLId(), getTarget(), 0, getInternalFormat(), width, height, layers , 0, getFormat(), getType(), 0 ); 

        resetDefinedLevels();
        setMaxLevel( numberOfMipmaps( getWidth(), getWidth() , 1 ) );
      }
    }

    GLsizei TextureCubemapArray::getMaximumSize()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_CUBE_MAP_TEXTURE_SIZE, &size );
      return size;
    }

    GLsizei TextureCubemapArray::getMaximumLayers()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_ARRAY_TEXTURE_LAYERS_EXT, &size );
      return size;
    }

    bool TextureCubemapArray::isSupported()
    {
      return !!GLEW_ARB_texture_cube_map_array;
    }


    /***************************/
    /* Texture2DMultisample    */
    /***************************/

    Texture2DMultisampleSharedPtr Texture2DMultisample::create( GLenum internalFormat, GLsizei samples, GLsizei width, GLsizei height, bool fixedLocations )
    {
      return( std::shared_ptr<Texture2DMultisample>( new Texture2DMultisample( internalFormat, samples, width, height, fixedLocations ) ) );
    }

    Texture2DMultisample::Texture2DMultisample( GLenum internalFormat, GLsizei width, GLsizei height, GLsizei samples, bool fixedLocations )
      : Texture( GL_TEXTURE_2D_MULTISAMPLE, internalFormat, GL_INVALID_ENUM, GL_INVALID_ENUM )
      , m_samples( 1 )
      , m_width( 0 )
      , m_height( 0 )
      , m_fixedLocations( fixedLocations )
    {
      resize( width, height );
      setSamples( samples );
    }

    void Texture2DMultisample::resize( GLsizei width, GLsizei height )
    {
      DP_ASSERT( getInternalFormat() != ~0 );
      DP_ASSERT( m_samples );

      if ( width != m_width || height != m_height )
      {
        DP_ASSERT( width <= getMaximumSize() && height <= getMaximumSize() );
        m_width = width;
        m_height = height;

        TextureBinding( getTarget(), getGLId() );
        glTexImage2DMultisample( getTarget(), m_samples, getInternalFormat(), m_width, m_height, m_fixedLocations );
      }
    }

    void Texture2DMultisample::setSamples( GLsizei samples )
    {
      DP_ASSERT( getInternalFormat() != ~0 );
      DP_ASSERT( m_width && m_height );

      if ( samples != m_samples )
      {
        m_samples = samples;

        TextureBinding( getTarget(), getGLId() );
        glTexImage2DMultisample( getTarget(), m_samples, getInternalFormat(), m_width, m_height, m_fixedLocations );
      }
    }

    GLsizei Texture2DMultisample::getMaximumSize()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_TEXTURE_SIZE, &size );
      return size;
    }

    GLsizei Texture2DMultisample::getMaximumSamples()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_SAMPLES, &size );
      return size;
    }

    GLsizei Texture2DMultisample::getMaximumIntegerSamples()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_INTEGER_SAMPLES, &size );
      return size;
    }

    GLsizei Texture2DMultisample::getMaximumColorSamples()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_COLOR_TEXTURE_SAMPLES, &size );
      return size;
    }

    GLsizei Texture2DMultisample::getMaximumDepthSamples()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_DEPTH_TEXTURE_SAMPLES, &size );
      return size;
    }

    bool Texture2DMultisample::isSupported()
    {
      return !!GLEW_ARB_texture_multisample;
    }


    /********************************/
    /* Texture2DMultisampleArray    */
    /********************************/

    Texture2DMultisampleArraySharedPtr Texture2DMultisampleArray::create( GLenum internalFormat, GLsizei samples, GLsizei width, GLsizei height, GLsizei layers, bool fixedLocations )
    {
      return( std::shared_ptr<Texture2DMultisampleArray>( new Texture2DMultisampleArray( internalFormat, samples, width, height, layers, fixedLocations ) ) );
    }

    Texture2DMultisampleArray::Texture2DMultisampleArray( GLenum internalFormat, GLsizei samples, GLsizei width, GLsizei height, GLsizei layers, bool fixedLocations )
      : Texture( GL_TEXTURE_2D_MULTISAMPLE_ARRAY, internalFormat, GL_INVALID_ENUM, GL_INVALID_ENUM )
      , m_samples( 1 )
      , m_width( 0 )
      , m_height( 0 )
      , m_fixedLocations( fixedLocations )
    {
      resize( width, height, layers );
      setSamples( samples );
    }

    void Texture2DMultisampleArray::resize( GLsizei width, GLsizei height, GLsizei layers )
    {
      DP_ASSERT( getInternalFormat() != ~0 );
      DP_ASSERT( m_samples );

      if ( width != m_width || height != m_height || layers != m_layers)
      {
        DP_ASSERT( width <= getMaximumSize() && height <= getMaximumSize() );
        DP_ASSERT( layers <= getMaximumLayers() );

        m_width = width;
        m_height = height;
        m_layers = layers;

        TextureBinding( getTarget(), getGLId() );
        glTexImage3DMultisample( getTarget(), m_samples, getInternalFormat(), m_width, m_height, m_layers, m_fixedLocations );
      }
    }

    void Texture2DMultisampleArray::setSamples( GLsizei samples )
    {
      DP_ASSERT( getInternalFormat() != ~0 );
      DP_ASSERT( m_width && m_height && m_layers );

      if ( samples != m_samples )
      {
        DP_ASSERT( samples <= getMaximumSamples() );
        m_samples = samples;

        TextureBinding( getTarget(), getGLId() );
        glTexImage3DMultisample( getTarget(), m_samples, getInternalFormat(), m_width, m_height, m_layers, m_fixedLocations );
      }
    }

    GLsizei Texture2DMultisampleArray::getMaximumSize()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_TEXTURE_SIZE, &size );
      return size;
    }

    GLsizei Texture2DMultisampleArray::getMaximumSamples()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_SAMPLES, &size );
      return size;
    }

    GLsizei Texture2DMultisampleArray::getMaximumColorSamples()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_COLOR_TEXTURE_SAMPLES, &size );
      return size;
    }

    GLsizei Texture2DMultisampleArray::getMaximumDepthSamples()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_DEPTH_TEXTURE_SAMPLES, &size );
      return size;
    }

    GLsizei Texture2DMultisampleArray::getMaximumLayers()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_ARRAY_TEXTURE_LAYERS_EXT, &size );
      return size;
    }

    GLsizei Texture2DMultisampleArray::getMaximumIntegerSamples()
    {
      GLsizei size;
      glGetIntegerv( GL_MAX_INTEGER_SAMPLES, &size );
      return size;
    }

    bool Texture2DMultisampleArray::isSupported()
    {
      return !!GLEW_ARB_texture_multisample;
    }


    /********************/
    /* TextureBuffer    */
    /********************/

    TextureBufferSharedPtr TextureBuffer::create( GLenum internalFormat, BufferSharedPtr const& buffer )
    {
      return( std::shared_ptr<TextureBuffer>( new TextureBuffer( internalFormat, buffer ) ) );
    }

    TextureBufferSharedPtr TextureBuffer::create( GLenum internalFormat, unsigned int size, GLvoid const* data, GLenum usage )
    {
      return( std::shared_ptr<TextureBuffer>( new TextureBuffer( internalFormat, Buffer::create( GL_TEXTURE_BUFFER, size, data, usage ) ) ) );
    }

    TextureBuffer::TextureBuffer( GLenum internalFormat, BufferSharedPtr const& buffer )
      : Texture( GL_TEXTURE_BUFFER, internalFormat, GL_INVALID_ENUM, GL_INVALID_ENUM )
    {
      setBuffer( buffer );
    }

    TextureBuffer::~TextureBuffer()
    {
    }

    BufferSharedPtr const& TextureBuffer::getBuffer() const
    {
      return( m_buffer );
    }

    void TextureBuffer::setBuffer( BufferSharedPtr const& buffer )
    {
      if ( m_buffer != buffer )
      {
        m_buffer = buffer;
        getGLInterface()->attachBuffer( getGLId(), GL_TEXTURE_BUFFER, getInternalFormat(), m_buffer ? m_buffer->getGLId() : 0 );
      }
    }

    GLint TextureBuffer::getMaximumSize()
    {
      GLint size;
      glGetIntegerv( GL_MAX_TEXTURE_BUFFER_SIZE, &size );
      return( size );
    }

    bool TextureBuffer::isSupported()
    {
      return( !!GLEW_ARB_texture_buffer_object || !!GLEW_EXT_texture_buffer_object );
    }

  } // namespace gl
} // namespace dp
