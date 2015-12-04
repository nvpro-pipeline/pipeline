// Copyright NVIDIA Corporation 2010-2013
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


#include <dp/sg/ui/qt5/SceniXQtUtil.h>

#include <dp/sg/core/TextureHost.h>
#include <dp/util/Image.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace qt5
      {

        using namespace dp::sg::core;

        static QImage errorImage;

        // Data in QImage::Format_ARGB32 
        const static unsigned int rgby[] =
        {
          0xFFFF0000,
          0xFF00FF00,
          0xFF0000FF,
          0xFFFFFF00
        };

        template <typename SrcType>
        struct UnsignedIntegerToUnsignedChar
        {
        public:
          unsigned char operator()( const char *&src ) const
          {
            unsigned char result = *((SrcType*)src) >> ((sizeof(SrcType) * 8) - 8);
            src += sizeof(SrcType);
            return result;
          }
        };

        template <typename SrcType>
        struct IntegerToUnsignedChar
        {
        public:
          unsigned char operator()( const char *&src ) const
          {
            unsigned char result = std::max((SrcType)0, *((SrcType*)src)) >> ((sizeof(SrcType) * 8) - 8);
            src += sizeof(SrcType);
            return result;
          }
        };


        template <typename SrcType>
        struct FloatToChar
        {
        public:
          FloatToChar( SrcType maxValue ) : m_scale( 255.0f / maxValue ) {}

          unsigned char operator()( const char *&src ) const
          {
            unsigned char result = static_cast<unsigned char>(*((SrcType*)src) * m_scale);
            src += sizeof(SrcType);
            return result;
          }
        protected:
          SrcType m_scale;
        };


        template <typename Converter>
        void convertRGBToRGBA( unsigned char *dst, const char *src, unsigned int numPixels, const Converter &converter)
        {
          for (unsigned int pixel = 0;pixel < numPixels;++pixel)
          {
            dst[2] = converter(src);
            dst[1] = converter(src);
            dst[0] = converter(src);
            dst[3] = 255;
            dst += 4;
          }
        }

        template <typename Converter>
        void convertRGBAToRGBA( unsigned char *dst, const char *src, unsigned int numPixels, const Converter &converter)
        {
          for (unsigned int pixel = 0;pixel < numPixels;++pixel)
          {
            dst[2] = converter(src);
            dst[1] = converter(src);
            dst[0] = converter(src);
            dst[3] = converter(src);
            dst += 4;
          }
        }

        template <typename Converter>
        void convertBGRToRGBA( unsigned char *dst, const char *src, unsigned int &numPixels, const Converter &converter)
        {
          for (unsigned int pixel = 0;pixel < numPixels;++pixel)
          {
            dst[0] = converter(src);
            dst[1] = converter(src);
            dst[2] = converter(src);
            dst[3] = 255;
            dst += 4;
          }
        }

        template <typename Converter>
        void convertBGRAToRGBA( unsigned char *dst, const char *src, unsigned int numPixels, const Converter &converter)
        {
          for (unsigned int pixel = 0;pixel < numPixels;++pixel)
          {
            dst[0] = converter(src);
            dst[1] = converter(src);
            dst[2] = converter(src);
            dst[3] = converter(src);
            dst += 4;
          }
        }

        /** \brief Creates an QImage out of a TextureHost 
        **/
        QImage createQImage( const TextureHostSharedPtr & textureImage, int image, int mipmap )
        {
          Buffer::DataReadLock buffer( textureImage->getPixels( image, mipmap ) );
          const void *srcData = buffer.getPtr();

          int width = textureImage->getWidth( image, mipmap );
          int height = textureImage->getHeight( image, mipmap );
          unsigned int numPixels = width * height;

          std::vector<unsigned char> tmpData;
          tmpData.resize( width * 4 * height );
          QImage result;

          // determine QImage format
          QImage::Format format = QImage::Format_ARGB32;
          bool supported = true;

          if (textureImage->getDepth() != 1)
          {
            supported = false;
          }
          else
          {
            // convert to RGBA
            switch ( textureImage->getFormat( image, mipmap ) )
            {
            case Image::IMG_BGR:
              switch (textureImage->getType())
              {
              case Image::IMG_UNSIGNED_BYTE:
                convertBGRToRGBA( &tmpData[0], (char*)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned char>() );
                break;
              case Image::IMG_UNSIGNED_SHORT:
                convertBGRToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned short>() );
                break;
              case Image::IMG_UNSIGNED_INT:
                convertBGRToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned int>() );
                break;
              case Image::IMG_BYTE:
                convertBGRToRGBA( &tmpData[0], (char*)srcData, numPixels, IntegerToUnsignedChar<char>() );
                break;
              case Image::IMG_SHORT:
                convertBGRToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<short>() );
                break;
              case Image::IMG_INT:
                convertBGRToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<int>() );
                break;
              case Image::IMG_FLOAT:
                {
                  const float *src = static_cast<const float *>(srcData);
                  float max = *std::max_element( src, src + 3 * numPixels );
                  convertBGRToRGBA( &tmpData[0], (char *)srcData, numPixels, FloatToChar<float>(max) );
                }
                break;
              default:
                supported = false;
              }
              break;
            case Image::IMG_RGB:
              switch (textureImage->getType())
              {
              case Image::IMG_UNSIGNED_BYTE:
                convertRGBToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned char>() );
                break;
              case Image::IMG_UNSIGNED_SHORT:
                convertRGBToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned short>() );
                break;
              case Image::IMG_UNSIGNED_INT:
                convertRGBToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned int>() );
                break;
              case Image::IMG_BYTE:
                convertRGBToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<char>() );
                break;
              case Image::IMG_SHORT:
                convertRGBToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<short>() );
                break;
              case Image::IMG_INT:
                convertRGBToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<int>() );
                break;
              case Image::IMG_FLOAT:
                {
                  const float *src = static_cast<const float *>(srcData);
                  float max = *std::max_element( src, src + 3 * numPixels );
                  convertRGBToRGBA( &tmpData[0], (char *)srcData, numPixels, FloatToChar<float>( max ) );
                }
                break;
              default:
                supported = false;
              }
              break;
            case Image::IMG_RGBA:
              switch (textureImage->getType())
              {
              case Image::IMG_UNSIGNED_BYTE:
                convertRGBAToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned char>() );
                break;
              case Image::IMG_UNSIGNED_SHORT:
                convertRGBAToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned short>() );
                break;
              case Image::IMG_UNSIGNED_INT:
                convertRGBAToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned int>() );
                break;
              case Image::IMG_BYTE:
                convertRGBAToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<char>() );
                break;
              case Image::IMG_SHORT:
                convertRGBAToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<short>() );
                break;
              case Image::IMG_INT:
                convertRGBAToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<int>() );
                break;
              case Image::IMG_FLOAT:
                {
                  const float *src = static_cast<const float *>(srcData);
                  float max = *std::max_element( src, src + 4 * numPixels );
                  convertRGBAToRGBA( &tmpData[0], (char *)srcData, numPixels, FloatToChar<float>( max ) );
                }
                break;
              default:
                supported = false;
              }
              break;
            case Image::IMG_BGRA:
              switch (textureImage->getType())
              {
              case Image::IMG_UNSIGNED_BYTE:
                convertBGRAToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned char>() );
                break;
              case Image::IMG_UNSIGNED_SHORT:
                convertBGRAToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned short>() );
                break;
              case Image::IMG_UNSIGNED_INT:
                convertBGRAToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned int>() );
                break;
              case Image::IMG_BYTE:
                convertBGRAToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<char>() );
                break;
              case Image::IMG_SHORT:
                convertBGRAToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<short>() );
                break;
              case Image::IMG_INT:
                convertBGRAToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<int>() );
                break;
              case Image::IMG_FLOAT:
                {
                  const float *src = static_cast<const float *>(srcData);
                  float max = *std::max_element( src, src + 4 * numPixels );
                  convertBGRAToRGBA( &tmpData[0], (char *)srcData, numPixels, FloatToChar<float>( max ) );
                }
                break;
              default:
                supported = false;
              }
              break;
            default:
              supported = false;
              break;
            };
          }
          if( !supported )
          {
            if( errorImage.isNull() )
            {
              errorImage = QImage( reinterpret_cast<const uchar *>(rgby), 2, 2, QImage::Format_ARGB32 );
            }

            result = errorImage;
          }
          else
          {
            // The QImage is mirrored vertically when using the pixelData of a TextureHost. Mirror it. 
            // This way it's also ensured that Qt does manage a copy of the pixel data.
            // DAR BUG: Won't work with Format_RGB888 and width which is NOT a multiple of four!
            // Because: "Constructs an image with the given width, height and format, that uses an existing memory buffer, data. 
            //           The width and height must be specified in pixels, data must be 32-bit aligned, 
            //           and ***each scanline of data in the image must also be 32-bit aligned***."
            result = QImage( reinterpret_cast<const uchar *>( &tmpData[0] ), 
                                    textureImage->getWidth( image, mipmap ), textureImage->getHeight( image, mipmap ), format ).mirrored();
          }

          return result;
        }


        /** \brief Creates an QImage out of a TextureHost 
        **/
        QImage createQImage( const dp::util::ImageSharedPtr& image /*, int image, int mipmap */ )
        {
          const void *srcData = image->getLayerData( 0, 0 );
          int width = static_cast<int>(image->getWidth( ));
          int height = static_cast<int>(image->getHeight( ));
          unsigned int numPixels = width * height;

          std::vector<unsigned char> tmpData;
          tmpData.resize( width * 4 * height );
          QImage result;

          // determine QImage format
          QImage::Format format = QImage::Format_ARGB32;
          bool supported = true;

          // convert to RGBA
          switch ( image->getPixelFormat( ) )
          {
          case dp::PixelFormat::BGR:
            switch (image->getDataType())
            {
            case dp::DataType::UNSIGNED_INT_8:
              convertBGRToRGBA( &tmpData[0], (char*)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned char>() );
              break;
            case dp::DataType::UNSIGNED_INT_16:
              convertBGRToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned short>() );
              break;
            case dp::DataType::UNSIGNED_INT_32:
              convertBGRToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned int>() );
              break;
            case dp::DataType::INT_8:
              convertBGRToRGBA( &tmpData[0], (char*)srcData, numPixels, IntegerToUnsignedChar<char>() );
              break;
            case dp::DataType::INT_16:
              convertBGRToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<short>() );
              break;
            case dp::DataType::INT_32:
              convertBGRToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<int>() );
              break;
            case dp::DataType::FLOAT_32:
              {
                const float *src = static_cast<const float *>(srcData);
                float max = *std::max_element( src, src + 3 * numPixels );
                convertBGRToRGBA( &tmpData[0], (char *)srcData, numPixels, FloatToChar<float>(max) );
              }
              break;
            default:
              supported = false;
            }
            break;
          case dp::PixelFormat::RGB:
            switch (image->getDataType())
            {
            case dp::DataType::UNSIGNED_INT_8:
              convertRGBToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned char>() );
              break;
            case dp::DataType::UNSIGNED_INT_16:
              convertRGBToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned short>() );
              break;
            case dp::DataType::UNSIGNED_INT_32:
              convertRGBToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned int>() );
              break;
            case dp::DataType::INT_8:
              convertRGBToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<char>() );
              break;
            case dp::DataType::INT_16:
              convertRGBToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<short>() );
              break;
            case dp::DataType::INT_32:
              convertRGBToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<int>() );
              break;
            case dp::DataType::FLOAT_32:
              {
                const float *src = static_cast<const float *>(srcData);
                float max = *std::max_element( src, src + 3 * numPixels );
                convertRGBToRGBA( &tmpData[0], (char *)srcData, numPixels, FloatToChar<float>( max ) );
              }
              break;
            default:
              supported = false;
            }
            break;
          case dp::PixelFormat::RGBA:
            switch (image->getDataType())
            {
            case dp::DataType::UNSIGNED_INT_8:
              convertRGBAToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned char>() );
              break;
            case dp::DataType::UNSIGNED_INT_16:
              convertRGBAToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned short>() );
              break;
            case dp::DataType::UNSIGNED_INT_32:
              convertRGBAToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned int>() );
              break;
            case dp::DataType::INT_8:
              convertRGBAToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<char>() );
              break;
            case dp::DataType::INT_16:
              convertRGBAToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<short>() );
              break;
            case dp::DataType::INT_32:
              convertRGBAToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<int>() );
              break;
            case dp::DataType::FLOAT_32:
              {
                const float *src = static_cast<const float *>(srcData);
                float max = *std::max_element( src, src + 4 * numPixels );
                convertRGBAToRGBA( &tmpData[0], (char *)srcData, numPixels, FloatToChar<float>( max ) );
              }
              break;
            default:
              supported = false;
            }
            break;
          case dp::PixelFormat::BGRA:
            switch (image->getDataType())
            {
            case dp::DataType::UNSIGNED_INT_8:
              convertBGRAToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned char>() );
              break;
            case dp::DataType::UNSIGNED_INT_16:
              convertBGRAToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned short>() );
              break;
            case dp::DataType::UNSIGNED_INT_32:
              convertBGRAToRGBA( &tmpData[0], (char *)srcData, numPixels, UnsignedIntegerToUnsignedChar<unsigned int>() );
              break;
            case dp::DataType::INT_8:
              convertBGRAToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<char>() );
              break;
            case dp::DataType::INT_16:
              convertBGRAToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<short>() );
              break;
            case dp::DataType::INT_32:
              convertBGRAToRGBA( &tmpData[0], (char *)srcData, numPixels, IntegerToUnsignedChar<int>() );
              break;
            case dp::DataType::FLOAT_32:
              {
                const float *src = static_cast<const float *>(srcData);
                float max = *std::max_element( src, src + 4 * numPixels );
                convertBGRAToRGBA( &tmpData[0], (char *)srcData, numPixels, FloatToChar<float>( max ) );
              }
              break;
            default:
              supported = false;
            }
            break;
          default:
            supported = false;
            break;
          };

          // The QImage is mirrored vertically when using the pixelData of a TextureHost. Mirror it. 
          // This way it's also ensured that Qt does manage a copy of the pixel data.
          // DAR BUG: Won't work with Format_RGB888 and width which is NOT a multiple of four!
          // Because: "Constructs an image with the given width, height and format, that uses an existing memory buffer, data. 
          //           The width and height must be specified in pixels, data must be 32-bit aligned, 
          //           and ***each scanline of data in the image must also be 32-bit aligned***."
          result = QImage( reinterpret_cast<const uchar *>( &tmpData[0] ), 
            width, height, format ).mirrored();

          return result;
        }

      } // namespace qt5
    } // namespace ui
  } // namespace sg
} // namespace dp
