// Copyright (c) 2011-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/Assert.h>
#include <dp/util/Image.h>
#include <dp/util/File.h>
#include <dp/util/Singleton.h>

#if defined(HAVE_IL)
  #include <il.h>
#endif

#include <iostream>
#include <cstring>

using namespace std;
using namespace dp::util;

namespace dp
{
  namespace util
  {

    ImageSharedPtr Image::create()
    {
      return( std::shared_ptr<Image>( new Image() ) );
    }

    ImageSharedPtr Image::create( size_t width, size_t height, PixelFormat pixelFormat, DataType dataType, void const* const* data, size_t numLayers, size_t mipmapLevels )
    {
      return( std::shared_ptr<Image>( new Image( width, height, pixelFormat, dataType, data, numLayers, mipmapLevels ) ) );
    }

    Image::Image()
      : m_width(0)
      , m_height(0)
      , m_pixelFormat(PixelFormat::UNKNOWN)
      , m_dataType(DataType::UNKNOWN)
      , m_bpp(0)
      , m_totalSize(0)
      , m_mipmapLevels(0)
      , m_numLayers(false)
    {
    }

    Image::~Image()
    {
    }

    Image::Image( size_t width
                , size_t height
                , PixelFormat pixelFormat
                , DataType dataType
                , void const* const* data   //if numLayers == 1 and mipmapLevels == 1 then this is taken as singal image payload pointer
                , size_t numLayers
                , size_t mipmapLevels )
      : m_width(width)
      , m_height(height)
      , m_pixelFormat(pixelFormat)
      , m_dataType(dataType)
      , m_totalSize(0)
      , m_numLayers(numLayers)
      , m_mipmapLevels(mipmapLevels)
    {
      m_bpp = getSizeOf(m_dataType)*getComponentCount(m_pixelFormat);
      if( mipmapLevels > 0 || numLayers > 1 )
      {
        setEntireData(width, height, pixelFormat, dataType, data, numLayers, mipmapLevels);
      }
      else
      {
        setSingleLayerData(width, height, pixelFormat, dataType, (const void*)data );
      }
    }

    bool Image::operator==( const Image& rhs )
    {
      if( this == &rhs )
      {
        return true;
      }

      if(  getWidth() != rhs.getWidth()
        || getHeight() != rhs.getHeight()
        || getBytesPerPixel() != rhs.getBytesPerPixel()
        || getNumLayers() != rhs.getNumLayers()
        || getMipmapLevels() != rhs.getMipmapLevels() )
      {
        return false;
      }

      for( size_t i = 0; i < m_data.size(); ++i )
      {
        size_t curMipmapLevel = m_mipmapLevels ? i%m_mipmapLevels : 0;
        size_t curMipmapDivisor = (1 << curMipmapLevel);
        size_t curWidth = m_width  / curMipmapDivisor;
        size_t curHeight = m_height / curMipmapDivisor;
        size_t curStride = m_bpp*m_width;
        size_t curSize = curStride*m_height;


        const char* bitsA = m_data[i].get();
        const char* bitsB = rhs.m_data[i].get();

        for( size_t ih = 0; ih < curHeight; ++ih )
        {
          if( memcmp( bitsA + ih*curStride, bitsB + ih*curStride, curStride) )
          {
            return false;
          }
        }
      }

      return true;
    }

    bool Image::operator!=( const Image& rhs )
    {
      return !(*this == rhs);
    }

    const void* Image::getLayerData( size_t face /*= 0*/, size_t mipmapLevel /*= 0*/ ) const
    {
      if( m_width == 0 )
      {
        return nullptr;
      }

      size_t i = face * m_mipmapLevels + mipmapLevel;
      DP_ASSERT( i < m_data.size() );

      return m_data[i].get();
    }

    std::vector<const void*> Image::getLayerDataArray( size_t faceStart, size_t faceEnd
                                                     , size_t mipmapStart, size_t mipmapEnd ) const
    {
      DP_ASSERT(faceEnd > faceStart);
      DP_ASSERT(mipmapEnd > mipmapStart);

      if( faceEnd == ~0 )
      {
        faceEnd = m_numLayers - 1;
      }
      if( mipmapEnd == ~0 )
      {
        mipmapEnd = m_mipmapLevels ? m_mipmapLevels - 1 : 0;
      }

      DP_ASSERT(faceEnd < m_numLayers);
      DP_ASSERT( !m_mipmapLevels || mipmapEnd < m_mipmapLevels );

      std::vector<const void*> out;

      for( size_t curLayer = faceStart; curLayer <= faceEnd; ++curLayer )
      {
        for( size_t curMipmapLevel = mipmapStart; curMipmapLevel <= mipmapEnd; ++curMipmapLevel )
        {
          out.push_back( m_data[curLayer*m_mipmapLevels + curMipmapLevel].get() );
        }
      }

      return out;
    }

    void Image::setSingleLayerData( size_t width
                                  , size_t height
                                  , PixelFormat pixelFormat
                                  , DataType dataType
                                  , const void* data )
    {
      m_width = width;
      m_height = height;
      m_pixelFormat = pixelFormat;
      m_dataType = dataType;
      m_numLayers = 1;
      m_mipmapLevels = 0;
      m_bpp = getSizeOf(m_dataType)*getComponentCount(m_pixelFormat);

      if( !m_data.empty() )
      {
        m_data.clear();
      }

      m_totalSize = m_bpp*m_width*m_height;
      m_data.push_back( boost::shared_array<char>(new char[m_totalSize]) );
      memcpy( m_data[0].get(), data, m_totalSize );

    }

    void Image::setEntireData( size_t width
                             , size_t height
                             , PixelFormat pixelFormat
                             , DataType dataType
                             , void const* const* data
                             , size_t numLayers
                             , size_t mipmapLevels )
    {
      m_width = width;
      m_height = height;
      m_pixelFormat = pixelFormat;
      m_dataType = dataType;
      m_numLayers = numLayers;
      m_mipmapLevels = mipmapLevels;

      m_totalSize = 0;
      size_t imagesPerLayer = m_mipmapLevels ? m_mipmapLevels : 1;

      m_bpp = getSizeOf(m_dataType)*getComponentCount(m_pixelFormat);

      if( !m_data.empty() )
      {
        m_data.clear();
      }
      m_data.resize( imagesPerLayer * numLayers );

      if( !data )
      {
        return;
      }

      //The image payloads in the m_data[] array are stored in face-major mipmap-minor order
      for( unsigned int i = 0; i < m_data.size(); ++i )
      {
        size_t curMipmapLevel = i%imagesPerLayer;
        size_t curMipmapDivisor = 1 << curMipmapLevel;
        size_t curWidth = m_width  / curMipmapDivisor;
        size_t curHeight = m_height / curMipmapDivisor;
        size_t curSize = m_bpp*curWidth*curHeight;

        m_data[i].reset( new char[curSize] );
        memcpy( m_data[i].get(), data[i], curSize );

        m_totalSize += curSize;
      }
    }

#if defined(HAVE_IL)
    static inline DataType convertDT_IL2DP( ILint ildt )
    {
      switch( ildt )
      {
      case IL_UNSIGNED_BYTE:
        return DataType::UNSIGNED_INT_8;
      case IL_UNSIGNED_SHORT:
        return DataType::UNSIGNED_INT_16;
      case IL_UNSIGNED_INT:
        return DataType::UNSIGNED_INT_32;
      case IL_BYTE:
        return DataType::INT_8;
      case IL_SHORT:
        return DataType::INT_16;
      case IL_INT:
        return DataType::INT_32;
      case IL_HALF:
        return DataType::FLOAT_16;
      case IL_FLOAT:
        return DataType::FLOAT_32;
      case IL_DOUBLE:
        return DataType::FLOAT_64;
      default:
        DP_ASSERT(!"Unsupported");
        return DataType::UNKNOWN;
      }
    }

    static inline ILint convertDT_DP2IL( DataType dpdt )
    {
      switch( dpdt )
      {
      case DataType::UNSIGNED_INT_8:
        return IL_UNSIGNED_BYTE;
      case DataType::UNSIGNED_INT_16:
        return IL_UNSIGNED_SHORT;
      case DataType::UNSIGNED_INT_32:
        return IL_UNSIGNED_INT;
      case DataType::INT_8:
        return IL_BYTE;
      case DataType::INT_16:
        return IL_SHORT;
      case DataType::INT_32:
        return IL_INT;
      case DataType::FLOAT_16:
        return IL_HALF;
      case DataType::FLOAT_32:
        return IL_FLOAT;
      case DataType::FLOAT_64:
        return IL_DOUBLE;
      default:
        DP_ASSERT(!"Unsupported");
        return 0;
      }
    }

    static inline PixelFormat convertPF_IL2DP( ILint ilpf )
    {
      switch( ilpf )
      {
      case IL_RGB:
        return PixelFormat::RGB;
      case IL_RGBA:
        return PixelFormat::RGBA;
      case IL_BGR:
        return PixelFormat::BGR;
      case IL_BGRA:
        return PixelFormat::BGRA;
      case IL_COLOUR_INDEX:
        return PixelFormat::LUMINANCE;
      case IL_LUMINANCE:
        return PixelFormat::LUMINANCE;
      case IL_ALPHA:
        return PixelFormat::ALPHA;
      case IL_LUMINANCE_ALPHA:
        return PixelFormat::LUMINANCE_ALPHA;
      default:
        DP_ASSERT(!"Unsupported");
        return PixelFormat::UNKNOWN;
      }
    }

    static inline ILint convertPF_DP2IL( PixelFormat ilpf )
    {
      switch( ilpf )
      {
      case PixelFormat::RGB:
        return IL_RGB;
      case PixelFormat::RGBA:
        return IL_RGBA;
      case PixelFormat::BGR:
        return IL_BGR;
      case PixelFormat::BGRA:
        return IL_BGRA;
      case PixelFormat::R:
        return IL_COLOR_INDEX;
      case PixelFormat::RG:
        return IL_LUMINANCE_ALPHA;
      case PixelFormat::LUMINANCE:
        return IL_LUMINANCE;
      case PixelFormat::ALPHA:
        return IL_ALPHA;
      case PixelFormat::LUMINANCE_ALPHA:
        return IL_LUMINANCE_ALPHA;
      case PixelFormat::STENCIL_INDEX:
        return IL_COLOR_INDEX;
      default:
        DP_ASSERT(!"Unsupported");
        return 0;
      }
    }
#endif

    bool imageToFile(const ImageSharedPtr& image, std::string filename, bool layersAsFaces)
    {
#if defined(HAVE_IL)
      unsigned int imageID;
      ilGenImages( 1, (ILuint *) &imageID );
      ilBindImage( imageID );

      std::string fext = dp::util::getFileExtension( filename );

      bool isDDS = !_stricmp(".DDS", fext.c_str());   // .dds needs special handling
      if ( isDDS )
      {
        // DirectDraw Surfaces have their origin at upper left
        ilEnable( IL_ORIGIN_SET );
        ilOriginFunc( IL_ORIGIN_UPPER_LEFT );
      }
      else
      {
        ilDisable( IL_ORIGIN_SET );
      }

      // DevIL does not know how to handle .jps and .pns. Since those formats are just renamed .jpgs and .pngs
      // pass over filename.(jps|pns).(jpg|png) and rename the file after saving it.
      // FIXME Sent bug report to DevIL. Remove this once jps/pns is added to DevIL.
      bool isStereoFormat = false;

      std::string saveFilename;
      if (!_stricmp(".JPS", fext.c_str()))
      {
        isStereoFormat = true;
        saveFilename = dp::util::replaceFileExtension( filename, ".jpg" );
      }
      else if (!_stricmp(".PNS", fext.c_str()))
      {
        isStereoFormat = true;
        saveFilename = dp::util::replaceFileExtension( filename, ".png" );
      }
      else
      {
        saveFilename = filename;
      }

      // image dimension in pixels
      unsigned int width = dp::checked_cast<unsigned int>(image->getWidth());
      unsigned int height = dp::checked_cast<unsigned int>(image->getHeight());
      size_t bpp = image->getBytesPerPixel();
      size_t numLayers = image->getNumLayers();
      size_t numMipmaps = image->getMipmapLevels();
      size_t singleLayerSize = width*height*bpp;

      // Pixel format and Data Type
      ILint format = convertPF_DP2IL( image->getPixelFormat() );
      ILint dataType = convertDT_DP2IL( image->getDataType() );

      // DevIL frequently loses track of the current state
      ilBindImage(imageID);
      if( layersAsFaces )
      {
        ilActiveImage(0);
      }

      DP_ASSERT(IL_NO_ERROR == ilGetError());

      vector<const void*> rawData = image->getLayerDataArray();

      for( ILint face = 0; face < numLayers; ++face )
      {
        // DevIL frequently loses track of the current state
        //
        ilBindImage(imageID);
        if( layersAsFaces )
        {
          ilActiveImage(0);
          ilActiveFace(face);
        }
        else
        {
          ilActiveImage(face);
          ilActiveFace(0);
        }

        //specify the IL image
        ilTexImage( width, height, face+1, dp::checked_cast<ILubyte>(image->getBytesPerPixel()), format, dataType, nullptr );
        //Set the data
        void* destpixels = ilGetData();
        memcpy( destpixels, rawData[face], singleLayerSize );
      }

      // By default, always overwrite
      ilEnable(IL_FILE_OVERWRITE);
      if ( ilSaveImage( (const ILstring)saveFilename.c_str() ) )
      {
        // For stereo formats rename the file to the original filename
        if (isStereoFormat)
        {
          // Windows will not rename a file if the destination filename already does exist.
          remove( filename.c_str() );
          rename( saveFilename.c_str(), filename.c_str() );
        }

        ilDeleteImages(1, &imageID);
        return true;
      }

      // clean up errors
      while( ilGetError() != IL_NO_ERROR )
      {}

      // free all resources associated with the DevIL image
      ilDeleteImages(1, &imageID);
#endif
      return false;
    }

    ImageSharedPtr imageFromFile( const string & filename, bool layersAsFaces )
    {
#if defined(HAVE_IL)
      unsigned int imageID;
      ilGenImages( 1, (ILuint *) &imageID );
      ilBindImage( imageID );

      if ( fileExists( filename ) )
      {
        ilEnable( IL_ORIGIN_SET );
        bool isDDS = !_stricmp(".DDS", dp::util::getFileExtension( filename ).c_str());   // .dds needs special handling
        ilOriginFunc( isDDS ? IL_ORIGIN_UPPER_LEFT : IL_ORIGIN_LOWER_LEFT );

        ilLoadImage( (const ILstring) filename.c_str() );

        // DevIL frequently loses track of the current state
        //
        ilBindImage(imageID);
        if( layersAsFaces )
        {
          ilActiveImage(0);
        }
        //
        DP_ASSERT(IL_NO_ERROR == ilGetError());


        // image dimension in pixels
        int width = ilGetInteger( IL_IMAGE_WIDTH );
        int height = ilGetInteger( IL_IMAGE_HEIGHT );
        int bpp = ilGetInteger( IL_IMAGE_BPP );

        // Pixel format and Data Type
        PixelFormat format = convertPF_IL2DP( ilGetInteger(IL_IMAGE_FORMAT) );
        DataType dataType = convertDT_IL2DP( ilGetInteger(IL_IMAGE_TYPE) );

        ILint numLayers = layersAsFaces ? ilGetInteger(IL_NUM_FACES)+1 : ilGetInteger(IL_NUM_IMAGES)+1;
        ILint numMipmaps = ilGetInteger(IL_NUM_MIPMAPS);

        std::vector< char* > payload;

        for( ILint face = 0; face < numLayers; ++face )
        {
          for( ILint mipmap = 0; mipmap < (numMipmaps ? numMipmaps : 1); ++mipmap )
          {
            // DevIL frequently loses track of the current state
            //
            ilBindImage(imageID);
            if( layersAsFaces )
            {
              ilActiveImage(0);
              ilActiveFace(face);
            }
            else
            {
              ilActiveImage(face);
              ilActiveFace(0);
            }

            ilActiveMipmap(mipmap);

            //Get the data
            payload.push_back( (char* )ilGetData() );
          }
        }

        int ilerror = ilGetError();
        ImageSharedPtr loadedImage = Image::create( width, height, format, dataType, numLayers + numMipmaps > 1 ? (void const* const*)&payload[0] : (void const* const*)payload[0], numLayers, numMipmaps );

        // free all resources associated with the DevIL image
        // note: free this memory before we eventually flip the image (that will allocate more memory)
        ilDeleteImages(1, &imageID);
        DP_ASSERT(IL_NO_ERROR == ilGetError());

        return( loadedImage );
      }
      else
      {
        std::cerr << "ERROR: load image failed!\n";
        goto ERROREXIT;
      }

ERROREXIT:
      // free all resources associated with the DevIL image
      ilDeleteImages(1, &imageID);
      DP_ASSERT(IL_NO_ERROR == ilGetError());
#endif
      return( ImageSharedPtr() );
    }

#if defined(HAVE_IL)
    class DevILInitShutdown
    {
    public:
      DevILInitShutdown();
      virtual ~DevILInitShutdown();
    };

    typedef dp::util::Singleton<DevILInitShutdown> GlobalDeVILInitShutdown;

    DevILInitShutdown::DevILInitShutdown()
    {
      ilInit();
    }

    DevILInitShutdown::~DevILInitShutdown()
    {
      ilShutDown();
    }

    static DevILInitShutdown* __dis = GlobalDeVILInitShutdown::instance();
#endif

  } // namespace util
} // namespace dp
