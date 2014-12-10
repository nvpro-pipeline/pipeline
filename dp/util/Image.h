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

#include <dp/util/Config.h>
#include <dp/util/SharedPtr.h>
#include <dp/util/Types.h>
#include <boost/shared_array.hpp>
#include <string>

namespace dp
{
  namespace util
  {
    DEFINE_PTR_TYPES( Image );

    class Image       // a 2D bitmap.
    {
    public:
      static DP_UTIL_API ImageSharedPtr create();
      static DP_UTIL_API ImageSharedPtr create( size_t width, size_t height, PixelFormat pixelFormat, DataType dataType, void const* const* data = nullptr, size_t numLayers = 1, size_t mipmapLevels = 0 );
      virtual DP_UTIL_API ~Image();

    public:
      DP_UTIL_API bool operator==( const Image& rhs );
      DP_UTIL_API bool operator!=( const Image& rhs );

      // Reading interface
      size_t getWidth()             const { return m_width; }
      size_t getHeight()            const { return m_height; }
      size_t getBytesPerPixel()     const { return m_bpp; }
      size_t getTotalSize()         const { return m_totalSize; }
      PixelFormat getPixelFormat()  const { return m_pixelFormat; }
      DataType getDataType()        const { return m_dataType; }
      size_t getNumLayers()         const { return m_numLayers; }
      size_t getMipmapLevels()      const { return m_mipmapLevels; }
      
      virtual DP_UTIL_API const void* getLayerData(size_t layer = 0, size_t mipmapLevel = 0) const;
      virtual DP_UTIL_API std::vector<const void*> getLayerDataArray( size_t layerStart = 0, size_t layerEnd = ~0, size_t mipmapStart = 0, size_t mipmapEnd = ~0 ) const;

      //Writing interface
      virtual DP_UTIL_API void setSingleLayerData( size_t width, size_t height, PixelFormat pixelFormat, DataType dataType, const void* data = nullptr );
      virtual DP_UTIL_API void setEntireData( size_t width, size_t height, PixelFormat pixelFormat, DataType dataType, void const* const* data = nullptr, size_t numLayers = 1, size_t mipmapLevels = 0 );

    private:
      //Default constructor
      DP_UTIL_API Image();

      //Constructor to allocate a mipmapped 2D image and/or a cubemap and possibly set payload data
      DP_UTIL_API Image( size_t width, size_t height, PixelFormat pixelFormat, DataType dataType, void const* const* data, size_t numLayers, size_t mipmapLevels );

    private:
      //Payload
      size_t m_width;           // width in pixels
      size_t m_height;          // height in pixels
      size_t m_bpp;             // bytes per pixel
      size_t m_mipmapLevels;
      size_t m_numLayers;
      std::vector< boost::shared_array< char > > m_data;  //Payload

      //Format
      PixelFormat m_pixelFormat;
      DataType m_dataType;

      //Other
      size_t m_totalSize;

    };

    DP_UTIL_API bool            imageToFile(const ImageSharedPtr& image, std::string filename, bool layersAsFaces = true);
    DP_UTIL_API ImageSharedPtr  imageFromFile(const std::string& filename, bool layersAsFaces = true);

  } // namespace util
} // namespace dp

