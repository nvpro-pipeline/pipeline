// Copyright NVIDIA Corporation 2012
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


#include <test/rix/core/helpers/Textures.h>
#include <test/rix/core/helpers/TextureHelper.h>

#include <dp/util/Image.h>

using namespace std;
using namespace dp::math;

namespace dp
{
  namespace rix
  {
    using namespace core;

    namespace util
    {

      template<typename T>
      void setTextureData( dp::rix::core::Renderer* rix, TextureSharedHandle const & textureHandle, dp::PixelFormat pixelFormat, dp::DataType dataType
                         , dp::rix::util::TextureObjectDataSharedPtr& data )
      {
        unsigned char components;
        bool swapRB = false;

        components = dp::getComponentCount(pixelFormat);

        vector<T> out;
        convertTextureData(data->m_data, out, components, swapRB);
        TextureDataPtr textureDataPtr( &out[0], pixelFormat, dataType );
        rix->textureSetData(textureHandle, textureDataPtr );
      }

      void calculateEyeZ( const math::Vec1f& depthBufVal
                        , math::Vecnt<3, unsigned char>& outVal
                        , void* nearFarData)
      {
        unsigned char intVal = 0;

        //Map the value from [0.0, 1.0] to [-1.0, 1.0]
        DP_ASSERT( 0.0f <= depthBufVal[0] && depthBufVal[0] <= 1.0f);
        float adjustedDepthBufVal = 2.0f * (depthBufVal[0] - 0.5f);

        const std::pair<float, float> nearFarPlanes = *(const std::pair<float, float>*)nearFarData;

        //Reverse the perspective divide, map the homogeneous clip space depth back to eye space, but make the resulting depth positive.
        float depthVal = 2.0f * nearFarPlanes.first * nearFarPlanes.second / ( (nearFarPlanes.second + nearFarPlanes.first) - adjustedDepthBufVal * (nearFarPlanes.second - nearFarPlanes.first) );

        //Map the value from [nearPlane, farPlane] to [0.0, 1.0]
        DP_ASSERT(nearFarPlanes.first <= depthVal && depthVal <= nearFarPlanes.second);
        depthVal = (depthVal - nearFarPlanes.first)/(nearFarPlanes.second - nearFarPlanes.first);

        //Map the value from [0.0, 1.0] to [0, 255], and then reverse it
        //so that the blacker the pixel the greater the depth.
        DP_ASSERT(0.0f <= depthVal && depthVal <= 1.0f);
        intVal = static_cast<unsigned char>(255.0f * depthVal);

        outVal[0] = intVal;
        outVal[1] = intVal;
        outVal[2] = intVal;
      }

      dp::util::ImageSharedPtr getEyeZFromDepthBuffer( const dp::util::ImageSharedPtr& depthBuffer, float nearPlane, float farPlane )
      {
        DP_ASSERT( depthBuffer->getDataType() == dp::DataType::FLOAT_32 );
        DP_ASSERT( getComponentCount( depthBuffer->getPixelFormat() ) == 1 );

        size_t width = depthBuffer->getWidth();
        size_t height = depthBuffer->getHeight();
        size_t numPixels = width * height;

        const math::Vec1f* rawData = ( const math::Vec1f* )depthBuffer->getLayerData();
        std::vector< math::Vec1f > depthPixels(rawData, rawData + numPixels);

        std::vector< math::Vecnt<3, unsigned char> > visualPixels(numPixels);

        std::pair<float, float> nearFarPlanes(nearPlane, farPlane);

        convertPixelData( depthPixels, visualPixels, calculateEyeZ, &nearFarPlanes );

        return dp::util::Image::create(width, height, dp::PixelFormat::RGB, dp::DataType::INT_8, (const void* const* )&visualPixels[0][0] );
      }
      
      void alphaToColors( const math::Vecnt<1, unsigned char>& alphaIn
                        , math::Vecnt<3, unsigned char>& colorOut
                        , void* unused )
      {
        colorOut[0] = alphaIn[0];
        colorOut[1] = alphaIn[0];
        colorOut[2] = alphaIn[0];
      }

      dp::util::ImageSharedPtr getGrayscaleFromAlphaImage( const dp::util::ImageSharedPtr& alphaImage )
      {
        DP_ASSERT( getComponentCount( alphaImage->getPixelFormat() ) == 1 );
        DP_ASSERT( alphaImage->getDataType() == dp::DataType::UNSIGNED_INT_8 );

        size_t width = alphaImage->getWidth();
        size_t height = alphaImage->getHeight();
        size_t numPixels = width * height;

        const math::Vecnt<1, unsigned char>* rawData = ( const math::Vecnt<1, unsigned char>* )alphaImage->getLayerData();
        std::vector< math::Vecnt<1, unsigned char> > alphaPixels(rawData, rawData + numPixels);

        std::vector< math::Vecnt<3, unsigned char> > colorPixels(numPixels);

        convertPixelData( alphaPixels, colorPixels, alphaToColors, nullptr );

        return dp::util::Image::create(width, height, dp::PixelFormat::RGB, dp::DataType::UNSIGNED_INT_8, (const void* const* )&colorPixels[0][0] );
      }
      
      void alphaToColorsRange( const math::Vecnt<1, float>& alphaIn
                             , math::Vecnt<3, unsigned char>& colorOut
                             , void* rangeData )
      {
        std::pair<float, float> range = *(std::pair<float, float>*)rangeData;
        float rangeStretch = range.second - range.first;

        colorOut[0] = static_cast<unsigned char>( 255.0f * dp::math::clamp( alphaIn[0], range.first, range.second ) / rangeStretch );
        colorOut[1] = colorOut[0];
        colorOut[2] = colorOut[0];
      }

      dp::util::ImageSharedPtr getGrayscaleFromAlphaImageFloatRange( const dp::util::ImageSharedPtr& alphaImage, float from, float to )
      {
        DP_ASSERT( getComponentCount( alphaImage->getPixelFormat() ) == 1 );
        DP_ASSERT( alphaImage->getDataType() == dp::DataType::FLOAT_32 );

        size_t width = alphaImage->getWidth();
        size_t height = alphaImage->getHeight();
        size_t numPixels = width * height;

        const math::Vecnt<1, float>* rawData = ( const math::Vecnt<1, float>* )alphaImage->getLayerData();
        std::vector< math::Vecnt<1, float> > alphaPixels(rawData, rawData + numPixels);

        std::vector< math::Vecnt<3, unsigned char> > colorPixels(numPixels);

        std::pair<float, float> range(from, to);

        convertPixelData( alphaPixels, colorPixels, alphaToColorsRange, &range );

        return dp::util::Image::create(width, height, dp::PixelFormat::RGB, dp::DataType::INT_8, (const void* const* )&colorPixels[0][0] );
      }

      TextureSharedHandle generateTexture( dp::rix::core::Renderer* rix 
                                         , dp::rix::util::TextureObjectDataSharedPtr data
                                         , dp::PixelFormat pixelFormat /*= PixelFormat::RGBA */
                                         , dp::DataType dataType /*= DataType::FLOAT_32 */
                                         , InternalTextureFormat internalFormat /*= ITF_RGBA8*/
                                         , bool generateMipmaps /*= false*/)
      {
        TextureDescription textureDescription( TT_2D, internalFormat, pixelFormat, dataType, data->m_size[0], data->m_size[1], 0, 0, generateMipmaps );
        TextureSharedHandle textureHandle = rix->textureCreate( textureDescription );

        switch( dataType )
        {
        case dp::DataType::UNSIGNED_INT_8:
          setTextureData<unsigned char>(rix, textureHandle, pixelFormat, dataType, data);
          break;
        case dp::DataType::UNSIGNED_INT_16:
          setTextureData<unsigned short>(rix, textureHandle, pixelFormat, dataType, data);
          break;
        case dp::DataType::UNSIGNED_INT_32:
          setTextureData<unsigned int>(rix, textureHandle, pixelFormat, dataType, data);
          break;
        case dp::DataType::INT_8:
          setTextureData<char>(rix, textureHandle, pixelFormat, dataType, data);
          break;
        case dp::DataType::INT_16:
          setTextureData<short>(rix, textureHandle, pixelFormat, dataType, data);
          break;
        case dp::DataType::INT_32:
          setTextureData<int>(rix, textureHandle, pixelFormat, dataType, data);
          break;
        case dp::DataType::FLOAT_32:
          setTextureData<float>(rix, textureHandle, pixelFormat, dataType, data);
          break;
        case dp::DataType::FLOAT_64:
          setTextureData<double>(rix, textureHandle, pixelFormat, dataType, data);
          break;
        default:
          DP_ASSERT(!"Unsupported data type");
          break;
        }

        return textureHandle;
      }

      TextureSharedHandle createTextureFromFile( dp::rix::core::Renderer* rix, std::string filename
                                               , InternalTextureFormat internalFormat /*= core::ITF_RGBA8*/
                                               , bool generateMipmaps /*= false*/ )
      {
        dp::util::ImageSharedPtr texImage = dp::util::imageFromFile(filename);
        DP_ASSERT(texImage);

        std::vector<const void*> texData = texImage->getLayerDataArray();
        unsigned int numMipmapLevels = dp::checked_cast<unsigned int>(texImage->getMipmapLevels());
        size_t texWidth = texImage->getWidth();
        size_t texHeight = texImage->getHeight();
        DP_ASSERT( texImage->getNumLayers() == 1 );

        // make textures with height 1 or less a 1D texture
        TextureType tt = TT_2D;
        if ( texHeight <= 1 )
        {
          tt = TT_1D;
          texHeight = 0;
        }

        dp::DataType dataType = texImage->getDataType();
        dp::PixelFormat pixelFormat = texImage->getPixelFormat();

        TextureDescription textureDescription( tt, internalFormat, pixelFormat, dataType, texWidth, texHeight, 0, 0, generateMipmaps || !!numMipmapLevels );
        TextureSharedHandle texture = rix->textureCreate( textureDescription );

        TextureDataPtr textureDataPtr = numMipmapLevels
                                      ? TextureDataPtr( &texData[0], numMipmapLevels, 1, pixelFormat, dataType )
                                      : TextureDataPtr( texData[0], pixelFormat, dataType );

        rix->textureSetData( texture, textureDataPtr );

        return texture;
      }

      TextureSharedHandle createCubemapFromFile( dp::rix::core::Renderer* rix, std::string filename
        , InternalTextureFormat internalFormat /*= core::ITF_RGBA8*/
        , bool generateMipmaps /*= false*/ )
      {
        dp::util::ImageSharedPtr cubeMapImage = dp::util::imageFromFile(filename);
        DP_ASSERT(cubeMapImage);

        dp::DataType dataType = cubeMapImage->getDataType();
        dp::PixelFormat pixelFormat = cubeMapImage->getPixelFormat();
        unsigned int numMipmaps = dp::checked_cast<unsigned int>( cubeMapImage->getMipmapLevels() );
        size_t numFaces = cubeMapImage->getNumLayers();
        size_t texWidth = cubeMapImage->getWidth();
        
        DP_ASSERT(numFaces == 6);
        vector<const void*> rawData = cubeMapImage->getLayerDataArray();

        TextureDescription textureDescription( TT_CUBEMAP, internalFormat, pixelFormat, dataType, texWidth, texWidth, 0, 0, generateMipmaps || !!numMipmaps );
        TextureSharedHandle texture = rix->textureCreate( textureDescription );
        TextureDataPtr textureDataPtr( &rawData[0], numMipmaps, 6, pixelFormat, dataType );
        rix->textureSetData( texture, textureDataPtr );

        return texture;
      }

    } // namespace util
  } // namespace rix
} // namespace dp
