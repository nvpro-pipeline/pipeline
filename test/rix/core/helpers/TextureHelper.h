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


#pragma once

#include <test/rix/core/helpers/inc/Config.h>
#include <test/rix/core/helpers/Textures.h>
#include <dp/rix/core/RiX.h>

#include <dp/util/Image.h>
#include <dp/math/Matmnt.h>

#include <limits>


namespace dp
{
  namespace rix
  {
    namespace util
    {
      template<typename T>
      void convertTextureData( const std::vector<math::Vec4f>& data, std::vector<T>& out
                             , unsigned char components, bool swapRB )
      {
        out.resize(data.size()*components);

        for(size_t i = 0; i < data.size(); i++)
        {
          for(unsigned char j = 0; j < components; j++)
          {
            out[i*components+j] = std::numeric_limits<T>::is_integer ? T(std::numeric_limits<T>::max()*double(data[i][j])) : T(data[i][j]);
          }
          if(swapRB)
          {
            std::swap( out[i*components], out[i*components+2] );
          }
        }
      }

      //template< typename T1, unsigned int n1, typename T2, unsigned int n2 >
      //typedef void (*FNCONVERTSINGLEPIXEL)( const math::Vecnt<n1, T1>& pixIn, math::Vecnt<n2, T2>& pixOut );
      //void (*pixelCallback)( const math::Vecnt<n1, T1>&, math::Vecnt<n2, T2>& )

      template< typename T1, unsigned int n1, typename T2, unsigned int n2, typename FNPIXELCALLBACK >
      void convertPixelData( const std::vector<math::Vecnt<n1, T1> >& dataIn
                           , std::vector<math::Vecnt<n2, T2> >& dataOut
                           , FNPIXELCALLBACK pixelCallback
                           , void* appData )
      {
        DP_ASSERT( dataIn.size() == dataOut.size() );

        std::vector< math::Vecnt<n1, T1> >::const_iterator itIn = dataIn.begin();
        std::vector< math::Vecnt<n2, T2> >::iterator itOut = dataOut.begin();

        while( itIn != dataIn.end() )
        {
          pixelCallback( *itIn++, *itOut++, appData );
        }
      }


      DPHELPERS_API dp::util::ImageSharedPtr getEyeZFromDepthBuffer( const dp::util::ImageSharedPtr& depthBuffer, float nearPlane, float farPlane );

      DPHELPERS_API dp::util::ImageSharedPtr getGrayscaleFromAlphaImage( const dp::util::ImageSharedPtr& alphaImage );

      DPHELPERS_API dp::util::ImageSharedPtr getGrayscaleFromAlphaImageFloatRange( const dp::util::ImageSharedPtr& alphaImage, float from, float to );

      DPHELPERS_API core::TextureSharedHandle generateTexture( dp::rix::core::Renderer* rix 
                                                             , dp::rix::util::TextureObjectDataSharedPtr data
                                                             , dp::PixelFormat pixelFormat = dp::PF_RGBA
                                                             , dp::DataType dataType = dp::DT_FLOAT_32
                                                             , core::InternalTextureFormat internalFormat = core::ITF_RGBA8
                                                             , bool generateMipmaps = false );

      DPHELPERS_API core::TextureSharedHandle createTextureFromFile( dp::rix::core::Renderer* rix, std::string filename
                                                                   , core::InternalTextureFormat internalFormat = core::ITF_RGBA8
                                                                   , bool generateMipmaps = false );

      DPHELPERS_API core::TextureSharedHandle createCubemapFromFile( dp::rix::core::Renderer* rix
                                                                   , std::string filename
                                                                   , core::InternalTextureFormat internalFormat = core::ITF_RGBA8
                                                                   , bool generateMipmaps = false);
    
    } // namespace util
  } // namespace rix
} // namespace dp
