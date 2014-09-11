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

#include <vector>
#include <map>

#include <dp/util/Config.h>
#include <dp/util/SmartPtr.h>
#include <dp/math/Vecnt.h>

namespace dp
{
  namespace util
  {
    namespace generator
    {
      class TextureObjectData : public RCObject
      {
      public:
        DP_UTIL_API TextureObjectData();
        DP_UTIL_API TextureObjectData(const SmartPtr<TextureObjectData>& rhs);
        DP_UTIL_API ~TextureObjectData();

        math::Vec2ui m_size;
        std::vector<math::Vec4f> m_data;
      };
      typedef SmartPtr<TextureObjectData> SmartTextureObjectData;

      // Creates a plain-colored texture
      DP_UTIL_API SmartTextureObjectData createTextureColored(const math::Vec2ui& size     // Dimensions of the texture
                                                             , const math::Vec4f& color ); // The color of the texture
      
      // Creates a checkered texture
      DP_UTIL_API SmartTextureObjectData createTextureCheckered( const math::Vec2ui& size       // Dimensions of the texture
                                                               , const math::Vec2ui& tileCount  // The number of color tiles in both directions
                                                               , const math::Vec4f& oddColor    // The color of the odd color tiles
                                                               , const math::Vec4f& evenColor );// The color of the even color tiles
      
      // Creates a three-colored gradient
      DP_UTIL_API SmartTextureObjectData createTextureGradient( const math::Vec2ui& size            // Dimensions of the texture
                                                              , const math::Vec4f& bottomColor      // The color that is centered along the bottom edge
                                                              , const math::Vec4f& topLeftColor     // The color that is centered on the top left corner
                                                              , const math::Vec4f& topRightColor ); // The color that is centered on the top right corner
      
      // Converts a grayscale height map into a normal map
      DP_UTIL_API SmartTextureObjectData convertHeightMapToNormalMap( const SmartTextureObjectData& heightMap // The height-map to convert
                                                                    , float factor );                         // The maximum virtual height of a texel (would ultimately be relative to texture dimensions)
      
      // Creates a simplex noise heightmap
      DP_UTIL_API SmartTextureObjectData createNoiseTexture( const math::Vec2ui& size   // Dimensions of the texture
                                                           , float frequencyX = 1.0f    // Multiplier of the sampling interval along the U texture coordinate
                                                           , float frequencyY = 1.0f ); // Multiplier of the sampling interval along the V texture coordinate
      
      // Creates a normal map of tiled pyramids
      DP_UTIL_API SmartTextureObjectData createPyramidNormalMap( const math::Vec2ui& size         // Dimensions of the texture
                                                               , const math::Vec2ui& pyramidTiles // The number of pyramids in both directions
                                                               , float pyramidHeight );           // The virtual height of the pyramids (would ultimately be relative to texture dimensions)

    } // namespace generator
  } // namespace util
} // namespace dp
