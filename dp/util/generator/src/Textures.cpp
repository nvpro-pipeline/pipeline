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


#include <dp/math/math.h>
#include <dp/math/Matmnt.h>
#include <dp/util/generator/Textures.h>
#include <dp/util/SimplexNoise1234.h>

namespace dp
{
  namespace util
  {
    using namespace std;
    using namespace math;

    namespace generator
    {

      TextureObjectDataSharedPtr TextureObjectData::create()
      {
        return( std::shared_ptr<TextureObjectData>( new TextureObjectData() ) );
      }

      TextureObjectData::TextureObjectData()
      {
      }

      TextureObjectData::~TextureObjectData()
      {
      }

      TextureObjectDataSharedPtr createTextureColored( const Vec2ui& size, const Vec4f& color )
      {
        TextureObjectDataSharedPtr texture = TextureObjectData::create();
        texture->m_size = size;
        unsigned int length = size[0] * size[1];

        texture->m_data.resize(length);
        
        for(unsigned int i = 0; i < length; i++)
        {
          texture->m_data[i] = color;
        }

        return texture;
      }

      TextureObjectDataSharedPtr createTextureCheckered(const Vec2ui& size, const Vec2ui& tileCount, const Vec4f& oddColor, const Vec4f& evenColor)
      {
        TextureObjectDataSharedPtr texture = TextureObjectData::create();
        texture->m_size = size;
        texture->m_data.resize(size[0] * size[1]);

        for(unsigned int iy = 0; iy < size[1]; iy++)
        {
          unsigned iycheck = tileCount[1] * iy / size[1];
          for(unsigned int ix = 0; ix < size[0]; ix++)
          {
            unsigned ixcheck = tileCount[0] * ix / size[0];
            texture->m_data[ iy * size[1] + ix ] = (iycheck^ixcheck)&1 ? oddColor : evenColor;
          }
        }

        return texture;
      }

      TextureObjectDataSharedPtr createTextureGradient(const Vec2ui& size, const Vec4f& bottomColor, const Vec4f& topLeftColor, const Vec4f& topRightColor)
      {
        TextureObjectDataSharedPtr texture = TextureObjectData::create();
        texture->m_size = size;
        texture->m_data.resize(size[0] * size[1]);

        for(unsigned int iy = 0; iy < size[1]; iy++)
        {
          float weight0 = (float)iy / (size[1] - 1);
          float sumWeight12 = 1.0f - weight0;
          for(unsigned int ix = 0; ix < size[0]; ix++)
          {
            float weight1 = sumWeight12 * ix / (size[0] - 1);
            float weight2 = sumWeight12 - weight1;
            Vec4f color = weight0 * bottomColor + weight1 * topLeftColor + weight2 * topRightColor;
            texture->m_data[ iy * size[1] + ix ] = color;
          }
        }

        return texture;
      }

      TextureObjectDataSharedPtr convertHeightMapToNormalMap( const TextureObjectDataSharedPtr& heightMap
                                                        , float factor ) //Maximum apparent bump depth
      {
        TextureObjectDataSharedPtr normalMap = TextureObjectData::create();
        unsigned int texWidth = heightMap->m_size[0];
        unsigned int texHeight = heightMap->m_size[1];
        normalMap->m_size = heightMap->m_size;
        normalMap->m_data.resize(texWidth * texHeight);

        //Texel lengths in texture space
        Vec2f incr( 2.0f / texWidth, 2.0f / texHeight );
        
        //Loop through all texels and get our tangents and binormals by taking the central differences
        for(unsigned int iy = 0; iy < texHeight; iy++)
        {
          for(unsigned int ix = 0; ix < texWidth; ix++)
          {
            unsigned int curTexel = iy * texWidth + ix;
            unsigned int nextTexelX = ix < texWidth - 1 ? curTexel + 1 : iy * texWidth;
            unsigned int prevTexelX = ix > 0 ? curTexel - 1 : iy * texWidth + texWidth - 1;
            unsigned int nextTexelY = iy < texHeight - 1 ? curTexel + texWidth : ix;
            unsigned int prevTexelY = iy > 0 ? curTexel - texWidth : curTexel + (texHeight - 1) * texWidth;
            
            Vec3f tangent( incr[0]
                         , 0.0f
                         , (heightMap->m_data[nextTexelX][0] - heightMap->m_data[prevTexelX][0]) * factor );

            Vec3f binormal( 0.0f
                          , incr[1]
                          , (heightMap->m_data[nextTexelY][0] - heightMap->m_data[prevTexelY][0]) * factor );

            Vec3f normal = tangent ^ binormal;
            normalize(normal);

            //clamp the normal to [0, 1]
            normalMap->m_data[curTexel] = Vec4f(0.5f, 0.5f, 0.5f, 0.0f) + 0.5f * Vec4f(normal, 0.0f);
          }
        }

        return normalMap;
      }

      TextureObjectDataSharedPtr createNoiseTexture( const math::Vec2ui& size, float frequencyX, float frequencyY )
      {
        TextureObjectDataSharedPtr texture = TextureObjectData::create();
        unsigned int texWidth = size[0];
        unsigned int texHeight = size[1];
        texture->m_size = size;
        texture->m_data.resize(size[0] * size[1]);

        for(unsigned int iy = 0; iy < texHeight; iy++)
        {
          for(unsigned int ix = 0; ix < texWidth; ix++)
          {
            unsigned int curTexel = iy * texWidth + ix;

            float intensity = 0.5f * ( SimplexNoise1234::noise( frequencyX * iy / texHeight - 1.0f, frequencyY * ix / texWidth - 1.0f ) + 1.0f );
            texture->m_data[curTexel][0] = intensity;
            texture->m_data[curTexel][1] = intensity;
            texture->m_data[curTexel][2] = intensity;
            texture->m_data[curTexel][3] = 1.0f;
          }
        }

        return texture;
      }

      TextureObjectDataSharedPtr createPyramidNormalMap( const math::Vec2ui& size
                                                       , const math::Vec2ui& pyramidTiles
                                                       , float pyramidHeight )   //Depth of a pyramid in texel space
      {
        TextureObjectDataSharedPtr texture = TextureObjectData::create();
        unsigned int texWidth = size[0];
        unsigned int texHeight = size[1];
        texture->m_size = size;
        texture->m_data.resize(size[0] * size[1]);

        //Texel lengths in texture space
        Vec2f incr( 1.0f / texWidth, 1.0f / texHeight );

        //Dimensions of one pyramid
        unsigned int pyramidIX = texWidth / pyramidTiles[0];
        unsigned int pyramidIY = texHeight / pyramidTiles[1];

        //Calculate all four occurring normals of the pyramid ahead of time
        Vec4f wNormalLeft   = Vec4f( -pyramidHeight, 0.0f,           0.5f * incr[0] * pyramidIX, 0.0f);
        Vec4f wNormalRight  = Vec4f(  pyramidHeight, 0.0f,           0.5f * incr[0] * pyramidIX, 0.0f);
        Vec4f wNormalTop    = Vec4f(           0.0f, pyramidHeight,  0.5f * incr[1] * pyramidIY, 0.0f);
        Vec4f wNormalBottom = Vec4f(           0.0f, -pyramidHeight, 0.5f * incr[1] * pyramidIY, 0.0f);

        //Normalize our normals
        wNormalLeft.normalize();
        wNormalRight.normalize();
        wNormalTop.normalize();
        wNormalBottom.normalize();

        //Clamp our normals to [0, 1]
        wNormalLeft   = 0.5f*wNormalLeft   + Vec4f(0.5f, 0.5f, 0.5f, 0.0f);
        wNormalRight  = 0.5f*wNormalRight  + Vec4f(0.5f, 0.5f, 0.5f, 0.0f);
        wNormalTop    = 0.5f*wNormalTop    + Vec4f(0.5f, 0.5f, 0.5f, 0.0f);
        wNormalBottom = 0.5f*wNormalBottom + Vec4f(0.5f, 0.5f, 0.5f, 0.0f);

        for(unsigned int iy = 0; iy < texHeight; iy++)
        {
          //Get our vertical texel position relative to the center of the current pyramid tile
          int iyrel = iy % pyramidIY - pyramidIY / 2;

          for(unsigned int ix = 0; ix < texWidth; ix++)
          {
            //Get our horizontal texel position relative to the center of the current pyramid tile
            int ixrel = ix % pyramidIX - pyramidIX / 2;
            unsigned int curTexel = iy * texWidth + ix;

            //Assign the appropriate normal according to what face of the pyramid we're on
            if( iyrel > abs(ixrel) )
            {
              texture->m_data[curTexel] = wNormalTop;
            }
            else if( iyrel > ixrel )
            {
              texture->m_data[curTexel] = wNormalLeft;
            }
            else if( iyrel > -ixrel )
            {
              texture->m_data[curTexel] = wNormalRight;
            }
            else
            {
              texture->m_data[curTexel] = wNormalBottom;
            }

          }
        }

        return texture;
      }

    } // namespace generator
  } // namespace util
} // namespace dp
