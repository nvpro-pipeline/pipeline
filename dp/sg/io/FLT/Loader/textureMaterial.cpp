// Copyright NVIDIA Corporation 2002-2005
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


//
// textureMaterial.cpp
//

#include <dp/sg/core/nvsgapi.h>
#include <dp/sg/core/nvsg.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/io/IO.h>
#include <flt.h>
#include "FLTLoader.h"
#include "FLTTextureAttribute.h"
#include "StackedAtlas.h"

// NVSG types used
#include <dp/sg/core/TextureHost.h>

#include <algorithm>

using namespace dp::math;
using namespace dp::util;
using namespace dp::sg::core;

using std::make_pair;
using std::map;
using std::pair;
using std::string;

static inline TextureWrapMode
RepetitionType( uint32 value )
{
  switch( value )
  {
    case FLTATTR_REP_REPEAT:
      return TWM_REPEAT;
    case FLTATTR_REP_CLAMP:
      return TWM_CLAMP_TO_EDGE;
    case FLTATTR_REP_MIRROR:
      return TWM_MIRROR_REPEAT;
    default:
      return TWM_REPEAT;
  }
}

// check for env map
static inline bool
CheckForEnvMap( FltFile * flt, int16 tMI )
{
  if( tMI != -1 )
  {
    FltTextureMapping * tm = fltLookupTextureMapping( flt, tMI );

    if( tm && tm->type == FLTTMP_ENVMAP )
    {
      return true;
    }
  }

  return false;
}

//
// Adapted from nvtexture.h in NVSDK
//
// Creates an unsigned normalization cube map with no mipmaps
//
// Dimension is expected to be a power of two
//
ParameterGroupDataSharedPtr FLTLoader::createNormalizationCubeMap( unsigned int dimension )
{
  TextureHostSharedPtr tih = TextureHost::create();
  {
    std::vector<uint8> data( dimension * dimension * 3 );

    // the cubemap will be created in this order
    //    GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB
    //  , GL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB
    //  , GL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB
    //  , GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB
    //  , GL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB
    //  , GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB 

    for (int i = 0; i < 6; i++)
    {
      Vec3f Normal;
      float w,h;

      for (unsigned int y = 0; y < dimension; y++)
      {
        h = (float)y / ((float)(dimension - 1));
        h *= 2.0f;
        h -= 1.0f;

        for (unsigned int x = 0; x < dimension; x++)
        {
          w = (float)x / ((float)(dimension - 1));
          w *= 2.0f;
          w -= 1.0f;

          switch( i ) 
          {
            case 0:
              setVec(Normal,1.0f, -h, -w);
              break;
            case 1:
              setVec(Normal,-1.0f, -h, w);
              break;
            case 2:
              setVec(Normal,w, 1.0f, h);
              break;
            case 3:
              setVec(Normal,w, -1.0f, -h);
              break;
            case 4:
              setVec(Normal,w, -h, 1.0f);
              break;
            case 5:
              setVec(Normal,-w, -h, -1.0f);
              break;
          }

          Normal.normalize();

          // Scale to be a color from 0 to 255 (127.5 is 0)
          Normal += Vec3f(1.0f, 1.0f, 1.0f);
          Normal *= 127.5f;

          // Store the color
          unsigned int loc = y * dimension + x * 3;

          data[ loc + 0 ] = (uint8)Normal[0];
          data[ loc + 1 ] = (uint8)Normal[1];
          data[ loc + 2 ] = (uint8)Normal[2];
        }
      }

      // add the new image
      unsigned int imgIndex = tih->addImage( dimension, dimension, 1, Image::IMG_RGB
                                          , Image::IMG_UNSIGNED_BYTE );

      // now store the image
      tih->setImageData( imgIndex, (void *)&data[0] );
      tih->setTextureTarget( TT_TEXTURE_CUBE );
    }
  }

  SamplerSharedPtr sampler = Sampler::create( tih );
  sampler->setMinFilterMode( TFM_MIN_LINEAR );
  sampler->setWrapModes( TWM_CLAMP_TO_EDGE, TWM_CLAMP_TO_EDGE, TWM_CLAMP_TO_EDGE );

  ParameterGroupDataSharedPtr texture = createStandardTextureParameterData( sampler );
  DP_VERIFY( texture->setParameter<dp::fx::EnumSpec::StorageType>( "envMode", TEM_REPLACE ) );

  return( texture );
}

ParameterGroupDataSharedPtr FLTLoader::loadAtlasTexture( FltFile * flt, const std::string & file )
{
  std::map<std::string, ParameterGroupDataSharedPtr>::iterator iter;
  bool found = false;

  iter = m_atlasCache.find( file );

  if( iter != m_atlasCache.end() )
  {
    return (*iter).second;
  }
  else
  {
    char fname[256];
    char searchName[256];

    // make sure it is not on the "not found" list first
    std::vector< std::string >::iterator fnf = std::find( m_texturesNotFound.begin(), m_texturesNotFound.end(), file );

    if( fnf != m_texturesNotFound.end() )
    {
      // do not report this as being missing again
      return 0;
    }

    strcpy( searchName, file.c_str() );

    if( fltFindFile( flt, searchName, fname ) )
    {
      found = true;
    }

    if( found )
    {
      m_numTextures++;

      std::vector< std::string > emptyPath;

      // first, create texture from file name
      TextureHostSharedPtr textureHost = dp::sg::io::loadTextureHost( fname, emptyPath );

      if( textureHost )
      {
        textureHost->setTextureTarget( TT_TEXTURE_2D );

        // build mipmaps if requested
        if( m_buildMipmapsAtLoadTime )
        {
          // use existing mips if available
          if ( textureHost->getNumberOfMipmaps() == 0 )
          {
            textureHost->createMipmaps();
          }
        }

        SamplerSharedPtr sampler = Sampler::create( textureHost );
        sampler->setMinFilterMode( TFM_MIN_LINEAR_MIPMAP_LINEAR );
        sampler->setWrapMode( TWCA_S, TWM_CLAMP_TO_EDGE );
        sampler->setWrapMode( TWCA_T, TWM_CLAMP_TO_EDGE );

        textureHost->incrementMipmapUseCount(); // Flag texture use mipmaps.

        ParameterGroupDataSharedPtr texture = createStandardTextureParameterData( sampler );
        texture->setName( fname );

        // add to cache
        m_atlasCache[ file ] = texture;

        return texture;
      }
      else
      {
        printf("Unable to load: \"%s\"\n", fname );
      }
    }
  }

  // not found!
  INVOKE_CALLBACK(onFileNotFound( file ));

  // push it on the list to make sure we don't report it as being missing
  // again.
  m_texturesNotFound.push_back( file );

  return 0;
}

//
// Lookup a texture file
//

bool
FLTLoader::lookupTexture( TextureEntry & tent, 
                          FltFile * flt, uint16 indx, int16 tmIdx, 
                          LookupEnum lookup )
{
  // set everything to default to start with
  tent.textureParameters.reset();
  tent.layer = -1;
  tent.uscale = tent.vscale = 1.f;
  tent.utrans = tent.vtrans = 0.f;

  // do not try to look up invalid textures
  if( indx == 0xffff )
    return false;

  std::map<mapEntryType,
  std::pair<FLTTextureAttribute,ParameterGroupDataSharedPtr> >::iterator iter;

  iter = m_textureCache.find( cvtToMapEntry( flt, indx, tmIdx ) );

  if( iter != m_textureCache.end() )
  {
    if( lookup == LOOKUP_DETAIL )
    {
      tent.uscale = (*iter).second.first.detailRepeatM;
      tent.vscale = (*iter).second.first.detailRepeatN;
    }

    tent.textureParameters = (*iter).second.second;

    return true;
  }
  else
  {
    FltTexture * t = fltLookupTexture( flt, indx );

    if( t )
    {
      char fname[256];
      char searchName[256];
      bool found = false;

      // make sure it is not on the "not found" list first
      std::vector< std::string >::iterator fnf = 
                    std::find( m_texturesNotFound.begin(),
                               m_texturesNotFound.end(),
                               std::string( t->ID ) );

      if( fnf != m_texturesNotFound.end() )
      {
        // do not report this as being missing again
        return false;
      }

      //
      // Check to see if we are using a texture atlas, and if so if we have
      // a match with this file.  Do not take detail textures or bump maps
      // from the atlas yet.
      //
      if( (lookup != LOOKUP_DETAIL) && (lookup != LOOKUP_NORMALMAP) &&
          getTaiManager() )
      {
        const TaiEntry * entry = getTaiManager()->findEntry( t->ID );

        if( entry )
        {
          // load the atlas if necessary
          tent.textureParameters = loadAtlasTexture( flt, entry->atlasFile );

          if( tent.textureParameters != 0 )
          {
            // values from the atlas
            // may have to flip these for OpenGL origin?
            tent.uscale = entry->width; 
            tent.vscale = entry->height;
            tent.utrans = entry->woffset;
            tent.vtrans = entry->hoffset;

            return true;
          }
        }
      }

      //
      // If we are building stacked atlases, try to find the file there as well.
      // If not, then find it the normal way.  No detail textures or normal
      // maps yet because we don't handle the texenv modes correctly in
      // the shader.
      //
      if( (lookup != LOOKUP_DETAIL) && (lookup != LOOKUP_NORMALMAP) &&
          getStackedAtlasManager() )
      {
        int layer;
        if( tent.textureParameters = getStackedAtlasManager()->submitTexture( t->ID, "", layer ) )
        {
          tent.layer = layer;

          // found, submitted, updated
          return true;
        }
      }

      strcpy( searchName, t->ID );

      if( fltFindFile( flt, searchName, fname ) )
      {
        found = true;
      }

      if( found )
      {
        m_numTextures++;

        std::vector< std::string > emptyPath;
        FLTTextureAttribute texAttrs;
        bool isEnvMap = false;

        if( lookup == LOOKUP_DETAIL )
        {
          // set some defaults, just in case
          texAttrs.detailRepeatM = 5;
          texAttrs.detailRepeatN = 5;

          tent.uscale = 5.f;
          tent.vscale = 5.f;
        }
        else
        {
          // check to see if this is environment mapped
          isEnvMap = CheckForEnvMap( flt, tmIdx );
        }

        // first, create texture from file name
        TextureHostSharedPtr tex = dp::sg::io::loadTextureHost( fname, emptyPath );

        if( tex )
        {
          // create file.attr
          strcpy( searchName, fname );
          strcat( searchName, ".attr" );

          // read attr file
          FltTxAttributes * attrs = fltLoadAttributes( searchName );

          TextureWrapMode wrapS = TWM_REPEAT, 
                                wrapT = TWM_REPEAT;
          TextureEnvMode envMode = TEM_MODULATE;

          if( attrs )
          {
            if( attrs->repetitionU == FLTATTR_REP_NONE ||
                attrs->repetitionV == FLTATTR_REP_NONE )
            {
              wrapS = wrapT = RepetitionType( attrs->repetitionType );
            }
            else
            {
              wrapS = RepetitionType( attrs->repetitionU );
              wrapT = RepetitionType( attrs->repetitionV );
            }

            //
            // EnvMode
            //
            switch( attrs->environmentType )
            {
              case FLTATTR_ENV_MODULATE:
                envMode = TEM_MODULATE;
                break;

              case FLTATTR_ENV_BLEND:
                envMode = TEM_BLEND;
                break;

              case FLTATTR_ENV_DECAL:
                envMode = TEM_DECAL;
                break;

              case FLTATTR_ENV_REPLACE:
                envMode = TEM_REPLACE;
                break;

              case FLTATTR_ENV_ADD:
                envMode = TEM_ADD;
                break;
            }

            // check for detail texture
            if( lookup == LOOKUP_DETAIL && attrs->detailTexture )
            {
              texAttrs.detailRepeatM = attrs->detailM;
              texAttrs.detailRepeatN = attrs->detailN;

              tent.uscale = texAttrs.detailRepeatM;
              tent.vscale = texAttrs.detailRepeatN;
            }

            fltFreeAttributes( attrs );
          }

          // build mipmaps if requested
          if( m_buildMipmapsAtLoadTime )
          {
            // use existing mips if available
            if( tex->getNumberOfMipmaps() == 0 )
            {
              tex->createMipmaps();
            }
          }

          SamplerSharedPtr sampler = Sampler::create( tex );
          sampler->setMinFilterMode( TFM_MIN_LINEAR_MIPMAP_LINEAR );
          sampler->setWrapMode( TWCA_S, wrapS );
          sampler->setWrapMode( TWCA_T, wrapT );

          tex->incrementMipmapUseCount(); // Flag texture to use mipmaps.

          // next, create a Texture
          ParameterGroupDataSharedPtr texture = createStandardTextureParameterData( sampler );
          texture->setName( fname );

          if ( lookup == LOOKUP_DETAIL )
          {
            DP_VERIFY( texture->setParameter<char>( "envScale", TES_2X ) );
          }
          else if ( lookup == LOOKUP_NORMALMAP )
          {
            DP_VERIFY( texture->setParameter<dp::fx::EnumSpec::StorageType>( "envMode", TEM_DOT3_RGB ) );
          }
          else
          {
            DP_VERIFY( texture->setParameter<dp::fx::EnumSpec::StorageType>( "envMode", envMode ) );
          }

          if ( isEnvMap )
          {
            DP_VERIFY( texture->setParameterArrayElement<dp::fx::EnumSpec::StorageType>( "genMode", TCA_S, TGM_REFLECTION_MAP ) );
            DP_VERIFY( texture->setParameterArrayElement<dp::fx::EnumSpec::StorageType>( "genMode", TCA_T, TGM_REFLECTION_MAP ) );
            DP_VERIFY( texture->setParameterArrayElement<dp::fx::EnumSpec::StorageType>( "genMode", TCA_R, TGM_REFLECTION_MAP ) );
          }

          // add to cache
          m_textureCache[ cvtToMapEntry( flt, indx, tmIdx ) ] = std::make_pair( texAttrs, texture );

          // set return
          tent.textureParameters = texture;

          return true;
        }
        else
        {
          INVOKE_CALLBACK(onInvalidValue(0, "Unknown File Format", 
                                          t->ID , 0));
        }
      }
    }
  }

  FltTexture * t = fltLookupTexture( flt, indx );

#if 0
  NVSG_TRACE_OUT_F(("Texture at index: %d: \"%s\" not found.\n",
    indx, (t)?t->ID:"BAD INDEX"));
#endif

  if( t )
  {
    INVOKE_CALLBACK(onFileNotFound( t->ID ));

    // push it on the list to make sure we don't report it as being missing
    // again.
    m_texturesNotFound.push_back( t->ID );
  }
  else
  {
    INVOKE_CALLBACK(onInvalidValue(0, "Reading Texture", 
                                      "Invalid Texture ID Specified", indx));
  }

  return false;
}

//
// Lookup a material
//
EffectDataSharedPtr const & FLTLoader::lookupMaterial( FltFile * flt, uint16 indx, Vec3f & color, float trans )
{
  EffectDataSharedPtr materialEffect;

  FltMaterial * fmat = fltLookupMaterial( flt, indx );
  map<FltMaterial*,EffectDataSharedPtr>::const_iterator it = m_materialMap.find( fmat );
  if ( it == m_materialMap.end() )
  {
    materialEffect = createStandardMaterialData( Vec3f( fmat->ambientRed * color[0], fmat->ambientGreen * color[1], fmat->ambientBlue * color[2] )
                                               , Vec3f( fmat->diffuseRed * color[0], fmat->diffuseGreen * color[1], fmat->diffuseBlue * color[2] )
                                               , Vec3f( fmat->specularRed * color[0], fmat->specularGreen * color[1], fmat->specularBlue * color[2] )
                                               , fmat->shininess
                                               , Vec3f( fmat->emissiveRed * color[0], fmat->emissiveGreen * color[1], fmat->emissiveBlue * color[2] )
                                               , fmat->alpha * ( 1.0f - trans / 65535.0f ) );
    it = m_materialMap.insert( make_pair( fmat, materialEffect ) ).first;
  }
  return( it->second );
}



