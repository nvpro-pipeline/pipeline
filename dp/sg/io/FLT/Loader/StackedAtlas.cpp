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


#include "StackedAtlas.h"
#include <dp/util/File.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/sg/io/IO.h>
#include <iostream>
#include <sstream>

using namespace dp::sg::core;

StackedAtlas::StackedAtlas( unsigned int width, unsigned int height )
  : m_width( width )
  , m_height(height)
  , m_textureCount(0)
{
}

StackedAtlas::~StackedAtlas()
{
}

const ParameterGroupDataSharedPtr & StackedAtlas::addTexture( TextureHostSharedPtr const& img, int & layer )
{
  if( m_texture == 0 )
  {
    m_texture = createStandardTextureParameterData();
  }

  const dp::fx::SmartParameterGroupSpec & pgs = m_texture->getParameterGroupSpec();
  const SamplerSharedPtr & sampler = m_texture->getParameter<SamplerSharedPtr>( pgs->findParameterSpec( "sampler" ) );
  DP_ASSERT( sampler );
  const TextureSharedPtr & texture = sampler->getTexture();
  if ( texture.isPtrTo<TextureHost>() )
  {
    TextureHostSharedPtr const& textureHost = texture.staticCast<TextureHost>();
    //
    // Extract Mip Maps ( they should be present )
    //
    std::vector< BufferSharedPtr > mipmaps( img->getNumberOfMipmaps() );

    for( unsigned int i = 0; i < img->getNumberOfMipmaps(); i++ )
    {
      mipmaps[i] = img->getPixels(0, i+1);
    }

    unsigned int which = textureHost->addImage( img->getWidth(), img->getHeight(), 1, img->getFormat(), img->getType() );

    textureHost->setImageData( which, img->getPixels(), mipmaps );
    layer = which;
  }
  else
  {
    // just use this one as the base
    sampler->setTexture( img );
    layer = 0;
  }

  m_textureCount++;

  return( m_texture );
}

const ParameterGroupDataSharedPtr & StackedAtlas::getTexture()
{
  return m_texture;
}

//
// StackedAtlasManager
//
StackedAtlasManager::StackedAtlasManager()
  : m_rescale( false )
  , m_createMipmaps( true )
  , m_width(0), m_height(0)
  , m_maxSlices( 512 ) // G80 current value, should be queried
{
}

StackedAtlasManager::~StackedAtlasManager()
{
  std::map< std::string, StackedAtlas * >::iterator saiter;
  
  saiter = m_atlases.begin();

  std::stringstream ss;
  int totalCount = 0;

  while( saiter != m_atlases.end() )
  {
    ss << (*saiter).second->getCount() << " - " << (*saiter).first << std::endl;

    totalCount += (*saiter).second->getCount();

    delete (*saiter).second;
    ++saiter;
  }

  ss << std::endl;
  ss << totalCount << " -> " << m_atlases.size() << " : " << 
    ((float)totalCount / (float)m_atlases.size()) * 100.f << "% reduction" << std::endl;

  std::cout << ss.str();

  //IO::errorMessage( "Msg4U", ss.str(), true );
}

void 
StackedAtlasManager::setMaxSlices( unsigned int maxSlices )
{
  m_maxSlices = maxSlices;
}

void 
StackedAtlasManager::setCreateMipmaps( bool create )
{
  m_createMipmaps = create;
}

void 
StackedAtlasManager::setSearchPaths( const std::vector< std::string > & searchPaths )
{
  m_searchPaths = searchPaths;
}

void 
StackedAtlasManager::setRescaleTextures( bool state, unsigned int width,  
                                         unsigned int height )
{
  m_rescale = state;
  m_width   = width;
  m_height  = height;
}

const ParameterGroupDataSharedPtr & StackedAtlasManager::submitTexture( const std::string & file
                                                                            , const std::string & attrString
                                                                            , int & layer )
{
  //
  // see if we have seen this texture before
  //
  std::map< std::string, std::pair<std::string,int> >::iterator fiter = m_cachedFiles.find( file );

  if ( fiter != m_cachedFiles.end() )
  {
    // it is in the list, return the right atlas
    std::map< std::string, StackedAtlas * >::iterator saiter = m_atlases.find( (*fiter).second.first );

    layer = (*fiter).second.second;
    return (*saiter).second->getTexture();
  }

  // we have not seen this one before, go looking for the file
  static ParameterGroupDataSharedPtr nullTexture;

  TextureHostSharedPtr tex = dp::sg::io::loadTextureHost( file, m_searchPaths );
  if( tex )
  {
    std::stringstream ss;

    bool forceMipmaps = false;
    if( m_rescale )
    {
      // rescale to width, height
      if( m_width != tex->getWidth() || m_height != tex->getHeight() )
      {
        tex->scale( 0, m_width, m_height, 1 );

        forceMipmaps = true;
      }
    }

    // we create mipmaps if there were none
    if( m_createMipmaps || forceMipmaps )
    {
      if( forceMipmaps || tex->getNumberOfMipmaps() < 1 )
      {
        tex->createMipmaps();
      }
    }

    // convert everything to bgra - more efficient for the hardware
    // anyway.
    tex->convertPixelFormat( Image::IMG_BGRA );

    // concatenate width, height and attribute string
    ss << tex->getWidth() << "x" << tex->getHeight() << attrString
      << "-" << numberOfComponents( tex->getFormat() );

    // see if we already have a stack like this
    std::map< std::string, StackedAtlas * >::iterator saiter = m_atlases.find( ss.str() );

    StackedAtlas * atlas = 0;

    if( saiter != m_atlases.end() && (*saiter).second->getCount() < m_maxSlices )
    {
      atlas = (*saiter).second;
    }
    else
    {
      // we either don't have one like this, or it is full, make a new one
      atlas = new StackedAtlas( tex->getWidth(), tex->getHeight() );
      // add to list
      m_atlases[ ss.str() ] = atlas;
    }

    const ParameterGroupDataSharedPtr & texture = atlas->addTexture( tex, layer );

    if ( texture )
    {
      // mark it as being cached
      m_cachedFiles[ file ] = std::make_pair(ss.str(),layer);

      return( texture );
    }
    else
    {
      return( nullTexture );
    }
  }
  else
  {
    // unable to find file...
    return( nullTexture );
  }
}

