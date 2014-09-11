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


#pragma once

#include <string>
#include <vector>
#include <map>
#include <dp/sg/core/ParameterGroupData.h>
#include <dp/sg/core/SharedPtr.h>

class StackedAtlas
{
public:
  StackedAtlas( unsigned int width, unsigned int height );
  ~StackedAtlas();

  unsigned int getWidth() { return m_width; }
  unsigned int getHeight() { return m_height; }
  unsigned int getCount() { return m_textureCount; }

  const dp::sg::core::ParameterGroupDataSharedPtr & addTexture( dp::sg::core::TextureHostSharedPtr const& img, int & );
  const dp::sg::core::ParameterGroupDataSharedPtr & getTexture();

private:

  dp::sg::core::ParameterGroupDataSharedPtr m_texture;
  unsigned int m_width;
  unsigned int m_height;
  unsigned int m_textureCount;
};

class StackedAtlasManager
{
public:
  StackedAtlasManager();
  ~StackedAtlasManager();

  void setMaxSlices( unsigned int maxSlices );
  void setCreateMipmaps( bool create );
  void setSearchPaths( const std::vector< std::string > & searchPaths );
  void setRescaleTextures( bool state, unsigned int width = 0,  
                                       unsigned int height = 0 );

  const dp::sg::core::ParameterGroupDataSharedPtr & submitTexture( const std::string & file
                                                         , const std::string & attrs
                                                         , int & layer );

private:

  bool m_rescale;
  bool m_createMipmaps;
  unsigned int m_width;
  unsigned int m_height;
  unsigned int m_maxSlices;
  std::map< std::string, std::pair<std::string,int> > m_cachedFiles;
  std::map< std::string, StackedAtlas * > m_atlases;
  std::vector< std::string > m_searchPaths;
};
