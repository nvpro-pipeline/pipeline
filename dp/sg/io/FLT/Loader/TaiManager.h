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

#include <stdio.h>
#include <string>
#include <list>

// load a tai file

enum AtlasType
{
  ATLAS_2D,
  ATLAS_3D  // not supported
};

struct TaiEntry
{
  // values read from file
  std::string   textureFile;
  std::string   atlasFile;
  unsigned int  atlasIndex;
  AtlasType     type;
  float         woffset;
  float         hoffset;
  float         doffset;
  float         width;
  float         height;

  void print( void ) const
  {
    printf("%s: %s(%d) %s (%f,%f,%f) (%fx%f)\n", 
        textureFile.c_str(), atlasFile.c_str(),
        atlasIndex, (type==ATLAS_2D)?"2D":"3D",
        woffset, hoffset, doffset,
        width, height );
  }
};

class TaiManager
{
public:

  TaiManager();
  ~TaiManager();
  bool loadFile( const std::string & file );
  const TaiEntry * findEntry( const std::string & file );

private:

  std::list< TaiEntry > m_entries;
};
