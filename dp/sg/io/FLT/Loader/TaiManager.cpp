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


#include "TaiManager.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// load a tai file

TaiManager::TaiManager()
{
}

TaiManager::~TaiManager()
{
}

const TaiEntry *
TaiManager::findEntry( const std::string & entry )
{
  std::list< TaiEntry >::iterator iter = m_entries.begin();
  std::string compare;

  // strip leading path
  size_t bs = entry.find_last_of( '\\' );
  size_t fs = entry.find_last_of( '/' );
  size_t pos = ~0;

  if( bs != std::string::npos )
  {
    pos = bs;

    if( fs != std::string::npos )
    {
      if( fs > bs )
      {
        pos = fs;
      }
    }
  }
  else if( fs != std::string::npos )
  {
    pos = fs;
  }
  else
  {
    return 0;
  }

  pos++;

  compare = entry.substr( pos );

  // linear search - should probably sort and bsearch
  while( iter != m_entries.end() )
  {
    if( compare == (*iter).textureFile )
    {
      return &(*iter);
    }

    ++iter;
  }

  return 0;
}

bool
TaiManager::loadFile( const std::string & file )
{
  FILE * f = fopen( file.c_str(), "r" );

  if( !f )
  {
    return false;
  }

  char buf[512];

  while( 1 )
  {
    TaiEntry entry;

    fgets( buf, 512, f );
    if( feof( f ) )
      break;

    //printf("line: \"%s\"\n", buf );

    // skip comments
    if( buf[0] == '#' )
      continue;

    // skip short lines too
    if( strlen( buf ) < 40 )
      continue;

    char * str = buf;
    char * ptr = buf;
    //
    // parse filename
    // 
    while( *ptr != '\t' ) 
      ptr++;

    *ptr++ = 0;
    // skip second tab
    ptr++;
    
    entry.textureFile = str;


    //
    // Parse atlas name
    //
    str = ptr;
    while( *ptr != ',' )
      ptr++;

    *ptr++ = 0;
    // skip white
    ptr++;

    entry.atlasFile = str;

    //
    // Parse atlas index
    //
    str = ptr;
    while( *ptr != ',' )
      ptr++;

    *ptr++ = 0;
    // skip white
    ptr++;

    entry.atlasIndex = atoi( str );

    //
    // Parse atlas type
    //
    str = ptr;
    while( *ptr != ',' )
      ptr++;

    *ptr++ = 0;
    // skip white
    ptr++;

    if( !strcmp( str, "2D" ) )
      entry.type = ATLAS_2D;
    else if( !strcmp( str, "3D" ) )
      entry.type = ATLAS_3D;
    else
      // unknown
      entry.type = ATLAS_2D;

    //
    // Parse woffset
    //
    str = ptr;
    while( *ptr != ',' )
      ptr++;

    *ptr++ = 0;
    // skip white
    ptr++;

    entry.woffset = (float)atof( str );

    //
    // Parse hoffset
    //
    str = ptr;
    while( *ptr != ',' )
      ptr++;

    *ptr++ = 0;
    // skip white
    ptr++;

    entry.hoffset = (float)atof( str );

    //
    // Parse doffset
    //
    str = ptr;
    while( *ptr != ',' )
      ptr++;

    *ptr++ = 0;
    // skip white
    ptr++;

    entry.doffset = (float)atof( str );

    //
    // Parse width
    //
    str = ptr;
    while( *ptr != ',' )
      ptr++;

    *ptr++ = 0;
    // skip white
    ptr++;

    entry.width = (float)atof( str );

    //
    // Parse height
    //
    str = ptr;
    while( (*ptr != 0) && (*ptr != '\n') )
      ptr++;

    // last entry
    *ptr = 0;

    entry.height = (float)atof( str );

    // correct for opengl offsets
    entry.hoffset = 1.0f - (entry.hoffset + entry.height);

    // add entry to the list
    m_entries.push_back( entry );
  }

  fclose( f );

  return (m_entries.size() > 0);
}

#ifdef __TEST__
int
main( int argc, char ** argv )
{
  TaiManager man;

  if( man.loadFile( "Default.tai" ) )
  {
    printf("Success\n");

    if( const TaiEntry * e = man.findEntry( "Textures\\rock.png" ) )
    {
      e->print();
    }
    else
    {
      printf("not found..\n");
    }
  }
  else
  {
    printf("Fail\n");
  }
}
#endif
