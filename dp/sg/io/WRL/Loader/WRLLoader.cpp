// Copyright (c) 2002-2016, NVIDIA CORPORATION. All rights reserved.
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


#include "WRLLoader.h"
#include <dp/Exception.h>
#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/IndexSet.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/io/IO.h>
#include <dp/sg/io/PlugInterfaceID.h>
#include <dp/util/File.h>
#include <dp/util/Locale.h>

#include <iterator>

using namespace dp::sg::core;
using namespace dp::math;
using namespace dp::util;
using namespace vrml;
using dp::util::PlugInCallback;
using dp::util::UPIID;
using dp::util::UPITID;
using dp::util::PlugIn;
using std::vector;
using std::string;
using std::map;
using std::pair;
using std::make_pair;
using std::multimap;

const UPITID PITID_SCENE_LOADER(UPITID_SCENE_LOADER, UPITID_VERSION); // plug-in type
const UPIID  PIID_WRL_SCENE_LOADER(".WRL", PITID_SCENE_LOADER); // plug-in ID

#define readMFColor( mf )     readMFType<SFColor>( mf, &WRLLoader::readSFColor )
#define readMFFloat( mf )     readMFType<SFFloat>( mf, &WRLLoader::readSFFloat )
#define readMFInt32( mf )     readMFType<SFInt32>( mf, &WRLLoader::readSFInt32 )
#define readMFRotation( mf )  readMFType<SFRotation>( mf, &WRLLoader::readSFRotation )
#define readMFString( mf )    readMFType<SFString>( mf, &WRLLoader::readSFString )
#define readMFVec2f( mf )     readMFType<SFVec2f>( mf, &WRLLoader::readSFVec2f )
#define readMFVec3f( mf )     readMFType<SFVec3f>( mf, &WRLLoader::readSFVec3f )

#if defined(_WIN32)
BOOL APIENTRY DllMain(HANDLE hModule, DWORD reason, LPVOID lpReserved)
{
  if (reason == DLL_PROCESS_ATTACH)
  {
    int i=0;
  }

  return TRUE;
}
#elif defined(LINUX)
void lib_init()
{
  int i=0;
}
#endif

bool getPlugInterface(const UPIID& piid, dp::util::PlugInSharedPtr & pi)
{
  if ( piid==PIID_WRL_SCENE_LOADER )
  {
    pi = WRLLoader::create();
    return( !!pi );
  }
  return false;
}

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();

  piids.push_back(PIID_WRL_SCENE_LOADER);
}

string::size_type findBraces( const string &str )
{
  string::size_type endIndex = str.length();
  for ( string::size_type i=0 ; i<endIndex ; i++ )
  {
    char c = str[i];
    if ( ( c == '{' ) || ( c == '}' ) || ( c == '[' ) || ( c == ']' ) )
    {
      return( i );
    }
  }
  return( string::npos );
}

string::size_type findDelimiter( const string &str, string::size_type startIndex )
{
  string::size_type endIndex = str.length();
  bool inString = false;
  for ( string::size_type i=startIndex ; i<endIndex ; i++ )
  {
    char c = str[i];
    if ( c == '"' )
    {
      inString = !inString;
    }
    else if ( !inString && ( ( c == ' ' ) || ( c == '\r' ) || ( c == '\t' ) || ( c == '\n' ) || ( c == ',' ) ) )
    {
      return( i );
    }
  }
  return( string::npos );
}

string::size_type findNotDelimiter( const string &str, string::size_type startIndex )
{
  string::size_type endIndex = str.length();
  for ( string::size_type i=startIndex ; i<endIndex ; i++ )
  {
    char c = str[i];
    if ( ( c != ' ' ) && ( c != '\r' ) && ( c != '\t' ) && ( c != '\n' ) && ( c != ',' ) )
    {
      return( i );
    }
  }
  return( string::npos );
}

SFInt32 max( const MFInt32 &mfInt32 )
{
  SFInt32 m = mfInt32[0];
  for ( size_t i=1 ; i<mfInt32.size() ; i++ )
  {
    if ( m < mfInt32[i] )
    {
      m = mfInt32[i];
    }
  }
  return( m );
}

SFInt32 min( const MFInt32 &mfInt32 )
{
  SFInt32 m = mfInt32[0];
  for ( size_t i=1 ; i<mfInt32.size() ; i++ )
  {
    if ( m > mfInt32[i] )
    {
      m = mfInt32[i];
    }
  }
  return( m );
}

WRLLoaderSharedPtr WRLLoader::create()
{
  return( std::shared_ptr<WRLLoader>( new WRLLoader() ) );
}

WRLLoader::WRLLoader()
: m_eof(false)
, m_strict(false)
, m_nextTokenStart(string::npos)
, m_nextTokenEnd(string::npos)
, m_lineLength(4096)
, m_stepsPerUnit(60)
, m_fh(NULL)
, m_lineNumber(0)
, m_rootNode(nullptr)
, m_scene(nullptr)
, m_smoothTraverser(nullptr)
{
  m_line = (char *) malloc( m_lineLength + 1 );

  // The circular ones used for Cone, Cylinder, and Sphere.
  m_subdivisions.sphereMin     = 12;  // minimum       // 30 degrees
  m_subdivisions.sphereDefault = 36;  // at radius 1.0 // 10 degrees
  m_subdivisions.sphereMax     = 90;  // maximum       //  4 degrees

  // The rectangular ones used for the Box. (It needs a much lower maximum!)
  m_subdivisions.boxMin     = 2;  // minimum
  m_subdivisions.boxDefault = 4;  // at size 1.0! (Box default size is 2.0 though.)
  m_subdivisions.boxMax     = 8;  // maximum

  // The user can define the subdivisions used for build built-in geometry!
  if ( const char * env = getenv( "DP_WRL_SUBDIVISIONS" ) )
  {
    std::string values( env );
    std::string token;
    string::size_type tokenEnd = 0;

    for (int i = 0; i < 6; ++i)
    {
      string::size_type tokenStart = findNotDelimiter( values, tokenEnd );
      if ( tokenStart != string::npos )
      {
        tokenEnd = findDelimiter( values, tokenStart );
        token.assign( values, tokenStart, tokenEnd - tokenStart );
        if ( !token.empty() )
        {
          int value = atoi( token.c_str() );
          switch ( i )
          {
            case 0 : m_subdivisions.sphereMin     = value; break;
            case 1 : m_subdivisions.sphereDefault = value; break;
            case 2 : m_subdivisions.sphereMax     = value; break;
            case 3 : m_subdivisions.boxMin        = value; break;
            case 4 : m_subdivisions.boxDefault    = value; break;
            case 5 : m_subdivisions.boxMax        = value; break;
          }
        }
      }
    }

    // Now make sure the input values are consistent.
    // Absolute minimum required for non-degenerated circular objects is 3.
    if ( m_subdivisions.sphereMin < 3 )
    {
      m_subdivisions.sphereMin = 3;
    }
    if ( m_subdivisions.sphereMax < m_subdivisions.sphereMin )
    {
      m_subdivisions.sphereMax = m_subdivisions.sphereMin;
    }
    // Make sure the subdivision at radius 1.0 is within the limits.
    m_subdivisions.sphereDefault = clamp(m_subdivisions.sphereDefault, m_subdivisions.sphereMin, m_subdivisions.sphereMax);

    // Now the same for the Box:
    // Absolute minimum required for non-degenerated Box object is 2.
    if ( m_subdivisions.boxMin < 2 )
    {
      m_subdivisions.boxMin = 2;
    }
    if ( m_subdivisions.boxMax < m_subdivisions.boxMin )
    {
      m_subdivisions.boxMax = m_subdivisions.boxMin;
    }
    // Make sure the subdivision at size 1.0 is within the limits.
    m_subdivisions.boxDefault = clamp(m_subdivisions.boxDefault, m_subdivisions.boxMin, m_subdivisions.boxMax);
  }
}

WRLLoader::~WRLLoader()
{
  free( m_line );
}

void WRLLoader::createBox( IndexedFaceSetSharedPtr & pIndexedFaceSet, const SFVec3f& size, bool textured )
{
  float width  = size[0];
  float height = size[1];
  float depth  = size[2];

  // Tessellate with square quads when inside the unclamped range.
  int w = clamp( (int) (m_subdivisions.boxDefault * width ), m_subdivisions.boxMin, m_subdivisions.boxMax );
  int h = clamp( (int) (m_subdivisions.boxDefault * height), m_subdivisions.boxMin, m_subdivisions.boxMax );
  int d = clamp( (int) (m_subdivisions.boxDefault * depth ), m_subdivisions.boxMin, m_subdivisions.boxMax );

  size_t numVertices = 2 * (h * d + d * w + h * w);
  size_t numIndices  = 5 * ((h - 1) * (d - 1) + (d - 1) * (w - 1) + (h - 1) * (w - 1));

  CoordinateSharedPtr pCoordinate = Coordinate::create();
  pCoordinate->point.reserve( numVertices ); // vertices
  pIndexedFaceSet->coord = pCoordinate;
  pIndexedFaceSet->coordIndex.reserve( numIndices );

  NormalSharedPtr pNormal = Normal::create();
  pNormal->vector.reserve( numVertices ); // normals
  pIndexedFaceSet->normal = pNormal;
  pIndexedFaceSet->normalIndex.reserve( numIndices );
  pIndexedFaceSet->normalPerVertex = true; // Is the default.

  TextureCoordinateSharedPtr pTextureCoordinate = TextureCoordinateSharedPtr::null;
  if ( textured )
  {
    pTextureCoordinate = TextureCoordinate::create();
    pTextureCoordinate->point.reserve( numVertices ); // texcoords
    pIndexedFaceSet->texCoord = pTextureCoordinate;
    pIndexedFaceSet->texCoordIndex.reserve( numIndices );
  }

  float xCoord;
  float yCoord;
  float zCoord;
  float uCoord;
  float vCoord;

  int indexOffset = 0; // The next sub-object will generate indices starting at this position.

  // Positive x-axis vertices, normals, texcoords:
  xCoord = width * 0.5f;
  for (int lat = 0; lat < h; lat++)
  {
    vCoord = (float) lat / (float) (h - 1); // [0.0, 1.0]
    yCoord = height * (vCoord - 0.5f);      // [-height/2, height/2]

    for (int lon = 0; lon < d; lon++)
    {
      uCoord = (float) lon / (float) (d - 1); // [0.0, 1.0]
      zCoord = -depth * (uCoord - 0.5f);      // [-depth/2, depth/2]

      pCoordinate->point.push_back( SFVec3f( xCoord, yCoord, zCoord ) );
      pNormal->vector.push_back( SFVec3f( 1.0f, 0.0f, 0.0f) );
      if (textured)
      {
        pTextureCoordinate->point.push_back( SFVec2f( uCoord, vCoord ) );
      }
    }
  }
  // Indices:
  for (int lat = 0; lat < h - 1; lat++)
  {
    for (int lon = 0; lon < d - 1; lon++)
    {
      int ll =  lat      * d +  lon     ;  // lower left
      int lr =  lat      * d + (lon + 1);  // lower right
      int ur = (lat + 1) * d + (lon + 1);  // upper right
      int ul = (lat + 1) * d +  lon     ;  // upper left

      ll += indexOffset;
      lr += indexOffset;
      ur += indexOffset;
      ul += indexOffset;

      pIndexedFaceSet->coordIndex.push_back( ll );
      pIndexedFaceSet->coordIndex.push_back( lr );
      pIndexedFaceSet->coordIndex.push_back( ur );
      pIndexedFaceSet->coordIndex.push_back( ul );
      pIndexedFaceSet->coordIndex.push_back( -1 );

      pIndexedFaceSet->normalIndex.push_back( ll );
      pIndexedFaceSet->normalIndex.push_back( lr );
      pIndexedFaceSet->normalIndex.push_back( ur );
      pIndexedFaceSet->normalIndex.push_back( ul );
      pIndexedFaceSet->normalIndex.push_back( -1 );

      if (textured)
      {
        pIndexedFaceSet->texCoordIndex.push_back( ll );
        pIndexedFaceSet->texCoordIndex.push_back( lr );
        pIndexedFaceSet->texCoordIndex.push_back( ur );
        pIndexedFaceSet->texCoordIndex.push_back( ul );
        pIndexedFaceSet->texCoordIndex.push_back( -1 );
      }
    }
  }
  indexOffset += h * d;

  // Positive y-axis vertices, normals, texcoords:
  yCoord = height * 0.5f;
  for (int lat = 0; lat < d; lat++)
  {
    vCoord = (float) lat / (float) (d - 1); // [0.0, 1.0]
    zCoord = -depth * (vCoord - 0.5f);      // [-depth/2, depth/2]

    for (int lon = 0; lon < w; lon++)
    {
      uCoord = (float) lon / (float) (w - 1); // [0.0, 1.0]
      xCoord = width * (uCoord - 0.5f);       // [-width/2, width/2]

      pCoordinate->point.push_back( SFVec3f( xCoord, yCoord, zCoord ) );
      pNormal->vector.push_back( SFVec3f( 0.0f, 1.0f, 0.0f) );
      if (textured)
      {
        pTextureCoordinate->point.push_back( SFVec2f( uCoord, vCoord ) );
      }
    }
  }
  for (int lat = 0; lat < d - 1; lat++)
  {
    for (int lon = 0; lon < w - 1; lon++)
    {
      int ll =  lat      * w +  lon     ;  // lower left
      int lr =  lat      * w + (lon + 1);  // lower right
      int ur = (lat + 1) * w + (lon + 1);  // upper right
      int ul = (lat + 1) * w +  lon     ;  // upper left

      ll += indexOffset;
      lr += indexOffset;
      ur += indexOffset;
      ul += indexOffset;

      pIndexedFaceSet->coordIndex.push_back( ll );
      pIndexedFaceSet->coordIndex.push_back( lr );
      pIndexedFaceSet->coordIndex.push_back( ur );
      pIndexedFaceSet->coordIndex.push_back( ul );
      pIndexedFaceSet->coordIndex.push_back( -1 );

      pIndexedFaceSet->normalIndex.push_back( ll );
      pIndexedFaceSet->normalIndex.push_back( lr );
      pIndexedFaceSet->normalIndex.push_back( ur );
      pIndexedFaceSet->normalIndex.push_back( ul );
      pIndexedFaceSet->normalIndex.push_back( -1 );

      if (textured)
      {
        pIndexedFaceSet->texCoordIndex.push_back( ll );
        pIndexedFaceSet->texCoordIndex.push_back( lr );
        pIndexedFaceSet->texCoordIndex.push_back( ur );
        pIndexedFaceSet->texCoordIndex.push_back( ul );
        pIndexedFaceSet->texCoordIndex.push_back( -1 );
      }
    }
  }
  indexOffset += d * w;

  // Positive z-axis vertices, normals, texcoords:
  zCoord = depth * 0.5f;
  for (int lat = 0; lat < h; lat++)
  {
    vCoord = (float) lat / (float) (h - 1); // [0.0, 1.0]
    yCoord = height * (vCoord - 0.5f);      // [-height/2, height/2]

    for (int lon = 0; lon < w; lon++)
    {
      uCoord = (float) lon / (float) (w - 1); // [0.0, 1.0]
      xCoord = width * (uCoord - 0.5f);       // [-width/2, width/2]

      pCoordinate->point.push_back( SFVec3f( xCoord, yCoord, zCoord ) );
      pNormal->vector.push_back( SFVec3f( 0.0f, 0.0f, 1.0f) );
      if (textured)
      {
        pTextureCoordinate->point.push_back( SFVec2f( uCoord, vCoord ) );
      }
    }
  }
  for (int lat = 0; lat < h - 1; lat++)
  {
    for (int lon = 0; lon < w - 1; lon++)
    {
      int ll =  lat      * w +  lon     ;  // lower left
      int lr =  lat      * w + (lon + 1);  // lower right
      int ur = (lat + 1) * w + (lon + 1);  // upper right
      int ul = (lat + 1) * w +  lon     ;  // upper left

      ll += indexOffset;
      lr += indexOffset;
      ur += indexOffset;
      ul += indexOffset;

      pIndexedFaceSet->coordIndex.push_back( ll );
      pIndexedFaceSet->coordIndex.push_back( lr );
      pIndexedFaceSet->coordIndex.push_back( ur );
      pIndexedFaceSet->coordIndex.push_back( ul );
      pIndexedFaceSet->coordIndex.push_back( -1 );

      pIndexedFaceSet->normalIndex.push_back( ll );
      pIndexedFaceSet->normalIndex.push_back( lr );
      pIndexedFaceSet->normalIndex.push_back( ur );
      pIndexedFaceSet->normalIndex.push_back( ul );
      pIndexedFaceSet->normalIndex.push_back( -1 );

      if (textured)
      {
        pIndexedFaceSet->texCoordIndex.push_back( ll );
        pIndexedFaceSet->texCoordIndex.push_back( lr );
        pIndexedFaceSet->texCoordIndex.push_back( ur );
        pIndexedFaceSet->texCoordIndex.push_back( ul );
        pIndexedFaceSet->texCoordIndex.push_back( -1 );
      }
    }
  }
  indexOffset += h * w;

  // Negative x-axis vertices, normals, texcoords:
  xCoord = -width * 0.5f;
  for (int lat = 0; lat < h; lat++)
  {
    vCoord = (float) lat / (float) (h - 1); // [0.0, 1.0]
    yCoord = height * (vCoord - 0.5f);      // [-height/2, height/2]

    for (int lon = 0; lon < d; lon++)
    {
      uCoord = (float) lon / (float) (d - 1); // [0.0, 1.0]
      zCoord = depth * (uCoord - 0.5f);       // [-depth/2, depth/2]

      pCoordinate->point.push_back( SFVec3f( xCoord, yCoord, zCoord ) );
      pNormal->vector.push_back( SFVec3f( -1.0f, 0.0f, 0.0f) );
      if (textured)
      {
        pTextureCoordinate->point.push_back( SFVec2f( uCoord, vCoord ) );
      }
    }
  }
  // Indices:
  for (int lat = 0; lat < h - 1; lat++)
  {
    for (int lon = 0; lon < d - 1; lon++)
    {
      int ll =  lat      * d +  lon     ;  // lower left
      int lr =  lat      * d + (lon + 1);  // lower right
      int ur = (lat + 1) * d + (lon + 1);  // upper right
      int ul = (lat + 1) * d +  lon     ;  // upper left

      ll += indexOffset;
      lr += indexOffset;
      ur += indexOffset;
      ul += indexOffset;

      pIndexedFaceSet->coordIndex.push_back( ll );
      pIndexedFaceSet->coordIndex.push_back( lr );
      pIndexedFaceSet->coordIndex.push_back( ur );
      pIndexedFaceSet->coordIndex.push_back( ul );
      pIndexedFaceSet->coordIndex.push_back( -1 );

      pIndexedFaceSet->normalIndex.push_back( ll );
      pIndexedFaceSet->normalIndex.push_back( lr );
      pIndexedFaceSet->normalIndex.push_back( ur );
      pIndexedFaceSet->normalIndex.push_back( ul );
      pIndexedFaceSet->normalIndex.push_back( -1 );

      if (textured)
      {
        pIndexedFaceSet->texCoordIndex.push_back( ll );
        pIndexedFaceSet->texCoordIndex.push_back( lr );
        pIndexedFaceSet->texCoordIndex.push_back( ur );
        pIndexedFaceSet->texCoordIndex.push_back( ul );
        pIndexedFaceSet->texCoordIndex.push_back( -1 );
      }
    }
  }
  indexOffset += h * d;

  // Negative y-axis vertices, normals, texcoords:
  yCoord = -height * 0.5f;
  for (int lat = 0; lat < d; lat++)
  {
    vCoord = (float) lat / (float) (d - 1); // [0.0, 1.0]
    zCoord = depth * (vCoord - 0.5f);      // [-depth/2, depth/2]

    for (int lon = 0; lon < w; lon++)
    {
      uCoord = (float) lon / (float) (w - 1); // [0.0, 1.0]
      xCoord = width * (uCoord - 0.5f);       // [-width/2, width/2]

      pCoordinate->point.push_back( SFVec3f( xCoord, yCoord, zCoord ) );
      pNormal->vector.push_back( SFVec3f( 0.0f, -1.0f, 0.0f) );
      if (textured)
      {
        pTextureCoordinate->point.push_back( SFVec2f( uCoord, vCoord ) );
      }
    }
  }
  for (int lat = 0; lat < d - 1; lat++)
  {
    for (int lon = 0; lon < w - 1; lon++)
    {
      int ll =  lat      * w +  lon     ;  // lower left
      int lr =  lat      * w + (lon + 1);  // lower right
      int ur = (lat + 1) * w + (lon + 1);  // upper right
      int ul = (lat + 1) * w +  lon     ;  // upper left

      ll += indexOffset;
      lr += indexOffset;
      ur += indexOffset;
      ul += indexOffset;

      pIndexedFaceSet->coordIndex.push_back( ll );
      pIndexedFaceSet->coordIndex.push_back( lr );
      pIndexedFaceSet->coordIndex.push_back( ur );
      pIndexedFaceSet->coordIndex.push_back( ul );
      pIndexedFaceSet->coordIndex.push_back( -1 );

      pIndexedFaceSet->normalIndex.push_back( ll );
      pIndexedFaceSet->normalIndex.push_back( lr );
      pIndexedFaceSet->normalIndex.push_back( ur );
      pIndexedFaceSet->normalIndex.push_back( ul );
      pIndexedFaceSet->normalIndex.push_back( -1 );

      if (textured)
      {
        pIndexedFaceSet->texCoordIndex.push_back( ll );
        pIndexedFaceSet->texCoordIndex.push_back( lr );
        pIndexedFaceSet->texCoordIndex.push_back( ur );
        pIndexedFaceSet->texCoordIndex.push_back( ul );
        pIndexedFaceSet->texCoordIndex.push_back( -1 );
      }
    }
  }
  indexOffset += d * w;

  // Negative z-axis vertices, normals, texcoords:
  zCoord = -depth * 0.5f;
  for (int lat = 0; lat < h; lat++)
  {
    vCoord = (float) lat / (float) (h - 1); // [0.0, 1.0]
    yCoord = height * (vCoord - 0.5f);      // [-height/2, height/2]

    for (int lon = 0; lon < w; lon++)
    {
      uCoord = (float) lon / (float) (w - 1); // [0.0, 1.0]
      xCoord = -width * (uCoord - 0.5f);      // [-width/2, width/2]

      pCoordinate->point.push_back( SFVec3f( xCoord, yCoord, zCoord ) );
      pNormal->vector.push_back( SFVec3f( 0.0f, 0.0f, -1.0f) );
      if (textured)
      {
        pTextureCoordinate->point.push_back( SFVec2f( uCoord, vCoord ) );
      }
    }
  }
  for (int lat = 0; lat < h - 1; lat++)
  {
    for (int lon = 0; lon < w - 1; lon++)
    {
      int ll =  lat      * w +  lon     ;  // lower left
      int lr =  lat      * w + (lon + 1);  // lower right
      int ur = (lat + 1) * w + (lon + 1);  // upper right
      int ul = (lat + 1) * w +  lon     ;  // upper left

      ll += indexOffset;
      lr += indexOffset;
      ur += indexOffset;
      ul += indexOffset;

      pIndexedFaceSet->coordIndex.push_back( ll );
      pIndexedFaceSet->coordIndex.push_back( lr );
      pIndexedFaceSet->coordIndex.push_back( ur );
      pIndexedFaceSet->coordIndex.push_back( ul );
      pIndexedFaceSet->coordIndex.push_back( -1 );

      pIndexedFaceSet->normalIndex.push_back( ll );
      pIndexedFaceSet->normalIndex.push_back( lr );
      pIndexedFaceSet->normalIndex.push_back( ur );
      pIndexedFaceSet->normalIndex.push_back( ul );
      pIndexedFaceSet->normalIndex.push_back( -1 );

      if (textured)
      {
        pIndexedFaceSet->texCoordIndex.push_back( ll );
        pIndexedFaceSet->texCoordIndex.push_back( lr );
        pIndexedFaceSet->texCoordIndex.push_back( ur );
        pIndexedFaceSet->texCoordIndex.push_back( ul );
        pIndexedFaceSet->texCoordIndex.push_back( -1 );
      }
    }
  }
  // indexOffset += h * w;
}

void WRLLoader::createCone( IndexedFaceSetSharedPtr & pIndexedFaceSet,
                            float radius, float height,
                            bool bottom, bool side, bool textured )
{
  if ( !(bottom || side ) ) // Any geometry to create?
  {
    return;
  }

  // Tessellate with square quads when inside the unclamped range.
  int m = clamp( (int) (m_subdivisions.sphereDefault * radius), m_subdivisions.sphereMin, m_subdivisions.sphereMax );
  int n = clamp( (int) (m_subdivisions.sphereDefault * height / (2.0f * PI)), 2, m_subdivisions.sphereMax );
  int k = clamp( (int) (m_subdivisions.sphereDefault * radius / (2.0f * PI)), 2, m_subdivisions.sphereMax );

  size_t numVertices = 0;
  size_t numIndices  = 0;
  if (bottom)
  {
    numVertices += (k - 1) * m + 1;
    numIndices  += (k - 2) * m * 5 + m * 4;
  }
  if (side)
  {
    numVertices += n * (m + 1);
    numIndices  += (n - 1) * m * 5;
  }

  CoordinateSharedPtr pCoordinate = Coordinate::create();
  pCoordinate->point.reserve( numVertices ); // vertices
  pIndexedFaceSet->coord = pCoordinate;
  pIndexedFaceSet->coordIndex.reserve( numIndices );

  NormalSharedPtr pNormal = Normal::create();
  pNormal->vector.reserve( numVertices ); // normals
  pIndexedFaceSet->normal = pNormal;
  pIndexedFaceSet->normalIndex.reserve( numIndices );
  pIndexedFaceSet->normalPerVertex = true; // Is the default.

  TextureCoordinateSharedPtr pTextureCoordinate = TextureCoordinateSharedPtr::null;
  if ( textured )
  {
    pTextureCoordinate = TextureCoordinate::create();
    pTextureCoordinate->point.reserve( numVertices ); // texcoords
    pIndexedFaceSet->texCoord = pTextureCoordinate;
    pIndexedFaceSet->texCoordIndex.reserve( numIndices );
  }

  int indexOffset = 0; // The next sub-object will generate indices starting at this position.
  float phi_step = 2.0f * PI / (float) m;

  if (bottom)
  {
    float scaleDec = 1.0f / (float) (k - 1);
    float yCoord = -height * 0.5f;
    for (int lat = 0; lat < k - 1; lat++) // Exclude the pole.
    {
      float scale = 1.0f - (float) lat * scaleDec;
      for (int lon = 0; lon < m; lon++)
      {
        // VRML defines the texture coordinates to start at the back of the cone,
        // which means all phi angles need to be offset by pi/2. Top and bottom tesselation must match.
        float phi = (float) lon * phi_step + PI_HALF;
        float sinPhi = sinf(phi);
        float cosPhi = cosf(phi);

        float xCoord =  cosPhi * scale;
        float zCoord = -sinPhi * scale;
        pCoordinate->point.push_back( SFVec3f( xCoord * radius, yCoord, zCoord * radius ) );
        pNormal->vector.push_back( SFVec3f( 0.0f, -1.0f, 0.0) ); // bottom
        if (textured)
        {
          // "The bottom texture appears right side up when the top of the cone is tilted toward the -Z axis"
          float texu =  zCoord * 0.5f + 0.5f; // [-1.0, 1.0] => [0.0, 1.0]
          float texv = -xCoord * 0.5f + 0.5f;
          pTextureCoordinate->point.push_back( SFVec2f( texu, texv ) );
        }
      }
    }

    pCoordinate->point.push_back(SFVec3f(0.0f, yCoord, 0.0f));  // Center point (south pole).
    pNormal->vector.push_back(SFVec3f(0.0f, -1.0f, 0.0f));      // bottom
    if (textured)
    {
      pTextureCoordinate->point.push_back(SFVec2f(0.5f, 0.5f)); // texture center
    }

    for (int lat = 0; lat < k - 2; lat++)
    {
      for (int lon = 0; lon < m; lon++)
      {
        int ll =  lat      * m + lon          ;  // lower left
        int lr =  lat      * m + (lon + 1) % m;  // lower right
        int ur = (lat + 1) * m + (lon + 1) % m;  // upper right
        int ul = (lat + 1) * m + lon          ;  // upper left

        // Bottom disc inverts the winding!
        pIndexedFaceSet->coordIndex.push_back( ll );
        pIndexedFaceSet->coordIndex.push_back( ul );
        pIndexedFaceSet->coordIndex.push_back( ur );
        pIndexedFaceSet->coordIndex.push_back( lr );
        pIndexedFaceSet->coordIndex.push_back( -1 );

        pIndexedFaceSet->normalIndex.push_back( ll );
        pIndexedFaceSet->normalIndex.push_back( ul );
        pIndexedFaceSet->normalIndex.push_back( ur );
        pIndexedFaceSet->normalIndex.push_back( lr );
        pIndexedFaceSet->normalIndex.push_back( -1 );

        if (textured)
        {
          pIndexedFaceSet->texCoordIndex.push_back( ll );
          pIndexedFaceSet->texCoordIndex.push_back( ul );
          pIndexedFaceSet->texCoordIndex.push_back( ur );
          pIndexedFaceSet->texCoordIndex.push_back( lr );
          pIndexedFaceSet->texCoordIndex.push_back( -1 );
        }
      }
    }

    // Close the center.
    for (int lon = 0; lon < m; lon++)
    {
      int ll     = (k - 2) * m + lon;            // lower left
      int lr     = (k - 2) * m + (lon + 1) % m;  // lower right
      int center = (k - 1) * m;                  // center

      // Bottom disc inverts the winding!
      pIndexedFaceSet->coordIndex.push_back( lr );
      pIndexedFaceSet->coordIndex.push_back( ll );
      pIndexedFaceSet->coordIndex.push_back( center );
      pIndexedFaceSet->coordIndex.push_back( -1 );

      pIndexedFaceSet->normalIndex.push_back( lr );
      pIndexedFaceSet->normalIndex.push_back( ll );
      pIndexedFaceSet->normalIndex.push_back( center );
      pIndexedFaceSet->normalIndex.push_back( -1 );

      if (textured)
      {
        pIndexedFaceSet->texCoordIndex.push_back( lr );
        pIndexedFaceSet->texCoordIndex.push_back( ll );
        pIndexedFaceSet->texCoordIndex.push_back( center );
        pIndexedFaceSet->texCoordIndex.push_back( -1 );
      }
    }

    indexOffset = (k - 1) * m + 1; // The next sub-object will generate indices starting at this position.
  } // bottom


  if (side)
  {
    Vec3f sideNormal(height, radius, 0.0f);
    normalize(sideNormal);

    // Latitudinal rings.
    // Starting at the bottom outer ring going upwards.
    for ( int lat = 0; lat < n; lat++ ) // Subdivisions along the height.
    {
      float texv = (float) lat / (float) (n - 1); // Range [0.0f, 1.0f]
      float yCoord = height * (texv - 0.5f);    // Range [-height/2, height/2]
      float scale = (1.0f - texv) * radius;

      // Generate vertices along the latitudinal rings.
      // On each latitude there are m + 1 vertices,
      // the last one and the first one are on identical positions but have different texture coordinates.
      for ( int lon = 0; lon <= m; lon++ ) // phi angle
      {
        // VRML defines the texture coordinates to start at the back of the cone,
        // which means all phi angles need to be offset by pi/2.
        float phi = (float) lon * phi_step + PI_HALF;
        float sinPhi = sinf( phi );
        float cosPhi = cosf( phi );
        float texu = (float) lon / (float) m; // Range [0.0f, 1.0f]

        pCoordinate->point.push_back( SFVec3f( cosPhi * scale, yCoord, -sinPhi * scale ) );
        // Rotate the side's normal around the y-axis.
        pNormal->vector.push_back( SFVec3f( cosPhi * sideNormal[0] + sinPhi * sideNormal[2],
                                            sideNormal[1],
                                           -sinPhi * sideNormal[0] + cosPhi * sideNormal[2] ) );
        if (textured)
        {
          pTextureCoordinate->point.push_back( SFVec2f( texu, texv ) );
        }
      }
    }

    // We have generated m + 1 vertices per lat.
    const int columns = m + 1;

    // Calculate indices. Using Quads for VRML.
    for ( int lat = 0; lat < n - 1; lat++ )
    {
      for ( int lon = 0; lon < m; lon++ )
      {
        SFInt32 ll =  lat      * columns + lon    ;  // lower left
        SFInt32 lr =  lat      * columns + lon + 1;  // lower right
        SFInt32 ur = (lat + 1) * columns + lon + 1;  // upper right
        SFInt32 ul = (lat + 1) * columns + lon    ;  // upper left

        ll += indexOffset;
        lr += indexOffset;
        ur += indexOffset;
        ul += indexOffset;

        pIndexedFaceSet->coordIndex.push_back( ll );
        pIndexedFaceSet->coordIndex.push_back( lr );
        pIndexedFaceSet->coordIndex.push_back( ur );
        pIndexedFaceSet->coordIndex.push_back( ul );
        pIndexedFaceSet->coordIndex.push_back( -1 );

        pIndexedFaceSet->normalIndex.push_back( ll );
        pIndexedFaceSet->normalIndex.push_back( lr );
        pIndexedFaceSet->normalIndex.push_back( ur );
        pIndexedFaceSet->normalIndex.push_back( ul );
        pIndexedFaceSet->normalIndex.push_back( -1 );

        if (textured)
        {
          pIndexedFaceSet->texCoordIndex.push_back( ll );
          pIndexedFaceSet->texCoordIndex.push_back( lr );
          pIndexedFaceSet->texCoordIndex.push_back( ur );
          pIndexedFaceSet->texCoordIndex.push_back( ul );
          pIndexedFaceSet->texCoordIndex.push_back( -1 );
        }
      }
    }
  } // side
}

void WRLLoader::createCylinder( IndexedFaceSetSharedPtr & pIndexedFaceSet,
                                float radius, float height,
                                bool bottom, bool side, bool top, bool textured )
{
  if ( !(bottom || side || top) ) // Any geometry to create?
  {
    return;
  }

  // Tessellate with square quads when inside the unclamped range.
  int m = clamp( (int) (m_subdivisions.sphereDefault * radius), m_subdivisions.sphereMin, m_subdivisions.sphereMax );
  int n = clamp( (int) (m_subdivisions.sphereDefault * height / (2.0f * PI)), 2, m_subdivisions.sphereMax );
  int k = clamp( (int) (m_subdivisions.sphereDefault * radius / (2.0f * PI)), 2, m_subdivisions.sphereMax );

  size_t numVertices = 0;
  size_t numIndices  = 0;
  if (bottom)
  {
    numVertices += (k - 1) * m + 1;
    numIndices  += (k - 2) * m * 5 + m * 4;
  }
  if (side)
  {
    numVertices += n * (m + 1);
    numIndices  += (n - 1) * m * 5;
  }
  if (top)
  {
    numVertices += (k - 1) * m + 1;
    numIndices  += (k - 2) * m * 5 + m * 4;
  }

  CoordinateSharedPtr pCoordinate = Coordinate::create();
  pCoordinate->point.reserve( numVertices ); // vertices
  pIndexedFaceSet->coord = pCoordinate;
  pIndexedFaceSet->coordIndex.reserve( numIndices );

  NormalSharedPtr pNormal = Normal::create();
  pNormal->vector.reserve( numVertices ); // normals
  pIndexedFaceSet->normal = pNormal;
  pIndexedFaceSet->normalIndex.reserve( numIndices );
  pIndexedFaceSet->normalPerVertex = true; // Is the default.

  TextureCoordinateSharedPtr pTextureCoordinate = TextureCoordinateSharedPtr::null;
  if ( textured )
  {
    pTextureCoordinate = TextureCoordinate::create();
    pTextureCoordinate->point.reserve( numVertices ); // texcoords
    pIndexedFaceSet->texCoord = pTextureCoordinate;
    pIndexedFaceSet->texCoordIndex.reserve( numIndices );
  }

  int indexOffset = 0; // The next sub-object will generate indices starting at this position.
  float phi_step = 2.0f * PI / (float) m;

  if (bottom)
  {
    float scaleDec = 1.0f / (float) (k - 1);
    float yCoord = -height * 0.5f;
    for (int lat = 0; lat < k - 1; lat++) // Exclude the pole.
    {
      float scale = 1.0f - (float) lat * scaleDec;
      for (int lon = 0; lon < m; lon++)
      {
        // VRML defines the texture coordinates to start at the back of the cylinder,
        // which means all phi angles need to be offset by pi/2. Top and bottom tesselation must match!
        float phi = (float) lon * phi_step + PI_HALF;
        float sinPhi = sinf(phi);
        float cosPhi = cosf(phi);

        float xCoord =  cosPhi * scale;
        float zCoord = -sinPhi * scale;
        pCoordinate->point.push_back( SFVec3f( xCoord * radius, yCoord, zCoord * radius ) );
        pNormal->vector.push_back( SFVec3f( 0.0f, -1.0f, 0.0) ); // bottom
        if (textured)
        {
          // "The bottom texture appears right side up when the top of the cylinder is tilted toward the -Z axis"
          float texu =  zCoord * 0.5f + 0.5f; // [-1.0, 1.0] => [0.0, 1.0]
          float texv = -xCoord * 0.5f + 0.5f;
          pTextureCoordinate->point.push_back( SFVec2f( texu, texv ) );
        }
      }
    }

    pCoordinate->point.push_back(SFVec3f(0.0f, yCoord, 0.0f));  // Center point (south pole).
    pNormal->vector.push_back(SFVec3f(0.0f, -1.0f, 0.0f));      // bottom
    if (textured)
    {
      pTextureCoordinate->point.push_back(SFVec2f(0.5f, 0.5f)); // texture center
    }

    for (int lat = 0; lat < k - 2; lat++)
    {
      for (int lon = 0; lon < m; lon++)
      {
        int ll =  lat      * m + lon          ;  // lower left
        int lr =  lat      * m + (lon + 1) % m;  // lower right
        int ur = (lat + 1) * m + (lon + 1) % m;  // upper right
        int ul = (lat + 1) * m + lon          ;  // upper left

        // Bottom disc inverts the winding!
        pIndexedFaceSet->coordIndex.push_back( ll );
        pIndexedFaceSet->coordIndex.push_back( ul );
        pIndexedFaceSet->coordIndex.push_back( ur );
        pIndexedFaceSet->coordIndex.push_back( lr );
        pIndexedFaceSet->coordIndex.push_back( -1 );

        pIndexedFaceSet->normalIndex.push_back( ll );
        pIndexedFaceSet->normalIndex.push_back( ul );
        pIndexedFaceSet->normalIndex.push_back( ur );
        pIndexedFaceSet->normalIndex.push_back( lr );
        pIndexedFaceSet->normalIndex.push_back( -1 );

        if (textured)
        {
          pIndexedFaceSet->texCoordIndex.push_back( ll );
          pIndexedFaceSet->texCoordIndex.push_back( ul );
          pIndexedFaceSet->texCoordIndex.push_back( ur );
          pIndexedFaceSet->texCoordIndex.push_back( lr );
          pIndexedFaceSet->texCoordIndex.push_back( -1 );
        }
      }
    }

    // Close the center.
    for (int lon = 0; lon < m; lon++)
    {
      int ll     = (k - 2) * m + lon;            // lower left
      int lr     = (k - 2) * m + (lon + 1) % m;  // lower right
      int center = (k - 1) * m;                  // center

      // Bottom disc inverts the winding!
      pIndexedFaceSet->coordIndex.push_back( lr );
      pIndexedFaceSet->coordIndex.push_back( ll );
      pIndexedFaceSet->coordIndex.push_back( center );
      pIndexedFaceSet->coordIndex.push_back( -1 );

      pIndexedFaceSet->normalIndex.push_back( lr );
      pIndexedFaceSet->normalIndex.push_back( ll );
      pIndexedFaceSet->normalIndex.push_back( center );
      pIndexedFaceSet->normalIndex.push_back( -1 );

      if (textured)
      {
        pIndexedFaceSet->texCoordIndex.push_back( lr );
        pIndexedFaceSet->texCoordIndex.push_back( ll );
        pIndexedFaceSet->texCoordIndex.push_back( center );
        pIndexedFaceSet->texCoordIndex.push_back( -1 );
      }
    }

    indexOffset = (k - 1) * m + 1; // The next sub-object will generate indices starting at this position.
  } // bottom


  if (side)
  {
    // Latitudinal rings.
    // Starting at the bottom outer ring going upwards.
    for ( int lat = 0; lat < n; lat++ ) // Subdivisions along the height.
    {
      float texv = (float) lat / (float) (n - 1); // Range [0.0f, 1.0f]
      float yCoord = height * (texv - 0.5f);    // Range [-height/2, height/2]

      // Generate vertices along the latitudinal rings.
      // On each latitude there are m + 1 vertices,
      // the last one and the first one are on identical positions but have different texture coordinates.
      for ( int lon = 0; lon <= m; lon++ ) // phi angle
      {
        // VRML defines the texture coordinates to start at the back of the cylinder,
        // which means all phi angles need to be offset by pi/2.
        float phi = (float) lon * phi_step + PI_HALF;
        float sinPhi = sinf( phi );
        float cosPhi = cosf( phi );
        float texu = (float) lon / (float) m; // Range [0.0f, 1.0f]

        pCoordinate->point.push_back( SFVec3f( cosPhi * radius, yCoord, -sinPhi * radius ) );
        pNormal->vector.push_back( SFVec3f( cosPhi, 0.0f, -sinPhi ) );
        if (textured)
        {
          pTextureCoordinate->point.push_back( SFVec2f( texu, texv ) );
        }
      }
    }

    // We have generated m + 1 vertices per lat.
    const int columns = m + 1;

    // Calculate indices. Using Quads for VRML.
    for ( int lat = 0; lat < n - 1; lat++ )
    {
      for ( int lon = 0; lon < m; lon++ )
      {
        SFInt32 ll =  lat      * columns + lon    ;  // lower left
        SFInt32 lr =  lat      * columns + lon + 1;  // lower right
        SFInt32 ur = (lat + 1) * columns + lon + 1;  // upper right
        SFInt32 ul = (lat + 1) * columns + lon    ;  // upper left

        ll += indexOffset;
        lr += indexOffset;
        ur += indexOffset;
        ul += indexOffset;

        pIndexedFaceSet->coordIndex.push_back( ll );
        pIndexedFaceSet->coordIndex.push_back( lr );
        pIndexedFaceSet->coordIndex.push_back( ur );
        pIndexedFaceSet->coordIndex.push_back( ul );
        pIndexedFaceSet->coordIndex.push_back( -1 );

        pIndexedFaceSet->normalIndex.push_back( ll );
        pIndexedFaceSet->normalIndex.push_back( lr );
        pIndexedFaceSet->normalIndex.push_back( ur );
        pIndexedFaceSet->normalIndex.push_back( ul );
        pIndexedFaceSet->normalIndex.push_back( -1 );

        if (textured)
        {
          pIndexedFaceSet->texCoordIndex.push_back( ll );
          pIndexedFaceSet->texCoordIndex.push_back( lr );
          pIndexedFaceSet->texCoordIndex.push_back( ur );
          pIndexedFaceSet->texCoordIndex.push_back( ul );
          pIndexedFaceSet->texCoordIndex.push_back( -1 );
        }
      }
    }

    indexOffset += n * (m + 1); // This many vertices have been generated.
  } // side

  if (top)
  {
    float scaleDec = 1.0f / (float) (k - 1);
    float yCoord = height * 0.5f;
    for (int lat = 0; lat < k - 1; lat++) // Exclude the pole.
    {
      // Nicer, more regular shape of the triangles.
      float scale = 1.0f - (float) lat * scaleDec;
      for (int lon = 0; lon < m; lon++)
      {
        // VRML defines the texture coordinates to start at the back of the cylinder,
        // which means all phi angles need to be offset by pi/2. Top and bottom tesselation must match.
        float phi = (float) lon * phi_step + PI_HALF;
        float sinPhi = sinf(phi);
        float cosPhi = cosf(phi);

        float xCoord =  cosPhi * scale; // [-1.0, 1.0]
        float zCoord = -sinPhi * scale;
        pCoordinate->point.push_back( SFVec3f( xCoord * radius, yCoord, zCoord * radius ) );
        pNormal->vector.push_back( SFVec3f( 0.0f, 1.0f, 0.0) ); // top
        if (textured)
        {
          // "The top texture appears right side up when the top of the cylinder is tilted toward the +Z axis"
          float texu = -zCoord * 0.5f + 0.5f; // [-1.0, 1.0] => [0.0, 1.0]
          float texv = -xCoord * 0.5f + 0.5f;
          pTextureCoordinate->point.push_back( SFVec2f( texu, texv ) );
        }
      }
    }

    pCoordinate->point.push_back(SFVec3f(0.0f, yCoord, 0.0f));  // Center point (north pole).
    pNormal->vector.push_back(SFVec3f(0.0f, 1.0f, 0.0f));       // top
    if (textured)
    {
      pTextureCoordinate->point.push_back(SFVec2f(0.5f, 0.5f)); // texture center
    }

    for (int lat = 0; lat < k - 2; lat++)
    {
      for (int lon = 0; lon < m; lon++)
      {
        int ll =  lat      * m + lon          ;  // lower left
        int lr =  lat      * m + (lon + 1) % m;  // lower right
        int ur = (lat + 1) * m + (lon + 1) % m;  // upper right
        int ul = (lat + 1) * m + lon          ;  // upper left

        ll += indexOffset;
        lr += indexOffset;
        ur += indexOffset;
        ul += indexOffset;

        // Top disc uses standard CCW ordering.
        pIndexedFaceSet->coordIndex.push_back( ll );
        pIndexedFaceSet->coordIndex.push_back( lr );
        pIndexedFaceSet->coordIndex.push_back( ur );
        pIndexedFaceSet->coordIndex.push_back( ul );
        pIndexedFaceSet->coordIndex.push_back( -1 );

        pIndexedFaceSet->normalIndex.push_back( ll );
        pIndexedFaceSet->normalIndex.push_back( lr );
        pIndexedFaceSet->normalIndex.push_back( ur );
        pIndexedFaceSet->normalIndex.push_back( ul );
        pIndexedFaceSet->normalIndex.push_back( -1 );

        if (textured)
        {
          pIndexedFaceSet->texCoordIndex.push_back( ll );
          pIndexedFaceSet->texCoordIndex.push_back( lr );
          pIndexedFaceSet->texCoordIndex.push_back( ur );
          pIndexedFaceSet->texCoordIndex.push_back( ul );
          pIndexedFaceSet->texCoordIndex.push_back( -1 );
        }
      }
    }

    // Close the center.
    for (int lon = 0; lon < m; lon++)
    {
      int ll     = (k - 2) * m + lon;            // lower left
      int lr     = (k - 2) * m + (lon + 1) % m;  // lower right
      int center = (k - 1) * m;                  // center

      ll     += indexOffset;
      lr     += indexOffset;
      center += indexOffset;

      pIndexedFaceSet->coordIndex.push_back( ll );
      pIndexedFaceSet->coordIndex.push_back( lr );
      pIndexedFaceSet->coordIndex.push_back( center );
      pIndexedFaceSet->coordIndex.push_back( -1 );

      pIndexedFaceSet->normalIndex.push_back( ll );
      pIndexedFaceSet->normalIndex.push_back( lr );
      pIndexedFaceSet->normalIndex.push_back( center );
      pIndexedFaceSet->normalIndex.push_back( -1 );

      if (textured)
      {
        pIndexedFaceSet->texCoordIndex.push_back( ll );
        pIndexedFaceSet->texCoordIndex.push_back( lr );
        pIndexedFaceSet->texCoordIndex.push_back( center );
        pIndexedFaceSet->texCoordIndex.push_back( -1 );
      }
    }
  } // top
}

void WRLLoader::createSphere( IndexedFaceSetSharedPtr & pIndexedFaceSet, float radius, bool textured )
{
  int m = clamp( (int) (m_subdivisions.sphereDefault * radius), m_subdivisions.sphereMin, m_subdivisions.sphereMax );
  int n = clamp( m >> 1, std::max(3, m_subdivisions.sphereMin >> 1), std::max(3, m_subdivisions.sphereMax >> 1) );

  const size_t numVertices = n * (m + 1);     // Number of vertices.
  const size_t numIndices  = (n - 1) * m * 5; // Number of indices (quad plus -1 end index = 5)

  CoordinateSharedPtr pCoordinate = Coordinate::create();
  pCoordinate->point.reserve( numVertices ); // vertices
  pIndexedFaceSet->coord = pCoordinate;
  pIndexedFaceSet->coordIndex.reserve( numIndices );

  NormalSharedPtr pNormal = Normal::create();
  pNormal->vector.reserve( numVertices ); // normals
  pIndexedFaceSet->normal = pNormal;
  pIndexedFaceSet->normalIndex.reserve( numIndices );
  pIndexedFaceSet->normalPerVertex = true; // Is the default.

  TextureCoordinateSharedPtr pTextureCoordinate;
  if ( textured )
  {
    pTextureCoordinate = TextureCoordinate::create();
    pTextureCoordinate->point.reserve( numVertices ); // texcoords
    pIndexedFaceSet->texCoord = pTextureCoordinate;
    pIndexedFaceSet->texCoordIndex.reserve( numIndices );
  }

  float phi_step = 2.0f * PI / (float) m;
  float theta_step = PI / (float) (n - 1);

  // Latitudinal rings.
  // Starting at the south pole going upwards.
  for ( int latitude = 0; latitude < n; latitude++ ) // theta angle
  {
    float theta = (float) latitude * theta_step;
    float sinTheta = sinf( theta );
    float cosTheta = cosf( theta );
    float texv = (float) latitude / (float) (n - 1); // Range [0.0f, 1.0f]

    // Generate vertices along the latitudinal rings.
    // On each latitude there are m + 1 vertices,
    // the last one and the first one are on identical positions but have different texture coordinates.
    for ( int longitude = 0; longitude <= m; longitude++ ) // phi angle
    {
      // VRML defines the texture coordinates to start at the back of the sphere,
      // which means all phi angles need to be offset by pi/2.
      float phi = (float) longitude * phi_step + PI_HALF;
      float sinPhi = sinf( phi );
      float cosPhi = cosf( phi );
      float texu = (float) longitude / (float) m; // Range [0.0f, 1.0f]

      // Unit sphere coordinates are the normals.
      SFVec3f v = SFVec3f( cosPhi * sinTheta,
                          -cosTheta,             // -y to start at the south pole.
                          -sinPhi * sinTheta );

      pCoordinate->point.push_back( v * radius );
      pNormal->vector.push_back( v );
      if (textured)
      {
        pTextureCoordinate->point.push_back( SFVec2f( texu, texv ) );
      }
    }
  }

  // We have generated m + 1 vertices per latitude.
  const int columns = m + 1;

  // Calculate indices. Using Quads for VRML.
  for ( int latitude = 0; latitude < n - 1; latitude++ )
  {
    for ( int longitude = 0; longitude < m; longitude++ )
    {
      SFInt32 ll =  latitude      * columns + longitude    ;  // lower left
      SFInt32 lr =  latitude      * columns + longitude + 1;  // lower right
      SFInt32 ur = (latitude + 1) * columns + longitude + 1;  // upper right
      SFInt32 ul = (latitude + 1) * columns + longitude    ;  // upper left

      pIndexedFaceSet->coordIndex.push_back( ll );
      pIndexedFaceSet->coordIndex.push_back( lr );
      pIndexedFaceSet->coordIndex.push_back( ur );
      pIndexedFaceSet->coordIndex.push_back( ul );
      pIndexedFaceSet->coordIndex.push_back( -1 );

      pIndexedFaceSet->normalIndex.push_back( ll );
      pIndexedFaceSet->normalIndex.push_back( lr );
      pIndexedFaceSet->normalIndex.push_back( ur );
      pIndexedFaceSet->normalIndex.push_back( ul );
      pIndexedFaceSet->normalIndex.push_back( -1 );

      if (textured)
      {
        pIndexedFaceSet->texCoordIndex.push_back( ll );
        pIndexedFaceSet->texCoordIndex.push_back( lr );
        pIndexedFaceSet->texCoordIndex.push_back( ur );
        pIndexedFaceSet->texCoordIndex.push_back( ul );
        pIndexedFaceSet->texCoordIndex.push_back( -1 );
      }
    }
  }
}

void WRLLoader::determineTexGen( IndexedFaceSetSharedPtr const& pIndexedFaceSet
                               , const ParameterGroupDataSharedPtr & parameterGroupData )
{
  DP_ASSERT( pIndexedFaceSet && parameterGroupData );
  DP_ASSERT( pIndexedFaceSet->coord.isPtrTo<Coordinate>() );

  MFVec3f const& point = pIndexedFaceSet->coord.staticCast<Coordinate>()->point;
  SFVec3f min = point[0];
  SFVec3f max = min;
  for ( size_t i=1 ; i<point.size() ; i++ )
  {
    for ( unsigned int j=0 ; j<3 ; j++ )
    {
      if ( max[j] < point[i][j] )
      {
        max[j] = point[i][j];
      }
      else if ( point[i][j] < min[j] )
      {
        min[j] = point[i][j];
      }
    }
  }
  SFVec3f dim = max - min;
  unsigned int first, second;
  if ( dim[0] < dim[1] )
  {
    if ( dim[1] < dim[2] )
    {
      first = 2;
      second = 1;
    }
    else
    {
      first = 1;
      second = ( dim[0] < dim[2] ) ? 2 : 0;
    }
  }
  else
  {
    if ( dim[0] < dim[2] )
    {
      first = 2;
      second = 0;
    }
    else
    {
      first = 0;
      second = ( dim[1] < dim[2] ) ? 2 : 1;
    }
  }
  Vec4f plane[2];

  Plane3f p0( Vec3f( (first==0)?1.0f:0.0f, (first==1)?1.0f:0.0f,
                     (first==2)?1.0f:0.0f ), min );
  plane[0] = Vec4f( p0.getNormal(), p0.getOffset() );

  Plane3f p1( Vec3f( (second==0)?1.0f:0.0f, (second==1)?1.0f:0.0f,
                     (second==2)?1.0f:0.0f ), min );
  plane[1] = Vec4f( p0.getNormal(), p0.getOffset() );

  const dp::fx::ParameterGroupSpecSharedPtr & pgs = parameterGroupData->getParameterGroupSpec();
  DP_ASSERT( pgs->getName() == "standardTextureParameters" );
  DP_VERIFY( parameterGroupData->setParameterArrayElement<dp::fx::EnumSpec::StorageType>( "genMode", static_cast<unsigned int>(TexCoordAxis::S), static_cast<dp::fx::EnumSpec::StorageType>(TexGenMode::LINEAR) ) );
  DP_VERIFY( parameterGroupData->setParameterArrayElement<dp::fx::EnumSpec::StorageType>( "genMode", static_cast<unsigned int>(TexCoordAxis::T), static_cast<dp::fx::EnumSpec::StorageType>(TexGenMode::LINEAR) ) );
  DP_VERIFY( parameterGroupData->setParameterArrayElement<Vec4f>( "texGenPlane", static_cast<unsigned int>(TexCoordAxis::S), plane[0] ) );
  DP_VERIFY( parameterGroupData->setParameterArrayElement<Vec4f>( "texGenPlane", static_cast<unsigned int>(TexCoordAxis::T), plane[1] ) );
}

template<typename VType>
void  WRLLoader::eraseIndex( unsigned int f, unsigned int i, unsigned int count, bool perVertex
                           , MFInt32 &index, VType &vec )
{
  if ( perVertex )
  {
    if ( index.empty() )
    {
      //  do nothing, the indices are already erased from the vertex index array
    }
    else
    {
      //  remove count indices
      index.erase( index.begin()+i, index.begin()+i+count );
    }
  }
  else
  {
    if ( index.empty() )
    {
      //  remove the entry from the vector itself
      vec.erase( vec.begin()+f, vec.begin()+f+1 );
    }
    else
    {
      //  remove the single entry from the index
      DP_ASSERT( f < index.size() );
      index.erase( index.begin()+f, index.begin()+f+1 );
    }
  }
}

SFNode  WRLLoader::findNode( const SFNode currentNode, string name )
{
  SFNode  node;
  if ( m_defNodes.find( name ) != m_defNodes.end() )
  {
    node = m_defNodes[name];
  }
  else if ( currentNode && ( currentNode->getName() == name ) )
  {
    node = currentNode;
  }
  else
  {
    for ( int i=(int)m_openNodes.size()-1 ; i>0 ; i-- )
    {
      if ( m_openNodes[i]->getName() == name )
      {
        node = m_openNodes[i];
        i = -1;
      }
    }
  }
  return( node );
}

vector<unsigned int> WRLLoader::getCombinedKeys( PositionInterpolatorSharedPtr const& center
                                               , OrientationInterpolatorSharedPtr const& rotation
                                               , PositionInterpolatorSharedPtr const& scale
                                               , PositionInterpolatorSharedPtr const& translation )
{
  vector<unsigned int> steps[4];
  unsigned int n = 0;
  if ( center )
  {
    DP_ASSERT( center->interpreted );
    steps[n++] = center->steps;
  }
  if ( rotation )
  {
    DP_ASSERT( rotation->interpreted );
    steps[n++] = rotation->steps;
  }
  if ( scale )
  {
    DP_ASSERT( scale->interpreted );
    steps[n++] = scale->steps;
  }
  if ( translation )
  {
    DP_ASSERT( translation->interpreted );
    steps[n++] = translation->steps;
  }
  DP_ASSERT( n > 0 );
  vector<unsigned int> combinedSteps = steps[0];

  bool ok = true;
  for ( unsigned int i=1 ; ok && i<n ; i++ )
  {
    ok = ( steps[i-1].size() == steps[i].size() );
    for ( size_t j=0 ; ok && j<steps[i].size() ; j++ )
    {
      ok = ( steps[i-1][j] == steps[i][j] );
    }
  }
  if ( ! ok )
  {
    for ( unsigned int i=1 ; i<n ; i++ )
    {
      vector<unsigned int> tmpSteps;
      merge( combinedSteps.begin(), combinedSteps.end(), steps[i].begin(), steps[i].end()
           , back_inserter( tmpSteps ) );
      combinedSteps.clear();
      unique_copy( tmpSteps.begin(), tmpSteps.end(), back_inserter( combinedSteps ) );
    }
  }

  return( combinedSteps );
}

bool WRLLoader::getNextLine( void )
{
  string::size_type index;
  do
  {
    m_eof = ( fgets( m_line, m_lineLength+1, m_fh ) == NULL );
    while ( !m_eof && ( strlen( m_line ) == m_lineLength ) && ( m_line[m_lineLength-1] != '\n' ) )
    {
      m_line = (char *) realloc( m_line, 2 * m_lineLength + 1 );
      m_eof = ( fgets( &m_line[m_lineLength], m_lineLength+1, m_fh ) == NULL );
      m_lineLength *= 2;
    }
    if ( !m_eof )
    {
      DP_ASSERT( strlen( m_line ) <= m_lineLength );
      m_currentString = m_line;
      index = findNotDelimiter( m_currentString, 0 );   // find_first_not_of is slower!
      m_lineNumber++;
    }
  } while ( !m_eof && ( index == string::npos ) );
  return( !m_eof );
}

string & WRLLoader::getNextToken( void )
{
  if ( ! m_ungetToken.empty() )
  {
    m_currentToken = m_ungetToken;
    m_ungetToken.clear();
  }
  else if ( m_eof )
  {
    m_currentToken.clear();
  }
  else
  {
    DP_ASSERT( m_nextTokenStart < m_nextTokenEnd );
    DP_ASSERT( ( m_nextTokenEnd == string::npos ) || ( m_nextTokenEnd < m_currentString.length() ) );
    m_currentToken.assign( m_currentString, m_nextTokenStart, m_nextTokenEnd-m_nextTokenStart );
    DP_ASSERT( m_currentToken[0] != '#' );
    setNextToken();
  }

  if ( m_currentToken.length() > 1 )
  {
    string::size_type index = findBraces( m_currentToken );   // find_first_of is slower!
    if ( index != string::npos )
    {
      if ( index == 0 )
      {
        m_ungetToken.assign( m_currentToken, 1, string::npos );
        m_currentToken.erase( 1, string::npos );
      }
      else
      {
        m_ungetToken.assign( m_currentToken, index, string::npos );
        m_currentToken.erase( index, string::npos );
      }
    }
  }
  return( m_currentToken );
}

SFNode  WRLLoader::getNode( const string &nodeName, string &token )
{
  SFNode  n;
  if ( token == "Anchor" )
  {
    n = readAnchor( nodeName );
  }
  else if ( token == "Appearance" )
  {
    n = readAppearance( nodeName );
  }
  else if ( token == "AudioClip" )
  {
    n = readAudioClip( nodeName );
  }
  else if ( token == "Background" )
  {
    n = readBackground( nodeName );
  }
  else if ( token == "Billboard" )
  {
    n = readBillboard( nodeName );
  }
  else if ( token == "Box" )
  {
    n = readBox( nodeName );
  }
  else if ( token == "Collision" )
  {
    n = readCollision( nodeName );
  }
  else if ( token == "Color" )
  {
    n = readColor( nodeName );
  }
  else if ( token == "ColorInterpolator" )
  {
    n = readColorInterpolator( nodeName );
  }
  else if ( token == "Cone" )
  {
    n = readCone( nodeName );
  }
  else if ( token == "Coordinate" )
  {
    n = readCoordinate( nodeName );
  }
  else if ( token == "CoordinateInterpolator" )
  {
    n = readCoordinateInterpolator( nodeName );
  }
  else if ( token == "Cylinder" )
  {
    n = readCylinder( nodeName );
  }
  else if ( token == "CylinderSensor" )
  {
    n = readCylinderSensor( nodeName );
  }
  else if ( token == "DirectionalLight" )
  {
    n = readDirectionalLight( nodeName );
  }
  else if ( token == "ElevationGrid" )
  {
    n = readElevationGrid( nodeName );
  }
  else if ( token == "Extrusion" )
  {
    n = readExtrusion( nodeName );
  }
  else if ( token == "Fog" )
  {
    n = readFog( nodeName );
  }
  else if ( token == "FontStyle" )
  {
    n = readFontStyle( nodeName );
  }
  else if ( token == "Group" )
  {
    n = readGroup( nodeName );
  }
  else if ( token == "ImageTexture" )
  {
    n = readImageTexture( nodeName );
  }
  else if ( token == "IndexedFaceSet" )
  {
    n = readIndexedFaceSet( nodeName );
  }
  else if ( token == "IndexedLineSet" )
  {
    n = readIndexedLineSet( nodeName );
  }
  else if ( token == "Inline" )
  {
    n = readInline( nodeName );
  }
  else if ( token == "LOD" )
  {
    n = readLOD( nodeName );
  }
  else if ( token == "Material" )
  {
    n = readMaterial( nodeName );
  }
  else if ( token == "MovieTexture" )
  {
    n = readMovieTexture( nodeName );
  }
  else if ( token == "NavigationInfo" )
  {
    n = readNavigationInfo( nodeName );
  }
  else if ( token == "Normal" )
  {
    n = readNormal( nodeName );
  }
  else if ( token == "NormalInterpolator" )
  {
    n = readNormalInterpolator( nodeName );
  }
  else if ( token == "OrientationInterpolator" )
  {
    n = readOrientationInterpolator( nodeName );
  }
  else if ( token == "PixelTexture" )
  {
    n = readPixelTexture( nodeName );
  }
  else if ( token == "PlaneSensor" )
  {
    n = readPlaneSensor( nodeName );
  }
  else if ( token == "PointLight" )
  {
    n = readPointLight( nodeName );
  }
  else if ( token == "PointSet" )
  {
    n = readPointSet( nodeName );
  }
  else if ( token == "PositionInterpolator" )
  {
    n = readPositionInterpolator( nodeName );
  }
  else if ( token == "ProximitySensor" )
  {
    n = readProximitySensor( nodeName );
  }
  else if ( token == "ScalarInterpolator" )
  {
    n = readScalarInterpolator( nodeName );
  }
  else if ( token == "Script" )
  {
    n = readScript( nodeName );
  }
  else if ( token == "Shape" )
  {
    n = readShape( nodeName );
  }
  else if ( token == "Sound" )
  {
    n = readSound( nodeName );
  }
  else if ( token == "Sphere" )
  {
    n = readSphere( nodeName );
  }
  else if ( token == "SphereSensor" )
  {
    n = readSphereSensor( nodeName );
  }
  else if ( token == "SpotLight" )
  {
    n = readSpotLight( nodeName );
  }
  else if ( token == "Switch" )
  {
    n = readSwitch( nodeName );
  }
  else if ( token == "Text" )
  {
    n = readText( nodeName );
  }
  else if ( token == "TextureCoordinate" )
  {
    n = readTextureCoordinate( nodeName );
  }
  else if ( token == "TextureTransform" )
  {
    n = readTextureTransform( nodeName );
  }
  else if ( token == "TimeSensor" )
  {
    n = readTimeSensor( nodeName );
  }
  else if ( token == "TouchSensor" )
  {
    n = readTouchSensor( nodeName );
  }
  else if ( token == "Transform" )
  {
    n = readTransform( nodeName );
  }
  else if ( token == "Viewpoint" )
  {
    n = readViewpoint( nodeName );
  }
  else if ( token == "VisibilitySensor" )
  {
    n = readVisibilitySensor( nodeName );
  }
  else if ( token == "WorldInfo" )
  {
    n = readWorldInfo( nodeName );
  }
  else if ( m_PROTONames.find( token ) != m_PROTONames.end() )
  {
    onUnsupportedToken( "PROTO", token );
    ignoreBlock( "{", "}", getNextToken() );
  }
  else
  {
    onUnknownToken( "Node Type", token );
  }

  return( n );
}

SceneSharedPtr WRLLoader::import( const string &filename )
{
  m_fh = fopen( filename.c_str(), "r" );
  if ( m_fh )
  {
    if ( testWRLVersion( filename ) )
    {
      m_topLevelGroup = vrml::Group::create();
      readStatements();
      interpretVRMLTree();
      m_topLevelGroup.reset();

      DP_ASSERT( m_scene && m_scene->getRootNode());
      //  clear the defNodes and inlines now
      m_defNodes.clear();
      m_inlines.clear();

      //  clean up the scene a bit: remove empty Triangles, shift ligths,...

      //  WRL files don't have target distances, so calculate them here...
      if ( m_scene->getNumberOfCameras() )
      {
        Sphere3f bs = m_scene->getBoundingSphere();
        if ( isPositive( bs ) )
        {
          for ( Scene::CameraIterator scci = m_scene->beginCameras() ; scci != m_scene->endCameras() ; ++scci )
          {
            DP_ASSERT( scci->isPtrTo<PerspectiveCamera>() );
            PerspectiveCameraSharedPtr const& pc = scci->staticCast<PerspectiveCamera>();
            pc->calcNearFarDistances( bs );
            pc->setFocusDistance( 0.5f * ( pc->getNearDistance() + pc->getFarDistance() ) );
          }
        }
      }
    }
    fclose( m_fh );
    m_fh = NULL;
  }

  return( m_scene );
}

void  WRLLoader::interpretVRMLTree( void )
{
  m_scene = Scene::create();    //  This is may be used while interpreting children
  m_rootNode = dp::sg::core::Group::create();
  {
    interpretChildren( m_topLevelGroup->children, m_rootNode );
  }

  m_scene->setRootNode( m_rootNode );
}

dp::sg::core::PipelineDataSharedPtr WRLLoader::interpretAppearance( AppearanceSharedPtr const& pAppearance )
{
  if ( ! pAppearance->materialPipeline )
  {
    pAppearance->materialPipeline = dp::sg::core::PipelineData::create( getStandardMaterialSpec() );
    pAppearance->materialPipeline->setName( pAppearance->getName() );

    if ( pAppearance->material )
    {
      DP_ASSERT( pAppearance->material.isPtrTo<vrml::Material>() );
      DP_VERIFY( pAppearance->materialPipeline->setParameterGroupData( interpretMaterial( pAppearance->material.staticCast<vrml::Material>() ) ) );
    }

    ParameterGroupDataSharedPtr textureData;
    if ( pAppearance->texture )
    {
      DP_ASSERT( pAppearance->texture.isPtrTo<vrml::Texture>() );
      textureData = interpretTexture( pAppearance->texture.staticCast<vrml::Texture>() );
      if ( textureData )
      {
        if ( pAppearance->textureTransform )
        {
          DP_ASSERT( pAppearance->textureTransform.isPtrTo<TextureTransform>() );
          interpretTextureTransform( pAppearance->textureTransform.staticCast<TextureTransform>(), textureData );
        }
        DP_VERIFY( pAppearance->materialPipeline->setParameterGroupData( textureData ) );
      }
    }

    bool transparent = ( pAppearance->material && ( 0.0f < pAppearance->material.staticCast<vrml::Material>()->transparency ) );
    if ( textureData && ! transparent )
    {
      const dp::fx::ParameterGroupSpecSharedPtr & pgs = textureData->getParameterGroupSpec();
      const SamplerSharedPtr & sampler = textureData->getParameter<SamplerSharedPtr>( pgs->findParameterSpec( "sampler" ) );
      if ( sampler )
      {
        const dp::sg::core::TextureSharedPtr & texture = sampler->getTexture();
        if ( texture && texture.isPtrTo<TextureHost>() )
        {
          Image::PixelFormat ipf = texture.staticCast<TextureHost>()->getFormat();
          transparent =   ( ipf == Image::PixelFormat::RGBA )
                      ||  ( ipf == Image::PixelFormat::BGRA )
                      ||  ( ipf == Image::PixelFormat::LUMINANCE_ALPHA );
        }
      }
    }
    pAppearance->materialPipeline->setTransparent( transparent );
  }

  return( pAppearance->materialPipeline );
}

void  WRLLoader::interpretBackground( BackgroundSharedPtr const& pBackground )
{
  //  just set the background color
  m_scene->setBackColor( Vec4f(interpretSFColor( pBackground->skyColor[0] ) ,1.0f));
}

dp::sg::core::BillboardSharedPtr WRLLoader::interpretBillboard( vrml::BillboardSharedPtr const& pVRMLBillboard )
{
  dp::sg::core::BillboardSharedPtr pBillboard;

  if ( pVRMLBillboard->pBillboard )
  {
    pBillboard = pVRMLBillboard->pBillboard;
  }
  else
  {
    pBillboard = dp::sg::core::Billboard::create();

    pBillboard->setName( pVRMLBillboard->getName() );
    if ( length( pVRMLBillboard->axisOfRotation ) < FLT_EPSILON )
    {
      pBillboard->setAlignment( dp::sg::core::Billboard::Alignment::VIEWER );
    }
    else
    {
      pBillboard->setAlignment( dp::sg::core::Billboard::Alignment::AXIS );
      pVRMLBillboard->axisOfRotation.normalize();
      pBillboard->setRotationAxis( pVRMLBillboard->axisOfRotation );
    }
    interpretChildren( pVRMLBillboard->children, pBillboard );
    pVRMLBillboard->pBillboard = pBillboard;
  }

  return( pBillboard );
}

inline bool evalTextured( const dp::sg::core::PrimitiveSharedPtr & pset, bool textured)
{
  if ( pset )
  {
    bool hasTexCoords = false;
    for ( unsigned int i=static_cast<unsigned int>(VertexAttributeSet::AttributeID::TEXCOORD0)
        ; !hasTexCoords && i<static_cast<unsigned int>(VertexAttributeSet::AttributeID::VERTEX_ATTRIB_COUNT)
        ; ++i )
    {
      hasTexCoords = !!pset->getVertexAttributeSet()->getNumberOfVertexData(static_cast<VertexAttributeSet::AttributeID>(i));
    }
    return hasTexCoords!=textured;
  }
  return false;
}

void  WRLLoader::interpretBox( BoxSharedPtr const& pBox, vector<PrimitiveSharedPtr> &primitives, bool textured )
{
  if (  evalTextured(pBox->pTriangles, textured)
     || evalTextured(pBox->pQuads, textured) )
  {
    pBox->pTriangles.reset();
    pBox->pQuads.reset();
  }

  if ( pBox->pTriangles || pBox->pQuads )
  {
    if ( pBox->pTriangles )
    {
      primitives.push_back( pBox->pTriangles );
    }
    if ( pBox->pQuads )
    {
      primitives.push_back( pBox->pQuads );
    }
  }
  else
  {
    IndexedFaceSetSharedPtr pIndexedFaceSet = IndexedFaceSet::create();
    pIndexedFaceSet->setName( pBox->getName() );

    createBox( pIndexedFaceSet, pBox->size, textured );

    interpretIndexedFaceSet( pIndexedFaceSet, primitives );
    if ( pIndexedFaceSet->pTriangles )
    {
      pBox->pTriangles = pIndexedFaceSet->pTriangles;
    }
    if ( pIndexedFaceSet->pQuads )
    {
      pBox->pQuads = pIndexedFaceSet->pQuads;
    }
  }
}

void  WRLLoader::interpretCone( ConeSharedPtr const& pCone, vector<PrimitiveSharedPtr> &primitives, bool textured )
{
  if (  evalTextured(pCone->pTriangles, textured)
     || evalTextured(pCone->pQuads, textured) )
  {
    pCone->pTriangles.reset();
    pCone->pQuads.reset();
  }

  if ( pCone->pTriangles || pCone->pQuads )
  {
    if ( pCone->pTriangles )
    {
      primitives.push_back( pCone->pTriangles );
    }
    if ( pCone->pQuads )
    {
      primitives.push_back( pCone->pQuads );
    }
  }
  else
  {
    IndexedFaceSetSharedPtr pIndexedFaceSet = IndexedFaceSet::create();
    pIndexedFaceSet->setName( pCone->getName() );

    createCone( pIndexedFaceSet, pCone->bottomRadius, pCone->height,
                pCone->bottom, pCone->side, textured );

    interpretIndexedFaceSet( pIndexedFaceSet, primitives );
    if ( pIndexedFaceSet->pTriangles )
    {
      pCone->pTriangles = pIndexedFaceSet->pTriangles;
    }
    if ( pIndexedFaceSet->pQuads )
    {
      pCone->pQuads = pIndexedFaceSet->pQuads;
    }
  }
}

void  WRLLoader::interpretCylinder( CylinderSharedPtr const& pCylinder, vector<PrimitiveSharedPtr> &primitives, bool textured )
{
  if (  evalTextured(pCylinder->pTriangles, textured)
     || evalTextured(pCylinder->pQuads, textured) )
  {
    pCylinder->pTriangles.reset();
    pCylinder->pQuads.reset();
  }

  if ( pCylinder->pTriangles || pCylinder->pQuads )
  {
    if ( pCylinder->pTriangles )
    {
      primitives.push_back( pCylinder->pTriangles );
    }
    if ( pCylinder->pQuads )
    {
      primitives.push_back( pCylinder->pQuads );
    }
  }
  else
  {
    IndexedFaceSetSharedPtr pIndexedFaceSet = IndexedFaceSet::create();
    pIndexedFaceSet->setName( pCylinder->getName() );

    createCylinder( pIndexedFaceSet, pCylinder->radius, pCylinder->height,
                    pCylinder->bottom, pCylinder->side, pCylinder->top, textured );

    interpretIndexedFaceSet( pIndexedFaceSet, primitives );
    if ( pIndexedFaceSet->pTriangles )
    {
      pCylinder->pTriangles = pIndexedFaceSet->pTriangles;
    }
    if ( pIndexedFaceSet->pQuads )
    {
      pCylinder->pQuads = pIndexedFaceSet->pQuads;
    }
  }
}

void  WRLLoader::interpretSphere( SphereSharedPtr const& pSphere, vector<PrimitiveSharedPtr> &primitives, bool textured )
{
  if (  evalTextured(pSphere->pTriangles, textured)
     || evalTextured(pSphere->pQuads, textured) )
  {
    pSphere->pTriangles.reset();
    pSphere->pQuads.reset();
  }

  if ( pSphere->pTriangles || pSphere->pQuads )
  {
    if ( pSphere->pTriangles )
    {
      primitives.push_back( pSphere->pTriangles );
    }
    if ( pSphere->pQuads )
    {
      primitives.push_back( pSphere->pQuads );
    }
  }
  else
  {
    IndexedFaceSetSharedPtr pIndexedFaceSet = IndexedFaceSet::create();
    pIndexedFaceSet->setName( pSphere->getName() );

    createSphere( pIndexedFaceSet, pSphere->radius, textured );

    interpretIndexedFaceSet( pIndexedFaceSet, primitives );
    if ( pIndexedFaceSet->pTriangles )
    {
      pSphere->pTriangles = pIndexedFaceSet->pTriangles;
    }
    if ( pIndexedFaceSet->pQuads )
    {
      pSphere->pQuads = pIndexedFaceSet->pQuads;
    }
  }
}

void  WRLLoader::interpretChildren( MFNode &children, dp::sg::core::GroupSharedPtr const& pGroup )
{
  for ( size_t i=0 ; i<children.size() ; i++ )
  {
    dp::sg::core::ObjectSharedPtr pObject = interpretSFNode( children[i] );
    if ( pObject )
    {
      DP_ASSERT( pObject.isPtrTo<Node>() );
      pGroup->addChild( pObject.staticCast<Node>() );
      // LightReferences will be added to the root with the WRLLoadTraverser !
    }
  }
}

void WRLLoader::interpretColor( ColorSharedPtr const& pColor )
{
  if ( ! pColor->interpreted )
  {
    if ( pColor->set_color )
    {
      interpretColorInterpolator( pColor->set_color, dp::checked_cast<unsigned int>(pColor->color.size()) );
    }
    pColor->interpreted = true;
  }
}

void WRLLoader::interpretColorInterpolator( ColorInterpolatorSharedPtr const& pColorInterpolator
                                          , unsigned int colorCount )
{
  if ( ! pColorInterpolator->interpreted )
  {
    SFTime cycleInterval = pColorInterpolator->set_fraction ? pColorInterpolator->set_fraction->cycleInterval : 1.0;
    resampleKeyValues( pColorInterpolator->key, pColorInterpolator->keyValue, colorCount
                     , pColorInterpolator->steps, cycleInterval );
    for ( size_t i=0 ; i<pColorInterpolator->keyValue.size() ; i++ )
    {
      clamp( pColorInterpolator->keyValue[i][0], 0.0f, 1.0f );
      clamp( pColorInterpolator->keyValue[i][1], 0.0f, 1.0f );
      clamp( pColorInterpolator->keyValue[i][2], 0.0f, 1.0f );
    }
    pColorInterpolator->interpreted = true;
  }
}

void WRLLoader::interpretCoordinate( CoordinateSharedPtr const& pCoordinate )
{
  if ( !pCoordinate->interpreted )
  {
    if ( pCoordinate->set_point )
    {
      interpretCoordinateInterpolator( pCoordinate->set_point
                                     , dp::checked_cast<unsigned int>(pCoordinate->point.size()) );
    }
    pCoordinate->interpreted = true;
  }
}

void WRLLoader::interpretCoordinateInterpolator( CoordinateInterpolatorSharedPtr const& pCoordinateInterpolator
                                               , unsigned int pointCount )
{
  if ( ! pCoordinateInterpolator->interpreted )
  {
    SFTime cycleInterval = pCoordinateInterpolator->set_fraction ? pCoordinateInterpolator->set_fraction->cycleInterval : 1.0;
    resampleKeyValues( pCoordinateInterpolator->key, pCoordinateInterpolator->keyValue
                     , pointCount, pCoordinateInterpolator->steps, cycleInterval );
    pCoordinateInterpolator->interpreted = true;
  }
}

LightSourceSharedPtr WRLLoader::interpretDirectionalLight( DirectionalLightSharedPtr const& directionalLight )
{
  LightSourceSharedPtr lightSource;
  if ( directionalLight->lightSource )
  {
    lightSource = directionalLight->lightSource;
  }
  else
  {
    directionalLight->direction.normalize();

    Vec3f color( interpretSFColor( directionalLight->color ) );
    lightSource = createStandardDirectedLight( directionalLight->direction
                                             , directionalLight->ambientIntensity * color
                                             , directionalLight->intensity * color
                                             , directionalLight->intensity * color );
    lightSource->setName( directionalLight->getName() );
    lightSource->setEnabled( directionalLight->on );
  }
  return( lightSource );
}

void  WRLLoader::interpretElevationGrid( ElevationGridSharedPtr const& pElevationGrid
                                       , vector<PrimitiveSharedPtr> &primitives )
{
  if ( pElevationGrid->pTriangles || pElevationGrid->pQuads )
  {
    if ( pElevationGrid->pTriangles )
    {
      primitives.push_back( pElevationGrid->pTriangles );
    }
    if ( pElevationGrid->pQuads )
    {
      primitives.push_back( pElevationGrid->pQuads );
    }
  }
  else
  {
    IndexedFaceSetSharedPtr pIndexedFaceSet = IndexedFaceSet::create();
    pIndexedFaceSet->setName( pElevationGrid->getName() );

    CoordinateSharedPtr pCoordinate = Coordinate::create();
    pCoordinate->point.reserve( pElevationGrid->height.size() );
    for ( int j=0 ; j<pElevationGrid->zDimension ; j++ )
    {
      for ( int i=0 ; i<pElevationGrid->xDimension ; i++ )
      {
        pCoordinate->point.push_back( Vec3f( pElevationGrid->xSpacing * i
                                           , pElevationGrid->height[i+j*pElevationGrid->xDimension]
                                           , pElevationGrid->zSpacing * j ) );
      }
    }
    pIndexedFaceSet->coord = pCoordinate;
    pIndexedFaceSet->coordIndex.reserve( 6 * ( pElevationGrid->xDimension - 1 ) * ( pElevationGrid->zDimension - 1 ) );
    vector<int> faceIndex;
    faceIndex.reserve( 2 * ( pElevationGrid->xDimension - 1 ) * ( pElevationGrid->zDimension - 1 ) );
    for ( int j=0 ; j<pElevationGrid->zDimension-1 ; j++ )
    {
      for ( int i=0 ; i<pElevationGrid->xDimension-1 ; i++ )
      {
        pIndexedFaceSet->coordIndex.push_back(  j    * pElevationGrid->xDimension + i     );
        pIndexedFaceSet->coordIndex.push_back( (j+1) * pElevationGrid->xDimension + i     );
        pIndexedFaceSet->coordIndex.push_back( (j+1) * pElevationGrid->xDimension + i + 1 );
        pIndexedFaceSet->coordIndex.push_back(  j    * pElevationGrid->xDimension + i + 1 );
        pIndexedFaceSet->coordIndex.push_back( -1 );
        faceIndex.push_back( j * pElevationGrid->xDimension + i );
      }
    }

    if ( pElevationGrid->texCoord )
    {
      pIndexedFaceSet->texCoord = pElevationGrid->texCoord;
    }
    else
    {
      TextureCoordinateSharedPtr pTextureCoordinate = TextureCoordinate::create();
      pTextureCoordinate->point.reserve( pElevationGrid->height.size() );
      float xStep = 1.0f / pElevationGrid->xDimension;
      float zStep = 1.0f / pElevationGrid->zDimension;
      for ( int j=0 ; j<pElevationGrid->zDimension ; j++ )
      {
        for ( int i=0 ; i<pElevationGrid->xDimension ; i++ )
        {
          pTextureCoordinate->point.push_back( Vec2f( i * xStep, j * zStep ) );
        }
      }
      pIndexedFaceSet->texCoord = pTextureCoordinate;
    }

    if ( pElevationGrid->color )
    {
      pIndexedFaceSet->color = pElevationGrid->color;
      pIndexedFaceSet->colorPerVertex = pElevationGrid->colorPerVertex;
      if ( ! pIndexedFaceSet->colorPerVertex )
      {
        pIndexedFaceSet->colorIndex = faceIndex;
      }
    }
    if ( pElevationGrid->normal )
    {
      pIndexedFaceSet->normal = pElevationGrid->normal;
      pIndexedFaceSet->normalPerVertex = pElevationGrid->normalPerVertex;
      if ( ! pIndexedFaceSet->normalPerVertex )
      {
        pIndexedFaceSet->normalIndex = faceIndex;
      }
    }
    pIndexedFaceSet->ccw = pElevationGrid->ccw;
    pIndexedFaceSet->creaseAngle = pElevationGrid->creaseAngle;
    pIndexedFaceSet->solid = pElevationGrid->solid;

    interpretIndexedFaceSet( pIndexedFaceSet, primitives );
    if ( pIndexedFaceSet->pTriangles )
    {
      pElevationGrid->pTriangles = pIndexedFaceSet->pTriangles;
    }
    if ( pIndexedFaceSet->pQuads )
    {
      pElevationGrid->pQuads     = pIndexedFaceSet->pQuads;
    }
  }
}

void  WRLLoader::interpretGeometry( vrml::GeometrySharedPtr const& pGeometry, vector<PrimitiveSharedPtr> &primitives
                                  , bool textured )
{
  if ( pGeometry.isPtrTo<Box>() )
  {
    interpretBox( pGeometry.staticCast<Box>(), primitives, textured );
  }
  else if ( pGeometry.isPtrTo<Cone>() )
  {
    interpretCone( pGeometry.staticCast<Cone>(), primitives, textured );
  }
  else if ( pGeometry.isPtrTo<Cylinder>() )
  {
    interpretCylinder( pGeometry.staticCast<Cylinder>(), primitives, textured );
  }
  else if ( pGeometry.isPtrTo<ElevationGrid>() )
  {
    interpretElevationGrid( pGeometry.staticCast<ElevationGrid>(), primitives );
  }
  else if ( pGeometry.isPtrTo<IndexedFaceSet>() )
  {
    interpretIndexedFaceSet( pGeometry.staticCast<IndexedFaceSet>(), primitives );
  }
  else if ( pGeometry.isPtrTo<IndexedLineSet>() )
  {
    interpretIndexedLineSet( pGeometry.staticCast<IndexedLineSet>(), primitives );
  }
  else if ( pGeometry.isPtrTo<PointSet>() )
  {
    interpretPointSet( pGeometry.staticCast<PointSet>(), primitives );
  }
  else
  {
    DP_ASSERT( pGeometry.isPtrTo<Sphere>() );
    interpretSphere( pGeometry.staticCast<Sphere>(), primitives, textured );
  }
}

dp::sg::core::GroupSharedPtr WRLLoader::interpretGroup( vrml::GroupSharedPtr const& pVRMLGroup )
{
  dp::sg::core::GroupSharedPtr pGroup;
  if ( pVRMLGroup->pGroup )
  {
    pGroup = pVRMLGroup->pGroup;
  }
  else
  {
    pGroup = dp::sg::core::Group::create();
    pGroup->setName( pGroup->getName() );
    interpretChildren( pVRMLGroup->children, pGroup );
    pVRMLGroup->pGroup = pGroup;
  }

  return( pGroup );
}

ParameterGroupDataSharedPtr WRLLoader::interpretImageTexture( ImageTextureSharedPtr const& pImageTexture )
{
  if ( !pImageTexture->textureData )
  {
    string  fileName;
    if ( interpretURL( pImageTexture->url, fileName ) )
    {
      map<string,TextureHostWeakPtr>::const_iterator it = m_textureFiles.find( fileName );
      TextureHostSharedPtr texImg;
      if ( it == m_textureFiles.end() )
      {
        texImg = dp::sg::io::loadTextureHost(fileName, m_fileFinder);
        DP_ASSERT( texImg );
        texImg->setTextureTarget(TextureTarget::TEXTURE_2D); // TEXTURE_2D is the only target known by VRML
        m_textureFiles[fileName] = texImg.getWeakPtr();
      }
      else
      {
        texImg = it->second.getSharedPtr().staticCast<TextureHost>();
      }

      SamplerSharedPtr sampler = Sampler::create( texImg );
      sampler->setWrapMode( TexWrapCoordAxis::S, pImageTexture->repeatS ? TextureWrapMode::REPEAT : TextureWrapMode::CLAMP_TO_EDGE );
      sampler->setWrapMode( TexWrapCoordAxis::T, pImageTexture->repeatT ? TextureWrapMode::REPEAT : TextureWrapMode::CLAMP_TO_EDGE );

      pImageTexture->textureData = createStandardTextureParameterData( sampler );
      pImageTexture->textureData->setName( pImageTexture->getName() );
    }
  }
  return( pImageTexture->textureData );
}

void  analyzeIndex( const MFInt32 & mfInt32, vector<unsigned int> & triVerts, vector<unsigned int> & triFaces
                  , vector<unsigned int> & quadVerts, vector<unsigned int> & quadFaces
                  , vector<unsigned int> & polygonVerts, vector<unsigned int> & polygonFaces )
{
  triVerts.clear();
  triFaces.clear();
  quadVerts.clear();
  quadFaces.clear();

  unsigned int faceIndex = 0;
  unsigned int endIndex  = 0;
  do
  {
    unsigned int startIndex = endIndex;
    for ( ; endIndex<mfInt32.size() && mfInt32[endIndex]!=-1 ; endIndex++ )
      ;
    endIndex++;
    switch( endIndex - startIndex )
    {
      case 0 :
      case 1 :
      case 2 :
      case 3 :
        DP_ASSERT( false );
        break;
      case 4 :
        triVerts.push_back( startIndex );
        triFaces.push_back( faceIndex );
        break;
      case 5 :
        quadVerts.push_back( startIndex );
        quadFaces.push_back( faceIndex );
        break;
      default :
        polygonVerts.push_back( startIndex );
        polygonFaces.push_back( faceIndex );
        break;
    }
    faceIndex++;
  } while ( endIndex < mfInt32.size() );
}

void  WRLLoader::interpretIndexedFaceSet( IndexedFaceSetSharedPtr const& pIndexedFaceSet
                                        , vector<PrimitiveSharedPtr> &primitives )
{
  if ( pIndexedFaceSet->pTriangles || pIndexedFaceSet->pQuads || pIndexedFaceSet->pPolygons )
  {
    if ( pIndexedFaceSet->pTriangles )
    {
      primitives.push_back( pIndexedFaceSet->pTriangles );
    }
    if ( pIndexedFaceSet->pQuads )
    {
      primitives.push_back( pIndexedFaceSet->pQuads );
    }
    if ( pIndexedFaceSet->pPolygons )
    {
      primitives.push_back( pIndexedFaceSet->pPolygons );
    }
  }
  else if ( pIndexedFaceSet->coordIndex.size() )
  {
    if ( pIndexedFaceSet->color )
    {
      DP_ASSERT( pIndexedFaceSet->color.isPtrTo<Color>() );
      interpretColor( pIndexedFaceSet->color.staticCast<Color>() );
    }
    if ( pIndexedFaceSet->coord )
    {
      DP_ASSERT( pIndexedFaceSet->coord.isPtrTo<Coordinate>() );
      interpretCoordinate( pIndexedFaceSet->coord.staticCast<Coordinate>() );
    }
    if ( pIndexedFaceSet->normal )
    {
      DP_ASSERT( pIndexedFaceSet->normal.isPtrTo<Normal>() );
      interpretNormal( pIndexedFaceSet->normal.staticCast<Normal>() );
    }
    // no need to interpret texCoord

    //  determine the triangles, quads and polygon faces
    vector<unsigned int> triVerts, triFaces, quadVerts, quadFaces, polygonVerts, polygonFaces;
    analyzeIndex( pIndexedFaceSet->coordIndex, triVerts, triFaces, quadVerts, quadFaces, polygonVerts, polygonFaces );

    dp::sg::core::GroupSharedPtr smoothGroup;
    if ( !pIndexedFaceSet->normal )
    {
      smoothGroup = dp::sg::core::Group::create();
    }
    if ( triFaces.size() )
    {
      VertexAttributeSetSharedPtr vas = interpretVertexAttributeSet( pIndexedFaceSet
                                                                   , dp::checked_cast<unsigned int>(3*triFaces.size())
                                                                   , triVerts, triFaces );

      //  create the Triangles
      PrimitiveSharedPtr pTriangles = Primitive::create( PrimitiveType::TRIANGLES );
      pTriangles->setName( pIndexedFaceSet->getName() );
      pTriangles->setVertexAttributeSet( vas );
      if ( !pIndexedFaceSet->normal )
      {
        pTriangles->generateNormals();
        GeoNodeSharedPtr smoothGeoNode = GeoNode::create();
        smoothGeoNode->setPrimitive( pTriangles );
        smoothGroup->addChild( smoothGeoNode );
      }

      primitives.push_back( pTriangles );
      pIndexedFaceSet->pTriangles = pTriangles;
    }

    if ( quadFaces.size() )
    {
      VertexAttributeSetSharedPtr vas = interpretVertexAttributeSet( pIndexedFaceSet
                                                                   , dp::checked_cast<unsigned int>(4*quadFaces.size())
                                                                   , quadVerts, quadFaces );

      //  create the Quads
      PrimitiveSharedPtr pQuads = Primitive::create( PrimitiveType::QUADS );
      pQuads->setName( pIndexedFaceSet->getName() );
      pQuads->setVertexAttributeSet( vas );
      if ( !pIndexedFaceSet->normal )
      {
        pQuads->generateNormals();
        GeoNodeSharedPtr smoothGeoNode = GeoNode::create();
        smoothGeoNode->setPrimitive( pQuads );
        smoothGroup->addChild( smoothGeoNode );
      }

      primitives.push_back( pQuads );
      pIndexedFaceSet->pQuads = pQuads;
    }

    if ( polygonVerts.size() )
    {
      // create the IndexSet
      vector<unsigned int> indices;
      unsigned int  numberOfVertices = 0;
      for ( size_t i=0 ; i<polygonVerts.size() ; i++ )
      {
        for ( size_t j=0 ; pIndexedFaceSet->coordIndex[polygonVerts[i]+j] != -1 ; j++ )
        {
          indices.push_back( numberOfVertices++ );
        }
        indices.push_back( ~0 );
      }
      indices.pop_back();
      IndexSetSharedPtr is = IndexSet::create();
      is->setData( &indices[0], dp::checked_cast<unsigned int>( indices.size() ) );

      // create the VertexAttributeSet
      VertexAttributeSetSharedPtr vas = interpretVertexAttributeSet( pIndexedFaceSet, numberOfVertices
                                                                   , polygonVerts, polygonFaces );

      // create the polygons Primitive
      PrimitiveSharedPtr pPolygons = Primitive::create( PrimitiveType::POLYGON );
      pPolygons->setName( pIndexedFaceSet->getName() );
      pPolygons->setIndexSet( is );
      pPolygons->setVertexAttributeSet( vas );
      if ( !pIndexedFaceSet->normal )
      {
        pPolygons->generateNormals();
        // don't smooth the normals of a polygon!
      }

      primitives.push_back( pPolygons );
      pIndexedFaceSet->pPolygons = pPolygons;
    }

    if ( !pIndexedFaceSet->normal && ( smoothGroup->getNumberOfChildren() != 0 ) )
    {
      m_smoothTraverser->setCreaseAngle( pIndexedFaceSet->creaseAngle );
      m_smoothTraverser->apply( NodeSharedPtr( smoothGroup ) );
    }
  }
}

void gatherVec3fPerFace( vector<Vec3f> &vTo, const MFVec3f &vFrom, const MFInt32 & index
                       , unsigned int numberOfVertices, unsigned int fromOff, const vector<unsigned int> & startIndices
                       , const vector<unsigned int> & faceIndices )
{
  DP_ASSERT( false );   // never encountered this path
  DP_ASSERT( startIndices.size() == faceIndices.size() );
  vTo.resize( numberOfVertices );
  for ( size_t i=0, idx=0 ; i<startIndices.size() ; i++ )
  {
    Vec3f v = vFrom[fromOff+faceIndices[i]];
    for ( size_t j=0 ; index[startIndices[i]+j] != -1 ; j++ )
    {
      DP_ASSERT( idx < numberOfVertices );
      vTo[idx++] = v;
    }
  }
}

void  gatherVec3fPerFaceIndexed( vector<Vec3f> & vTo, const MFVec3f & vFrom, const MFInt32 & vertexIndex
                               , const MFInt32 & faceIndex, unsigned int numberOfVertices, unsigned int fromOff
                               , const vector<unsigned int> & startIndices, const vector<unsigned int> & faceIndices )
{
  DP_ASSERT( startIndices.size() == faceIndices.size() );
  vTo.resize( numberOfVertices );
  for ( size_t i=0, idx=0 ; i<startIndices.size() ; i++ )
  {
    Vec3f v = vFrom[fromOff+faceIndex[faceIndices[i]]];
    for ( size_t j=0 ; vertexIndex[startIndices[i]+j] != -1 ; j++ )
    {
      DP_ASSERT( idx < numberOfVertices );
      vTo[idx++] = v;
    }
  }
}

template<typename T>
void gatherPerVertex( vector<T> &vTo, const vector<T> & vFrom, const MFInt32 & index
                    , unsigned int numberOfVertices, unsigned int fromOff
                    , const vector<unsigned int> & startIndices, bool ccw )
{
  vTo.resize( numberOfVertices );
  for ( size_t i=0, idx=0 ; i<startIndices.size() ; i++ )
  {
    if ( ccw )
    {
      for ( size_t j=0 ; index[startIndices[i]+j] != -1 ; j++ )
      {
        DP_ASSERT( idx < numberOfVertices );
        vTo[idx++] = vFrom[fromOff+index[startIndices[i]+j]];
      }
    }
    else
    {
      size_t lastIdx = 0;
      for ( ; index[startIndices[i]+lastIdx] != -1 ; lastIdx++ )
        ;
      DP_ASSERT( index[startIndices[i]+lastIdx] == -1 );
      for ( size_t j=lastIdx-1 ; j<=lastIdx ; j-- )   // use wrap-around from 0 "down" to size_t max to end this loop
      {
        DP_ASSERT( idx < numberOfVertices );
        vTo[idx++] = vFrom[fromOff+index[startIndices[i]+j]];
      }
    }
  }
}

template<typename T>
void WRLLoader::resampleKeyValues( MFFloat & keys, vector<T> & values, unsigned int valuesPerKey
                                 , vector<unsigned int> & steps, SFTime cycleInterval )
{
  DP_ASSERT( ! keys.empty() );
  //          step                used keys
  vector<pair<unsigned int,pair<unsigned int,unsigned int> > > stepMap;

  float stepSize = (float)( keys.back() / ( m_stepsPerUnit * cycleInterval ) );
  float halfStepSize = 0.5f * stepSize;

  //  the first key is used as the first step, no matter when it is...
  stepMap.push_back( make_pair(0,make_pair(0,0)) );
  unsigned int startIndex = 0;
  if ( keys[0] < halfStepSize )
  {
    //  first key starts at 0 -> skip any key in the first halfStepSize interval
    while ( startIndex < keys.size() && keys[startIndex] < halfStepSize )
    {
      ++startIndex;
    }
  }

  while ( startIndex < keys.size() )
  {
    // start a new key step
    unsigned int step = static_cast<unsigned int>(( keys[startIndex] + halfStepSize ) / stepSize);
    DP_ASSERT( stepMap.back().first < step );
    float stepPos = step * stepSize;

    if ( abs( stepPos - keys[startIndex] ) < FLT_EPSILON )
    {
      // key coincides with step position -> use just that key
      stepMap.push_back( make_pair(step,make_pair(startIndex,0)) );
    }
    else if ( stepPos < keys[startIndex] )
    {
      // key is to the right of the step position -> use previous and this key
      stepMap.push_back( make_pair(step,make_pair(startIndex-1,startIndex)) );
    }
    else
    {
      // key is to the left of the step position -> scan for the first key to the right
      while ( ( startIndex+1 < keys.size() ) && ( keys[startIndex+1] < stepPos - FLT_EPSILON ) )
      {
        startIndex++;
      }
      if ( startIndex + 1 < keys.size() )
      {
        if ( abs( stepPos - keys[startIndex+1] ) < FLT_EPSILON )
        {
          // next key coincides with step position -> use just that key
          stepMap.push_back( make_pair(step,make_pair(startIndex+1,0)) );
        }
        else
        {
          // -> use startIndex and next
          stepMap.push_back( make_pair(step,make_pair(startIndex,startIndex+1)) );
        }
      }
      else
      {
        // startIndex is the last key (and to the left of step position), so just use it
        stepMap.push_back( make_pair(step,make_pair(startIndex,0)) );
      }
    }
    // skip keys in that same step
    while ( ( startIndex < keys.size() ) && ( keys[startIndex] <= stepPos + halfStepSize ) )
    {
      startIndex++;
    }
  }

  //  now, size of stepMap gives the number of needed keys, with stepMap[i].first being the step
  //  index and stepMap[i].second.first/second the two value to interpolate (if second.second is
  //  zero, just use second.first)
  MFFloat keysOut;
  keysOut.reserve( stepMap.size() );
  vector<T> valuesOut;
  valuesOut.reserve( stepMap.size() * valuesPerKey );
  steps.reserve( stepMap.size() );

  //  create one set of data per stepMap entry
  for ( size_t i=0 ; i<stepMap.size() ; i++ )
  {
    // store the step index to use in steps, and the corresponding key in keysOut
    steps.push_back( stepMap[i].first );
    keysOut.push_back( stepMap[i].first * stepSize );

    if ( stepMap[i].second.second == 0 )
    {
      // there's only one key in that step -> copy it over
      unsigned int idx = stepMap[i].second.first * valuesPerKey;
      valuesOut.insert( valuesOut.end(), values.begin()+idx, values.begin()+idx+valuesPerKey );
    }
    else
    {
      // there are two keys -> calculate the weighted sum
      float keyStep = stepMap[i].first * stepSize;
      DP_ASSERT(    ( keys[stepMap[i].second.first] < keyStep )
                  &&  ( keyStep < keys[stepMap[i].second.second] ) );
      float dist = keys[stepMap[i].second.second] - keys[stepMap[i].second.first];
      DP_ASSERT( FLT_EPSILON < dist );
      float alpha = ( keyStep - keys[stepMap[i].second.first] ) / dist;

      unsigned int idx0 = stepMap[i].second.first * valuesPerKey;
      unsigned int idx1 = stepMap[i].second.second * valuesPerKey;
      DP_ASSERT( idx1 - idx0 == valuesPerKey );
      for ( unsigned int k=0 ; k<valuesPerKey ; k++ )
      {
        valuesOut.push_back( lerp( alpha, values[idx0+k], values[idx1+k] ) );
      }
    }
  }

  // swap keys and values now!
  keys.swap( keysOut );
  values.swap( valuesOut );
}

VertexAttributeSetSharedPtr WRLLoader::interpretVertexAttributeSet( IndexedFaceSetSharedPtr const& pIndexedFaceSet
                                                                  , unsigned int numberOfVertices
                                                                  , const vector<unsigned int> & startIndices
                                                                  , const vector<unsigned int> & faceIndices )
{
  DP_ASSERT( pIndexedFaceSet->coord.isPtrTo<Coordinate>() );
  DP_ASSERT( !pIndexedFaceSet->normal || pIndexedFaceSet->normal.isPtrTo<Normal>() );
  DP_ASSERT( !pIndexedFaceSet->color || pIndexedFaceSet->color.isPtrTo<Color>() );
  CoordinateSharedPtr const& pCoordinate = pIndexedFaceSet->coord.staticCast<Coordinate>();

  VertexAttributeSetSharedPtr vash;
#if defined(KEEP_ANIMATION)
  if ( pCoordinate->set_point || ( pIndexedFaceSet->normal && pIndexedFaceSet->normal.staticCast<Normal>()->set_vector ) || ( pIndexedFaceSet->color && pIndexedFaceSet->color.staticCast<Color>()->set_color ) )
  {
    AnimatedVertexAttributeSetSharedPtr avash = AnimatedVertexAttributeSet::create();
    //  set the animated vertices
    if ( pCoordinate->set_point )
    {
      DP_ASSERT( pCoordinate->interpreted );
      LinearInterpolatedVertexAttributeAnimationDescriptionSharedPtr livaadh = LinearInterpolatedVertexAttributeAnimationDescription::create();
      {
        LinearInterpolatedVertexAttributeAnimationDescriptionLock liadva(livaadh);
        unsigned int keyCount = dp::checked_cast<unsigned int>(pCoordinate->set_point->key.size());
        liadva->reserveKeys( keyCount );

        vector<Vec3f> vertices;
        unsigned int pointCount = dp::checked_cast<unsigned int>(pCoordinate->point.size());
        for ( unsigned int i=0 ; i<keyCount ; i++ )
        {
          gatherPerVertex<Vec3f>( vertices, pCoordinate->set_point->keyValue, pIndexedFaceSet->coordIndex
                                , numberOfVertices, i*pointCount, startIndices, pIndexedFaceSet->ccw );
          VertexAttribute va;
          va.setData( 3, dp::DataType::FLOAT_32, &vertices[0], 0, dp::checked_cast<unsigned int>(vertices.size()) );
          liadva->addKey( pCoordinate->set_point->steps[i], va );
        }
      }
      VertexAttributeAnimationSharedPtr vaah = VertexAttributeAnimation::create();
      VertexAttributeAnimationLock(vaah)->setDescription( livaadh );
      {
        AnimatedVertexAttributeSetLock avas(avash);
        avas->setAnimation( VertexAttributeSet::AttributeID::POSITION, vaah );
      }
    }

    //  set the animated normals
    if ( pIndexedFaceSet->normal && pIndexedFaceSet->normal.staticCast<Normal>()->set_vector )
    {
      NormalSharedPtr const& pNormal = pIndexedFaceSet->normal.staticCast<Normal>();
      DP_ASSERT( pNormal->interpreted );
      DP_ASSERT( pCoordinate->set_point->key == pNormal->set_vector->key );

      LinearInterpolatedVertexAttributeAnimationDescriptionSharedPtr livaadh = LinearInterpolatedVertexAttributeAnimationDescription::create();
      {
        LinearInterpolatedVertexAttributeAnimationDescriptionLock liadva(livaadh);
        unsigned int keyCount = dp::checked_cast<unsigned int>(pNormal->set_vector->key.size());
        liadva->reserveKeys( keyCount );

        vector<Vec3f> normals;
        unsigned int normalCount = dp::checked_cast<unsigned int>(pNormal->vector.size());
        if ( pIndexedFaceSet->normalPerVertex )
        {
          if ( pIndexedFaceSet->normalIndex.empty() )
          {
            for ( unsigned int i=0 ; i<keyCount ; i++ )
            {
              gatherPerVertex<Vec3f>( normals, pNormal->set_vector->keyValue, pIndexedFaceSet->coordIndex
                                    , numberOfVertices, i*normalCount, startIndices, pIndexedFaceSet->ccw );
              VertexAttribute va;
              va.setData( 3, dp::DataType::FLOAT_32, &normals[0], 0, dp::checked_cast<unsigned int>(normals.size()) );
              liadva->addKey( pNormal->set_vector->steps[i], va );
            }
          }
          else
          {
            for ( unsigned int i=0 ; i<keyCount ; i++ )
            {
              gatherPerVertex<Vec3f>( normals, pNormal->set_vector->keyValue, pIndexedFaceSet->normalIndex
                                    , numberOfVertices, i*normalCount, startIndices, pIndexedFaceSet->ccw );
              VertexAttribute va;
              va.setData( 3, dp::DataType::FLOAT_32, &normals[0], 0, dp::checked_cast<unsigned int>(normals.size()) );
              liadva->addKey( pNormal->set_vector->steps[i], va );
            }
          }
        }
        else
        {
          if ( pIndexedFaceSet->normalIndex.empty() )
          {
            for ( unsigned int i=0 ; i<keyCount ; i++ )
            {
              gatherVec3fPerFace( normals, pNormal->set_vector->keyValue, pIndexedFaceSet->coordIndex
                                , numberOfVertices, i*normalCount, startIndices, faceIndices );
              VertexAttribute va;
              va.setData( 3, dp::DataType::FLOAT_32, &normals[0], 0, dp::checked_cast<unsigned int>(normals.size()) );
              liadva->addKey( pNormal->set_vector->steps[i], va );
            }
          }
          else
          {
            for ( unsigned int i=0 ; i<keyCount ; i++ )
            {
              gatherVec3fPerFaceIndexed( normals, pNormal->set_vector->keyValue, pIndexedFaceSet->coordIndex
                                       , pIndexedFaceSet->normalIndex, numberOfVertices
                                       , i*normalCount, startIndices, faceIndices );
              VertexAttribute va;
              va.setData( 3, dp::DataType::FLOAT_32, &normals[0], 0, dp::checked_cast<unsigned int>(normals.size()) );
              liadva->addKey( pNormal->set_vector->steps[i], va );
            }
          }
        }
      }
      VertexAttributeAnimationSharedPtr vaah = VertexAttributeAnimation::create();
      VertexAttributeAnimationLock(vaah)->setDescription( livaadh );
      {
        AnimatedVertexAttributeSetLock avas(avash);
        avas->setAnimation( VertexAttributeSet::AttributeID::NORMAL, vaah );
      }
    }

    //  set the animated colors
    if ( pIndexedFaceSet->color && pIndexedFaceSet->color.staticCast<Color>()->set_color )
    {
      ColorSharedPtr const& pColor = pIndexedFaceSet->color.staticCast<Color>();
      DP_ASSERT( pColor->interpreted );
      DP_ASSERT( pCoordinate->set_point->key == pColor->set_color->key );

      LinearInterpolatedVertexAttributeAnimationDescriptionSharedPtr livaadh = LinearInterpolatedVertexAttributeAnimationDescription::create();
      {
        LinearInterpolatedVertexAttributeAnimationDescriptionLock liadva(livaadh);
        unsigned int keyCount = dp::checked_cast<unsigned int>(pColor->set_color->key.size());
        liadva->reserveKeys( keyCount );

        vector<Vec3f> colors;
        unsigned int colorCount = dp::checked_cast<unsigned int>(pColor->color.size());
        if ( pIndexedFaceSet->colorPerVertex )
        {
          if ( pIndexedFaceSet->colorIndex.empty() )
          {
            for ( unsigned int i=0 ; i<keyCount ; i++ )
            {
              gatherPerVertex<Vec3f>( colors, pColor->set_color->keyValue, pIndexedFaceSet->coordIndex
                                    , numberOfVertices, i*colorCount, startIndices, pIndexedFaceSet->ccw );
              VertexAttribute va;
              va.setData( 3, dp::DataType::FLOAT_32, &colors[0], 0, dp::checked_cast<unsigned int>(colors.size()) );
              liadva->addKey( pColor->set_color->steps[i], va );
            }
          }
          else
          {
            for ( unsigned int i=0 ; i<keyCount ; i++ )
            {
              gatherPerVertex<Vec3f>( colors, pColor->set_color->keyValue, pIndexedFaceSet->colorIndex
                                    , numberOfVertices, i*colorCount, startIndices, pIndexedFaceSet->ccw );
              VertexAttribute va;
              va.setData( 3, dp::DataType::FLOAT_32, &colors[0], 0, dp::checked_cast<unsigned int>(colors.size()) );
              liadva->addKey( pColor->set_color->steps[i], va );
            }
          }
        }
        else
        {
          if ( pIndexedFaceSet->colorIndex.empty() )
          {
            for ( unsigned int i=0 ; i<keyCount ; i++ )
            {
              gatherVec3fPerFace( colors, pColor->set_color->keyValue, pIndexedFaceSet->coordIndex
                                , numberOfVertices, i*colorCount, startIndices, faceIndices );
              VertexAttribute va;
              va.setData( 3, dp::DataType::FLOAT_32, &colors[0], 0, dp::checked_cast<unsigned int>(colors.size()) );
              liadva->addKey( pColor->set_color->steps[i], va );
            }
          }
          else
          {
            for ( unsigned int i=0 ; i<keyCount ; i++ )
            {
              gatherVec3fPerFaceIndexed( colors, pColor->set_color->keyValue, pIndexedFaceSet->coordIndex
                                       , pIndexedFaceSet->colorIndex, numberOfVertices
                                       , i*colorCount, startIndices, faceIndices );
              VertexAttribute va;
              va.setData( 3, dp::DataType::FLOAT_32, &colors[0], 0, dp::checked_cast<unsigned int>(colors.size()) );
              liadva->addKey( pColor->set_color->steps[i], va );
            }
          }
        }
      }
      VertexAttributeAnimationSharedPtr vaah = VertexAttributeAnimation::create();
      VertexAttributeAnimationLock(vaah)->setDescription( livaadh );
      {
        AnimatedVertexAttributeSetLock avas(avash);
        avas->setAnimation( VertexAttributeSet::DP_COLOR, vaah );
      }
    }
    vash = avash;
  }
  else
#endif
  {
    vash = VertexAttributeSet::create();
  }

  {
    //  set the vertices
    vector<Vec3f> vertices;
    gatherPerVertex<Vec3f>( vertices, pCoordinate->point, pIndexedFaceSet->coordIndex
                          , numberOfVertices, 0, startIndices, pIndexedFaceSet->ccw );
    vash->setVertices( &vertices[0], numberOfVertices );

    //  set the normals
    if ( pIndexedFaceSet->normal )
    {
      NormalSharedPtr const& pNormal = pIndexedFaceSet->normal.staticCast<Normal>();
      vector<Vec3f> normals;
      if ( pIndexedFaceSet->normalPerVertex )
      {
        gatherPerVertex<Vec3f>( normals, pNormal->vector,
                                pIndexedFaceSet->normalIndex.empty()
                                ? pIndexedFaceSet->coordIndex
                                : pIndexedFaceSet->normalIndex,
                                numberOfVertices, 0, startIndices, pIndexedFaceSet->ccw );
      }
      else
      {
        if ( pIndexedFaceSet->normalIndex.empty() )
        {
          DP_ASSERT( false );
          gatherVec3fPerFace( normals, pNormal->vector, pIndexedFaceSet->coordIndex, numberOfVertices, 0, startIndices
                            , faceIndices );
        }
        else
        {
          gatherVec3fPerFaceIndexed( normals, pNormal->vector, pIndexedFaceSet->coordIndex, pIndexedFaceSet->normalIndex
                                   , numberOfVertices, 0, startIndices, faceIndices );
        }
      }
      vash->setNormals( &normals[0], numberOfVertices );
    }

    //  set the texture coordinates
    if ( pIndexedFaceSet->texCoord )
    {
      TextureCoordinateSharedPtr const& pTextureCoordinate = pIndexedFaceSet->texCoord.staticCast<TextureCoordinate>();
      vector<Vec2f> texCoords;
      gatherPerVertex<Vec2f>( texCoords, pTextureCoordinate->point,
                              pIndexedFaceSet->texCoordIndex.empty()
                              ? pIndexedFaceSet->coordIndex
                              : pIndexedFaceSet->texCoordIndex,
                              numberOfVertices, 0, startIndices, pIndexedFaceSet->ccw );
      vash->setTexCoords( 0, &texCoords[0], numberOfVertices );
    }

    //  set the colors
    if ( pIndexedFaceSet->color )
    {
      ColorSharedPtr const& pColor = pIndexedFaceSet->color.staticCast<Color>();
      vector<Vec3f> colors;
      if ( pIndexedFaceSet->colorPerVertex )
      {
        gatherPerVertex<Vec3f>( colors, pColor->color,
                                pIndexedFaceSet->colorIndex.empty()
                                ? pIndexedFaceSet->coordIndex
                                : pIndexedFaceSet->colorIndex,
                                numberOfVertices, 0, startIndices, pIndexedFaceSet->ccw );
      }
      else
      {
        if ( pIndexedFaceSet->colorIndex.empty() )
        {
          DP_ASSERT( false );
          gatherVec3fPerFace( colors, pColor->color, pIndexedFaceSet->coordIndex, numberOfVertices, 0, startIndices
                            , faceIndices );
        }
        else
        {
          gatherVec3fPerFaceIndexed( colors, pColor->color, pIndexedFaceSet->coordIndex, pIndexedFaceSet->colorIndex
                                   , numberOfVertices, 0, startIndices, faceIndices );
        }
      }
      vash->setColors( &colors[0], numberOfVertices );
    }
  }

  return( vash );
}

void  WRLLoader::interpretIndexedLineSet( IndexedLineSetSharedPtr const& pIndexedLineSet
                                        , vector<PrimitiveSharedPtr> &primitives )
{
  if ( pIndexedLineSet->pLineStrips )
  {
    primitives.push_back( pIndexedLineSet->pLineStrips );
  }
  else if ( pIndexedLineSet->coordIndex.size() )
  {
    vector<unsigned int> indices( pIndexedLineSet->coordIndex.size() );
    unsigned int ic = 0;
    for ( size_t i=0 ; i<pIndexedLineSet->coordIndex.size() ; i++ )
    {
      indices[i] = ( pIndexedLineSet->coordIndex[i] == -1 ) ? ~0 : ic++;
    }
    if ( indices.back() == ~0 )
    {
      indices.pop_back();
    }
    IndexSetSharedPtr iset( IndexSet::create() );
    iset->setData( &indices[0] , dp::checked_cast<unsigned int>(indices.size()) );

    DP_ASSERT( pIndexedLineSet->coord.isPtrTo<Coordinate>() );
    CoordinateSharedPtr const& pCoordinate = pIndexedLineSet->coord.staticCast<Coordinate>();
    DP_ASSERT( pCoordinate && ! pCoordinate->set_point );
    vector<Vec3f> vertices( ic );
    for ( size_t i=0, j=0 ; i<pIndexedLineSet->coordIndex.size() ; i++ )
    {
      if ( pIndexedLineSet->coordIndex[i] != -1 )
      {
        vertices[j++] = pCoordinate->point[pIndexedLineSet->coordIndex[i]];
      }
    }
    VertexAttributeSetSharedPtr cvas = VertexAttributeSet::create();
    cvas->setVertices( &vertices[0], dp::checked_cast<unsigned int>(vertices.size()) );

    if ( pIndexedLineSet->color )
    {
      ColorSharedPtr const& pColor = pIndexedLineSet->color.staticCast<Color>();
      vector<Vec3f> colors( vertices.size() );
      if ( pIndexedLineSet->colorPerVertex )
      {
        MFInt32 &colorIndices = pIndexedLineSet->colorIndex.empty()
                                ? pIndexedLineSet->coordIndex
                                : pIndexedLineSet->colorIndex;
        for ( size_t i=0, j=0 ; i<colorIndices.size() ; i++ )
        {
          if ( colorIndices[i] != -1 )
          {
            colors[j++] = pColor->color[colorIndices[i]];
          }
        }
      }
      else
      {
        if ( pIndexedLineSet->colorIndex.empty() )
        {
          for ( size_t i=0, j=0, k=0 ; i<pIndexedLineSet->coordIndex.size() ; i++ )
          {
            if ( pIndexedLineSet->coordIndex[i] == -1 )
            {
              k++;
            }
            else
            {
              colors[j++] = pColor->color[k];
            }
          }
        }
        else
        {
          for ( size_t i=0, j=0, k=0 ; i<pIndexedLineSet->coordIndex.size() ; i++ )
          {
            if ( pIndexedLineSet->coordIndex[i] == -1 )
            {
              k++;
            }
            else
            {
              colors[j++] = pColor->color[pIndexedLineSet->colorIndex[k]];
            }
          }
        }
      }
      cvas->setColors( &colors[0], dp::checked_cast<unsigned int>(colors.size()) );
    }

    PrimitiveSharedPtr pLineStrips = Primitive::create( PrimitiveType::LINE_STRIP );
    pLineStrips->setName( pIndexedLineSet->getName() );
    pLineStrips->setIndexSet( iset );
    pLineStrips->setVertexAttributeSet( cvas );
    primitives.push_back( pLineStrips );
    pIndexedLineSet->pLineStrips = pLineStrips;
  }
}

NodeSharedPtr WRLLoader::interpretInline( InlineSharedPtr const& pInline )
{
  if ( ! pInline->pNode )
  {
    string  fileName;
    if ( interpretURL( pInline->url, fileName ) )
    {
      dp::sg::ui::ViewStateSharedPtr viewState = dp::sg::io::loadScene( fileName, m_fileFinder, callback() );
      DP_ASSERT( viewState && viewState->getScene() );
      pInline->pNode = viewState->getScene()->getRootNode();
    }
  }
  return( pInline->pNode );
}

dp::sg::core::LODSharedPtr WRLLoader::interpretLOD( vrml::LODSharedPtr const& pVRMLLOD )
{
  dp::sg::core::LODSharedPtr pLOD;

  if ( pVRMLLOD->pLOD )
  {
    pLOD = pVRMLLOD->pLOD;
  }
  else
  {
    pLOD = dp::sg::core::LOD::create();
    pLOD->setName( pVRMLLOD->getName() );
    interpretChildren( pVRMLLOD->children, pLOD );
    if ( pVRMLLOD->range.size() != 0 )
    {
      pLOD->setRanges( &pVRMLLOD->range[0], dp::checked_cast<unsigned int>(pVRMLLOD->range.size()) );
    }
    else
    {
      vector<float> ranges( pLOD->getNumberOfChildren() - 1 );
      float dist = 0.0f;
      for ( dp::sg::core::Group::ChildrenIterator gci = pLOD->beginChildren() ; gci != pLOD->endChildren() ; ++gci )
      {
        DP_ASSERT( dp::math::isValid((*gci)->getBoundingSphere()) );
        float radius = (*gci)->getBoundingSphere().getRadius();
        if ( dist < radius )
        {
          dist = radius;
        }
      }
      dist *= 10.0f;
      for ( size_t i=0 ; i<ranges.size() ; i++, dist*=10.0f )
      {
        ranges[i] = dist;
      }
      pLOD->setRanges( &ranges[0], dp::checked_cast<unsigned int>(ranges.size()) );
    }
    pLOD->setCenter( pVRMLLOD->center );

    pVRMLLOD->pLOD = pLOD;
  }

  return( pLOD );
}

ParameterGroupDataSharedPtr WRLLoader::interpretMaterial( vrml::MaterialSharedPtr const& material )
{
  ParameterGroupDataSharedPtr materialParameters;
  if ( ! material->materialParameters )
  {
    const dp::fx::EffectSpecSharedPtr & es = getStandardMaterialSpec();
    dp::fx::EffectSpec::iterator pgsit = es->findParameterGroupSpec( string( "standardMaterialParameters" ) );
    DP_ASSERT( pgsit != es->endParameterGroupSpecs() );
    material->materialParameters = dp::sg::core::ParameterGroupData::create( *pgsit );

    Vec3f ambientColor = material->ambientIntensity * interpretSFColor( material->diffuseColor );
    Vec3f diffuseColor = interpretSFColor( material->diffuseColor );
    Vec3f specularColor = interpretSFColor( material->specularColor );
    float specularExponent = 128.0f * material->shininess;
    Vec3f emissiveColor = interpretSFColor( material->emissiveColor );
    float opacity = 1.0f - material->transparency;

    DP_VERIFY( material->materialParameters->setParameter( "backAmbientColor", ambientColor ) );
    DP_VERIFY( material->materialParameters->setParameter( "backDiffuseColor", diffuseColor ) );
    DP_VERIFY( material->materialParameters->setParameter( "backSpecularColor", specularColor ) );
    DP_VERIFY( material->materialParameters->setParameter( "backSpecularExponent", specularExponent ) );
    DP_VERIFY( material->materialParameters->setParameter( "backEmissiveColor", emissiveColor ) );
    DP_VERIFY( material->materialParameters->setParameter( "backOpacity", opacity ) );
    DP_VERIFY( material->materialParameters->setParameter( "frontAmbientColor", ambientColor ) );
    DP_VERIFY( material->materialParameters->setParameter( "frontDiffuseColor", diffuseColor ) );
    DP_VERIFY( material->materialParameters->setParameter( "frontSpecularColor", specularColor ) );
    DP_VERIFY( material->materialParameters->setParameter( "frontSpecularExponent", specularExponent ) );
    DP_VERIFY( material->materialParameters->setParameter( "frontEmissiveColor", emissiveColor ) );
    DP_VERIFY( material->materialParameters->setParameter( "frontOpacity", opacity ) );
    DP_VERIFY( material->materialParameters->setParameter( "unlitColor", Vec4f( diffuseColor, opacity ) ) );
  }
  return( material->materialParameters );
}

ParameterGroupDataSharedPtr WRLLoader::interpretMovieTexture( MovieTextureSharedPtr const& pMovieTexture )
{
  onUnsupportedToken( "VRMLLoader", "MovieTexture" );
  return( ParameterGroupDataSharedPtr() );
}

void WRLLoader::interpretNormal( NormalSharedPtr const& pNormal )
{
  if ( !pNormal->interpreted )
  {
    if ( pNormal->set_vector )
    {
      interpretNormalInterpolator( pNormal->set_vector, dp::checked_cast<unsigned int>(pNormal->vector.size()) );
    }
    pNormal->interpreted = true;
  }
}

void WRLLoader::interpretNormalInterpolator( NormalInterpolatorSharedPtr const& pNormalInterpolator
                                           , unsigned int vectorCount )
{
  if ( ! pNormalInterpolator->interpreted )
  {
    SFTime cycleInterval = pNormalInterpolator->set_fraction ? pNormalInterpolator->set_fraction->cycleInterval : 1.0;
    resampleKeyValues( pNormalInterpolator->key, pNormalInterpolator->keyValue, vectorCount
                     , pNormalInterpolator->steps, cycleInterval );
    for_each( pNormalInterpolator->keyValue.begin(), pNormalInterpolator->keyValue.end()
            , std::mem_fun_ref(&Vecnt<3,float>::normalize) );
    pNormalInterpolator->interpreted = true;
  }
}

void  WRLLoader::interpretOrientationInterpolator( OrientationInterpolatorSharedPtr const& pOrientationInterpolator )
{
  if ( ! pOrientationInterpolator->interpreted )
  {
    SFTime cycleInterval = pOrientationInterpolator->set_fraction
                           ? pOrientationInterpolator->set_fraction->cycleInterval
                           : 1.0;
    resampleKeyValues( pOrientationInterpolator->key, pOrientationInterpolator->keyValue, 1
                     , pOrientationInterpolator->steps, cycleInterval );
    pOrientationInterpolator->keyValueQuatf.resize( pOrientationInterpolator->keyValue.size() );
    for ( size_t i=0 ;i<pOrientationInterpolator->keyValueQuatf.size() ; i++ )
    {
      pOrientationInterpolator->keyValueQuatf[i] = interpretSFRotation( pOrientationInterpolator->keyValue[i] );
    }
    pOrientationInterpolator->interpreted = true;
  }
}

ParameterGroupDataSharedPtr WRLLoader::interpretPixelTexture( PixelTextureSharedPtr const& pPixelTexture )
{
  onUnsupportedToken( "VRMLLoader", "PixelTexture" );
  return( ParameterGroupDataSharedPtr() );
}

LightSourceSharedPtr WRLLoader::interpretPointLight( vrml::PointLightSharedPtr const& pVRMLPointLight )
{
  LightSourceSharedPtr lightSource;
  if ( pVRMLPointLight->lightSource )
  {
    lightSource = pVRMLPointLight->lightSource;
  }
  else
  {
    Vec3f color( interpretSFColor( pVRMLPointLight->color ) );
    lightSource = createStandardPointLight( pVRMLPointLight->location
                                          , pVRMLPointLight->ambientIntensity * color
                                          , pVRMLPointLight->intensity * color
                                          , pVRMLPointLight->intensity * color
                                          , { pVRMLPointLight->attenuation[0], pVRMLPointLight->attenuation[1], pVRMLPointLight->attenuation[2] } );
    lightSource->setName( pVRMLPointLight->getName() );
    lightSource->setEnabled( pVRMLPointLight->on );
  }
  return( lightSource );
}

void  WRLLoader::interpretPointSet( PointSetSharedPtr const& pPointSet, vector<PrimitiveSharedPtr> &primitives )
{
  if ( pPointSet->pPoints )
  {
    primitives.push_back( pPointSet->pPoints );
  }
  else if ( pPointSet->coord )
  {
    VertexAttributeSetSharedPtr cvas = VertexAttributeSet::create();

    DP_ASSERT( pPointSet->coord.isPtrTo<Coordinate>() );
    CoordinateSharedPtr const& pCoordinate = pPointSet->coord.staticCast<Coordinate>();
    DP_ASSERT( pCoordinate->point.size() < UINT_MAX );
    vector<Vec3f> vertices( pCoordinate->point.size() );
    for ( unsigned i=0 ; i<pCoordinate->point.size() ; i++ )
    {
      vertices[i] = pCoordinate->point[i];
    }
    cvas->setVertices( &vertices[0], dp::checked_cast<unsigned int>(vertices.size()) );

    if ( pPointSet->color )
    {
      DP_ASSERT( pPointSet->color.isPtrTo<Color>() );
      ColorSharedPtr const& pColor = pPointSet->color.staticCast<Color>();
      DP_ASSERT( pCoordinate->point.size() <= pColor->color.size() );
      vector<Vec3f> colors( pCoordinate->point.size() );
      for ( size_t i=0 ; i<pCoordinate->point.size() ; i++ )
      {
        colors[i] = pColor->color[i];
      }
      cvas->setColors( &colors[0], dp::checked_cast<unsigned int>(colors.size()) );
    }

    PrimitiveSharedPtr pPoints = Primitive::create( PrimitiveType::POINTS );
    pPoints->setName( pPointSet->getName() );
    pPoints->setVertexAttributeSet( cvas );

    primitives.push_back( pPoints );
    pPointSet->pPoints = pPoints;
  }
}

void WRLLoader::interpretPositionInterpolator( PositionInterpolatorSharedPtr const& pPositionInterpolator )
{
  if ( !pPositionInterpolator->interpreted )
  {
    SFTime cycleInterval = pPositionInterpolator->set_fraction ? pPositionInterpolator->set_fraction->cycleInterval : 1.0;
    resampleKeyValues( pPositionInterpolator->key, pPositionInterpolator->keyValue, 1
                     , pPositionInterpolator->steps, cycleInterval );
    pPositionInterpolator->interpreted = true;
  }
}

dp::sg::core::ObjectSharedPtr WRLLoader::interpretSFNode( const SFNode n )
{
  dp::sg::core::ObjectSharedPtr pObject;
  if ( n.isPtrTo<Appearance>() )
  {
    DP_ASSERT( false );
  }
  else if ( n.isPtrTo<Background>() )
  {
    interpretBackground( n.staticCast<Background>() );
  }
  else if ( n.isPtrTo<Coordinate>() )
  {
    //  NOP
  }
  else if ( n.isPtrTo<vrml::Group>() )
  {
    if ( n.isPtrTo<vrml::Billboard>() )
    {
      pObject = interpretBillboard( n.staticCast<vrml::Billboard>() );
    }
    else if ( n.isPtrTo<vrml::LOD>() )
    {
      pObject = interpretLOD( n.staticCast<vrml::LOD>() );
    }
    else if ( n.isPtrTo<vrml::Switch>() )
    {
      pObject = interpretSwitch( n.staticCast<vrml::Switch>() );
    }
    else if ( n.isPtrTo<vrml::Transform>() )
    {
      pObject = interpretTransform( n.staticCast<vrml::Transform>() );
    }
    else
    {
      pObject = interpretGroup( n.staticCast<vrml::Group>() );
    }
  }
  else if ( n.isPtrTo<Inline>() )
  {
    pObject = interpretInline( n.staticCast<Inline>() );
  }
  else if ( n.isPtrTo<Interpolator>() )
  {
    //  NOP
  }
  else if ( n.isPtrTo<Light>() )
  {
    if ( n.isPtrTo<DirectionalLight>() )
    {
      pObject = interpretDirectionalLight( n.staticCast<DirectionalLight>() );
    }
    else if ( n.isPtrTo<vrml::PointLight>() )
    {
      pObject = interpretPointLight( n.staticCast<vrml::PointLight>() );
    }
    else
    {
      DP_ASSERT( n.isPtrTo<vrml::SpotLight>() );
      pObject = interpretSpotLight( n.staticCast<vrml::SpotLight>() );
    }
  }
  else if ( n.isPtrTo<vrml::Shape>() )
  {
    pObject = interpretShape( n.staticCast<vrml::Shape>() );
  }
  else if ( n.isPtrTo<TimeSensor>() )
  {
    //  NOP
  }
  else if ( n.isPtrTo<Viewpoint>() )
  {
    pObject = interpretViewpoint( n.staticCast<Viewpoint>() );
  }
  else
  {
    DP_ASSERT( false );
  }

  return( pObject );
}

Quatf WRLLoader::interpretSFRotation( const SFRotation &r )
{
  Vec3f v( r[0], r[1], r[2] );
  v.normalize();
  return( Quatf( v, r[3] ) );
}

NodeSharedPtr WRLLoader::interpretShape( vrml::ShapeSharedPtr const& pShape )
{
  if ( ! pShape->pNode )
  {
    dp::sg::core::PipelineDataSharedPtr materialPipeline;
    vector<PrimitiveSharedPtr> primitives;

    if ( pShape->appearance )
    {
      DP_ASSERT( pShape->appearance.isPtrTo<Appearance>() );
      materialPipeline = interpretAppearance( pShape->appearance.staticCast<Appearance>() );
    }
    if ( pShape->geometry )
    {
      DP_ASSERT(      pShape->geometry.isPtrTo<Geometry>()
                  &&  ( !pShape->appearance || pShape->appearance.isPtrTo<Appearance>() ) );
      interpretGeometry( pShape->geometry.staticCast<Geometry>(), primitives
                       , pShape->appearance && pShape->appearance.staticCast<Appearance>()->texture );
    }

    if ( materialPipeline && pShape->geometry )
    {
      const ParameterGroupDataSharedPtr & pgd = materialPipeline->findParameterGroupData( string( "standardTextureParameters" ) );
      if ( pgd )
      {
        if ( pShape->geometry.isPtrTo<IndexedFaceSet>() )
        {
          IndexedFaceSetSharedPtr const& pIndexedFaceSet = pShape->geometry.staticCast<IndexedFaceSet>();
          if ( ! pIndexedFaceSet->texCoord )
          {
            determineTexGen( pIndexedFaceSet, pgd );
          }
        }
        else if (   pShape->geometry.isPtrTo<Box>()
                ||  pShape->geometry.isPtrTo<Cone>()
                ||  pShape->geometry.isPtrTo<Cylinder>()
                ||  pShape->geometry.isPtrTo<ElevationGrid>()
                ||  pShape->geometry.isPtrTo<IndexedLineSet>()
                ||  pShape->geometry.isPtrTo<PointSet>()
                ||  pShape->geometry.isPtrTo<Sphere>() )
        {
          //  NOP
        }
        else if (   pShape->geometry.isPtrTo<Extrusion>()
                 || pShape->geometry.isPtrTo<Text>() )
        {
          DP_ASSERT( false );
        }
      }
    }

    if ( ! primitives.empty() )
    {
      if ( primitives.size() == 1 )
      {
        GeoNodeSharedPtr geoNode = GeoNode::create();
        geoNode->setName( pShape->getName() );
        geoNode->setMaterialPipeline( materialPipeline );
        geoNode->setPrimitive( primitives[0] );
        pShape->pNode = geoNode;
      }
      else
      {
        dp::sg::core::GroupSharedPtr group = dp::sg::core::Group::create();
        group->setName( pShape->getName() );
        for ( size_t i=0 ; i<primitives.size(); i++ )
        {
          GeoNodeSharedPtr geoNode = GeoNode::create();
          geoNode->setMaterialPipeline( materialPipeline );
          geoNode->setPrimitive( primitives[i] );
          group->addChild( geoNode );
        }
        pShape->pNode = group;
      }
    }
  }
  return( pShape->pNode );
}

LightSourceSharedPtr WRLLoader::interpretSpotLight( vrml::SpotLightSharedPtr const& pVRMLSpotLight )
{
  LightSourceSharedPtr lightSource;
  if ( pVRMLSpotLight->lightSource )
  {
    lightSource = pVRMLSpotLight->lightSource;
  }
  else
  {
    Vec3f color( interpretSFColor( pVRMLSpotLight->color ) );
    float exponent = FLT_EPSILON < pVRMLSpotLight->beamWidth
                    ? pVRMLSpotLight->beamWidth < pVRMLSpotLight->cutOffAngle
                      ? pVRMLSpotLight->cutOffAngle / pVRMLSpotLight->beamWidth
                      : 0.0f
                    : 10000.0f;   // some really large value for beamWidth == 0
    lightSource = createStandardSpotLight( pVRMLSpotLight->location
                                         , pVRMLSpotLight->direction
                                         , pVRMLSpotLight->ambientIntensity * color
                                         , pVRMLSpotLight->intensity * color
                                         , pVRMLSpotLight->intensity * color
                                         , { pVRMLSpotLight->attenuation[0], pVRMLSpotLight->attenuation[1], pVRMLSpotLight->attenuation[2] }
                                         , exponent
                                         , pVRMLSpotLight->cutOffAngle );
    lightSource->setName( pVRMLSpotLight->getName() );
    lightSource->setEnabled( pVRMLSpotLight->on );
  }
  return( lightSource );
}

ParameterGroupDataSharedPtr WRLLoader::interpretTexture( vrml::TextureSharedPtr const& pTexture )
{
  ParameterGroupDataSharedPtr textureData;
  if ( pTexture.isPtrTo<ImageTexture>() )
  {
    textureData = interpretImageTexture( pTexture.staticCast<ImageTexture>() );
  }
  else if ( pTexture.isPtrTo<MovieTexture>() )
  {
    textureData = interpretMovieTexture( pTexture.staticCast<MovieTexture>() );
  }
  else
  {
    DP_ASSERT( pTexture.isPtrTo<PixelTexture>() );
    textureData = interpretPixelTexture( pTexture.staticCast<PixelTexture>() );
  }
  return( textureData );
}

dp::sg::core::SwitchSharedPtr WRLLoader::interpretSwitch( vrml::SwitchSharedPtr const& pVRMLSwitch )
{
  dp::sg::core::SwitchSharedPtr pSwitch;

  if ( pVRMLSwitch->pSwitch )
  {
    pSwitch = pVRMLSwitch->pSwitch;
  }
  else
  {
    pSwitch = dp::sg::core::Switch::create();
    pSwitch->setName( pVRMLSwitch->getName() );
    interpretChildren( pVRMLSwitch->children, pSwitch );
    if ( ( pVRMLSwitch->whichChoice < 0 ) || ( (SFInt32)pSwitch->getNumberOfChildren() <= pVRMLSwitch->whichChoice ) )
    {
      pSwitch->setInactive();
    }
    else
    {
      pSwitch->setActive( pVRMLSwitch->whichChoice );
    }
    pVRMLSwitch->pSwitch = pSwitch;
  }

  return( pSwitch );
}

void WRLLoader::interpretTextureTransform( TextureTransformSharedPtr const& pTextureTransform
                                         , const ParameterGroupDataSharedPtr & textureData )
{
  Trafo t;
  t.setCenter( Vec3f( pTextureTransform->center[0], pTextureTransform->center[1], 0.0f ) );
  t.setOrientation( Quatf( Vec3f( 0.0f, 0.0f, 1.0f ), pTextureTransform->rotation) );
  t.setScaling( Vec3f( pTextureTransform->scale[0], pTextureTransform->scale[1], 1.0f ) );
  t.setTranslation( Vec3f( pTextureTransform->translation[0], pTextureTransform->translation[1], 0.0f ) );

  DP_VERIFY( textureData->setParameter<Mat44f>( "textureMatrix", t.getMatrix() ) );
}

dp::sg::core::TransformSharedPtr WRLLoader::interpretTransform( vrml::TransformSharedPtr const& pVRMLTransform )
{
  dp::sg::core::TransformSharedPtr pTransform;

  if ( pVRMLTransform->pTransform )
  {
    pTransform = pVRMLTransform->pTransform;
  }
  else
  {
#if defined(KEEP_ANIMATION)
    PositionInterpolatorSharedPtr const& center = pVRMLTransform->set_center;
    OrientationInterpolatorSharedPtr const& rotation = pVRMLTransform->set_rotation;
    PositionInterpolatorSharedPtr const& scale = pVRMLTransform->set_scale;
    PositionInterpolatorSharedPtr const& translation = pVRMLTransform->set_translation;
    if ( center || rotation || scale || translation )
    {
      if (center )
      {
        interpretPositionInterpolator( center );
      }
      Quatf transformRot;
      if ( rotation )
      {
        interpretOrientationInterpolator( rotation );
      }
      else
      {
        transformRot = interpretSFRotation( pVRMLTransform->rotation );
      }
      if ( scale )
      {
        interpretPositionInterpolator( scale );
      }
      if ( translation )
      {
        interpretPositionInterpolator( translation );
      }
      Quatf scaleOrientation = interpretSFRotation( pVRMLTransform->scaleOrientation );

      vector<unsigned int> keys = getCombinedKeys( center, rotation, scale, translation );

      LinearInterpolatedTrafoAnimationDescriptionSharedPtr litadh = LinearInterpolatedTrafoAnimationDescription::create();
      LinearInterpolatedTrafoAnimationDescriptionLock liadt( litadh );
      liadt->reserveKeys( dp::checked_cast<unsigned int>(keys.size()) );

      for ( size_t i=0, centerStep=0, rotationStep=0, scaleStep=0, translationStep=0 ; i<keys.size() ; i++ )
      {
        Trafo trafo;
        if ( center )
        {
          if ( center->steps[centerStep] < keys[i] )
          {
            centerStep++;
            DP_ASSERT( keys[i] <= center->steps[centerStep] );
          }
          if ( keys[i] == center->steps[centerStep] )
          {
            trafo.setCenter( center->keyValue[centerStep] );
          }
          else
          {
            DP_ASSERT( keys[i] < center->steps[centerStep] );
            float alpha = (float)( keys[i] - center->steps[centerStep-1] )
                        / (float)( center->steps[centerStep] - center->steps[centerStep-1] );
            trafo.setCenter( lerp( alpha, center->keyValue[centerStep-1], center->keyValue[centerStep] ) );
          }
        }
        else
        {
          trafo.setCenter( pVRMLTransform->center );
        }
        if ( rotation )
        {
          if ( rotation->steps[rotationStep] < keys[i] )
          {
            rotationStep++;
            DP_ASSERT( keys[i] <= rotation->steps[rotationStep] );
          }
          if ( keys[i] == rotation->steps[rotationStep] )
          {
            trafo.setOrientation( rotation->keyValueQuatf[rotationStep] );
          }
          else
          {
            DP_ASSERT( keys[i] < rotation->steps[rotationStep] );
            float alpha = (float)( keys[i] - rotation->steps[rotationStep-1] )
                        / (float)( rotation->steps[rotationStep] - rotation->steps[rotationStep-1] );
            Quatf rot = lerp( alpha, rotation->keyValueQuatf[rotationStep-1], rotation->keyValueQuatf[rotationStep] );
            rot.normalize();
            trafo.setOrientation( rot );
          }
        }
        else
        {
          trafo.setOrientation( transformRot );
        }
        if ( scale )
        {
          if ( scale->steps[scaleStep] < keys[i] )
          {
            scaleStep++;
            DP_ASSERT( keys[i] <= scale->steps[scaleStep] );
          }
          if ( keys[i] == scale->steps[scaleStep] )
          {
            trafo.setScaling( scale->keyValue[scaleStep] );
          }
          else
          {
            DP_ASSERT( keys[i] < scale->steps[scaleStep] );
            float alpha = (float)( keys[i] - scale->steps[scaleStep-1] )
                        / (float)( scale->steps[scaleStep] - scale->steps[scaleStep-1] );
            trafo.setScaling( lerp( alpha, scale->keyValue[scaleStep-1], scale->keyValue[scaleStep] ) );
          }
        }
        else
        {
          trafo.setScaling( pVRMLTransform->scale );
        }
        trafo.setScaleOrientation( scaleOrientation );
        if ( translation )
        {
          if ( translation->steps[translationStep] < keys[i] )
          {
            translationStep++;
            DP_ASSERT( keys[i] <= translation->steps[translationStep] );
          }
          if ( keys[i] == translation->steps[translationStep] )
          {
            trafo.setTranslation( translation->keyValue[translationStep] );
          }
          else
          {
            DP_ASSERT( keys[i] < translation->steps[translationStep] );
            float alpha = (float)( keys[i] - translation->steps[translationStep-1] )
                        / (float)( translation->steps[translationStep] - translation->steps[translationStep-1] );
            trafo.setTranslation( lerp( alpha, translation->keyValue[translationStep-1], translation->keyValue[translationStep] ) );
          }
        }
        else
        {
          trafo.setTranslation( pVRMLTransform->translation );
        }
        liadt->addKey( keys[i], trafo );
      }

      TrafoAnimationSharedPtr tah = TrafoAnimation::create();
      TrafoAnimationLock(tah)->setDescription( litadh );

      AnimatedTransformSharedPtr pAnimatedTransform = AnimatedTransform::create();
      AnimatedTransformLock(pAnimatedTransform)->setAnimation(tah);
      pTransform = pAnimatedTransform;
    }
    else
#endif
    {
      Trafo trafo;
      trafo.setCenter( pVRMLTransform->center );
      trafo.setOrientation( interpretSFRotation( pVRMLTransform->rotation ) );
      trafo.setScaling( pVRMLTransform->scale );
      trafo.setScaleOrientation( interpretSFRotation( pVRMLTransform->scaleOrientation ) );
      trafo.setTranslation( pVRMLTransform->translation );

      pTransform = dp::sg::core::Transform::create();
      pTransform->setTrafo( trafo );
    }

    pTransform->setName( pVRMLTransform->getName() );
    interpretChildren( pVRMLTransform->children, pTransform );

    pVRMLTransform->pTransform = pTransform;
  }

  return( pTransform );
}

bool  WRLLoader::interpretURL( const MFString &url, string &fileName )
{
  bool found = false;
  for ( size_t i=0 ; ! found && i<url.size() ; i++ )
  {
    fileName = m_fileFinder.find( url[i] );
    found = !fileName.empty();
  }
  onFilesNotFound( found, url );
  return( found );
}

dp::sg::core::ObjectSharedPtr  WRLLoader::interpretViewpoint( ViewpointSharedPtr const& pViewpoint )
{
#if (KEEP_ANIMATION)
  OrientationInterpolatorSharedPtr const& orientation = pViewpoint->set_orientation;
  PositionInterpolatorSharedPtr const& position = pViewpoint->set_position;
  if ( orientation || position )
  {
    Quatf rot;
    if ( orientation )
    {
      interpretOrientationInterpolator( orientation );
    }
    else
    {
      rot = interpretSFRotation( pViewpoint->orientation );
    }
    if ( position )
    {
      interpretPositionInterpolator( position );
    }

    vector<unsigned int> keys = getCombinedKeys( position, orientation, NULL, NULL );
    LinearInterpolatedTrafoAnimationDescriptionSharedPtr litadh = LinearInterpolatedTrafoAnimationDescription::create();
    {
      LinearInterpolatedTrafoAnimationDescriptionLock liadt(litadh);
      liadt->reserveKeys( dp::checked_cast<unsigned int>(keys.size()) );

      for ( unsigned int i=0, orientationStep=0, positionStep=0 ; i<keys.size() ; i++ )
      {
        Trafo trafo;
        if ( orientation )
        {
          if ( orientation->steps[orientationStep] < keys[i] )
          {
            orientationStep++;
            DP_ASSERT( keys[i] <= orientation->steps[orientationStep] );
          }
          if ( keys[i] == orientation->steps[orientationStep] )
          {
            trafo.setOrientation( orientation->keyValueQuatf[orientationStep] );
          }
          else
          {
            DP_ASSERT( keys[i] < orientation->steps[orientationStep] );
            float alpha = (float)( keys[i] - orientation->steps[orientationStep-1] )
                        / (float)( orientation->steps[orientationStep] - orientation->steps[orientationStep-1] );
            Quatf rot = lerp( alpha, orientation->keyValueQuatf[orientationStep-1], orientation->keyValueQuatf[orientationStep] );
            rot.normalize();
            trafo.setOrientation( rot );
          }
        }
        else
        {
          trafo.setOrientation( rot );
        }
        if ( position )
        {
          if ( position->steps[positionStep] < keys[i] )
          {
            positionStep++;
            DP_ASSERT( keys[i] <= position->steps[positionStep] );
          }
          if ( keys[i] == position->steps[positionStep] )
          {
            trafo.setTranslation( position->keyValue[positionStep] );
          }
          else
          {
            DP_ASSERT( keys[i] < position->steps[positionStep] );
            float alpha = (float)( keys[i] - position->steps[positionStep-1] )
                        / (float)( position->steps[positionStep] - position->steps[positionStep-1] );
            trafo.setTranslation( lerp( alpha, position->keyValue[positionStep-1], position->keyValue[positionStep] ) );
          }
        }
        else
        {
          trafo.setTranslation( pViewpoint->position );
        }
        liadt->addKey( keys[i], trafo );
      }
    }

    TrafoAnimationSharedPtr tah = TrafoAnimation::create();
    TrafoAnimationLock(tah)->setDescription( litadh );

    m_scene->addCameraAnimation( tah );
  }
#endif

  PerspectiveCameraSharedPtr pc( PerspectiveCamera::create() );
  pc->setName( pViewpoint->getName() );
  pc->setFieldOfView( pViewpoint->fieldOfView );
  pc->setOrientation( interpretSFRotation( pViewpoint->orientation ) );
  pc->setPosition( pViewpoint->position );

  m_scene->addCamera( pc );

  return( dp::sg::core::ObjectSharedPtr::null );
}

bool  WRLLoader::isValidScaling( PositionInterpolatorSharedPtr const& pPositionInterpolator ) const
{
  bool  isValid = ! pPositionInterpolator->keyValue.empty();
  for ( size_t i=0 ; isValid && i<pPositionInterpolator->keyValue.size() ; i++ )
  {
    isValid &= isValidScaling( pPositionInterpolator->keyValue[i] );
  }
  return( isValid );
}

bool  WRLLoader::isValidScaling( const SFVec3f &sfVec3f ) const
{
  return( ( FLT_EPSILON < sfVec3f[0] ) && ( FLT_EPSILON < sfVec3f[1] ) && ( FLT_EPSILON < sfVec3f[2] ) );
}

SceneSharedPtr WRLLoader::load( string const& filename, dp::util::FileFinder const& fileFinder, dp::sg::ui::ViewStateSharedPtr & viewState )
{
  if ( !dp::util::fileExists(filename) )
  {
    throw dp::FileNotFoundException( filename );
  }

  DP_ASSERT( m_textureFiles.empty() );

  // set the locale temporarily to the default "C" to make atof behave predictably
  dp::util::Locale tl("C");

  //  (re-)initialize member variables
  m_eof = false;
  m_smoothTraverser.reset( new dp::sg::algorithm::SmoothTraverser() );

  viewState.reset(); // loading of ViewState currently not supported

  m_fileFinder = fileFinder;
  m_fileFinder.addSearchPath( dp::util::getFilePath( filename ) );

  // run the importer
  try
  {
    m_scene = import( filename );
  }
  catch( ... )
  {
    if ( m_fh )
    {
      fclose( m_fh );
      m_fh = NULL;
    }

    //clean up resources
    m_textureFiles.clear();
    m_smoothTraverser = nullptr;
    m_rootNode.reset();
    m_scene.reset();

    throw;
  }
  if ( ! m_scene && callback() )
  {
    callback()->onFileEmpty( filename );
  }

  SceneSharedPtr scene = m_scene;

  //clean up resources
  m_textureFiles.clear();
  m_smoothTraverser = nullptr;
  m_rootNode.reset();
  m_scene.reset();
  m_fileFinder.clear();

  if ( !scene )
  {
    throw std::runtime_error( std::string("Empty scene loaded") );
  }
  return( scene );
}

bool  WRLLoader::onIncompatibleValues( int value0, int value1, const string &node, const string &field0, const string &field1 ) const
{
  return( callback() ? callback()->onIncompatibleValues( m_lineNumber, node, field0, value0, field1, value1) : true );
}

template<typename T> bool  WRLLoader::onInvalidValue( T value, const string &node, const string &field ) const
{
  return( callback() ? callback()->onInvalidValue( m_lineNumber, node, field, value ) : true );
}

bool  WRLLoader::onEmptyToken( const string &tokenType, const string &token ) const
{
  return( callback() ? callback()->onEmptyToken( m_lineNumber, tokenType, token ) : true );
}

bool  WRLLoader::onFileNotFound( const SFString &url ) const
{
  return( callback() ? callback()->onFileNotFound( url ) : true );
}

bool  WRLLoader::onFilesNotFound( bool found, const MFString &url ) const
{
  if ( !found && callback() )
  {
    return( callback()->onFilesNotFound( url ) );
  }
  return( true );
}

void  WRLLoader::onUnexpectedEndOfFile( bool error ) const
{
  if ( callback() && error )
  {
    callback()->onUnexpectedEndOfFile( m_lineNumber );
  }
}

void  WRLLoader::onUnexpectedToken( const string &expected, const string &token ) const
{
  if ( expected != token && callback() )
  {
    callback()->onUnexpectedToken( m_lineNumber, expected, token );
  }
}

void  WRLLoader::onUnknownToken( const string &tokenType, const string &token ) const
{
  std::ostringstream oss;
  oss << "WRLLoader: Unknown " << tokenType << " <" << token << "> encountered in Line " << m_lineNumber << std::endl;
  throw std::runtime_error( oss.str().c_str() );
}

bool  WRLLoader::onUndefinedToken( const string &tokenType, const string &token ) const
{
  return( callback() ? callback()->onUndefinedToken( m_lineNumber, tokenType, token ) : true );
}

bool  WRLLoader::onUnsupportedToken( const string &tokenType, const string &token ) const
{
  return( callback() ? callback()->onUnsupportedToken( m_lineNumber, tokenType, token ) : true );
}

AnchorSharedPtr WRLLoader::readAnchor( const string &nodeName )
{
  AnchorSharedPtr pAnchor = Anchor::create();
  pAnchor->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "children" )
    {
      readMFNode( pAnchor );
    }
    else if ( token == "description" )
    {
      readSFString( pAnchor->description, getNextToken() );
    }
    else if ( token == "parameter" )
    {
      readMFString( pAnchor->parameter );
    }
    else if ( token == "url" )
    {
      readMFString( pAnchor->url );
    }
    else
    {
      onUnknownToken( "Anchor", token );
    }
    token = getNextToken();
  }

  return( pAnchor );
}

AppearanceSharedPtr WRLLoader::readAppearance( const string &nodeName )
{
  AppearanceSharedPtr pAppearance = Appearance::create();
  pAppearance->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "material" )
    {
      readSFNode( pAppearance, pAppearance->material, getNextToken() );
      if ( pAppearance->material && ! pAppearance->material.isPtrTo<vrml::Material>() )
      {
        onUnsupportedToken( "Appearance.material", pAppearance->material->getType() );
        pAppearance->material.reset();
      }
    }
    else if ( token == "texture" )
    {
      readSFNode( pAppearance, pAppearance->texture, getNextToken() );
      if ( pAppearance->texture && ! pAppearance->texture.isPtrTo<vrml::Texture>() )
      {
        onUnsupportedToken( "Appearance.texture", pAppearance->texture->getType() );
        pAppearance->texture.reset();
      }
    }
    else if ( token == "textureTransform" )
    {
      readSFNode( pAppearance, pAppearance->textureTransform, getNextToken() );
      if ( pAppearance->textureTransform && ! pAppearance->textureTransform.isPtrTo<TextureTransform>() )
      {
        onUnsupportedToken( "Appearance.textureTransform", pAppearance->textureTransform->getType() );
        pAppearance->textureTransform.reset();
      }
    }
    else
    {
      onUnknownToken( "Appearance", token );
    }
    token = getNextToken();
  }

  return( pAppearance );
}

AudioClipSharedPtr WRLLoader::readAudioClip( const string &nodeName )
{
  AudioClipSharedPtr pAudioClip = AudioClip::create();
  pAudioClip->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "description" )
    {
      readSFString( pAudioClip->description, getNextToken() );
    }
    else if ( token == "loop" )
    {
      readSFBool( pAudioClip->loop );
    }
    else if ( token == "pitch" )
    {
      readSFFloat( pAudioClip->pitch, getNextToken() );
    }
    else if ( token == "startTime" )
    {
      readSFTime( pAudioClip->startTime );
    }
    else if ( token == "stopTime" )
    {
      readSFTime( pAudioClip->stopTime );
    }
    else if ( token == "url" )
    {
      readMFString( pAudioClip->url );
    }
    else
    {
      onUnknownToken( "AudioClip", token );
    }
    token = getNextToken();
  }

  return( pAudioClip );
}

BackgroundSharedPtr WRLLoader::readBackground( const string &nodeName )
{
  BackgroundSharedPtr pBackground = Background::create();
  pBackground->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "groundAngle" )
    {
      readMFFloat( pBackground->groundAngle );
    }
    else if ( token == "groundColor" )
    {
      readMFColor( pBackground->groundColor );
    }
    else if ( token == "backUrl" )
    {
      readMFString( pBackground->backUrl );
    }
    else if ( token == "bottomUrl" )
    {
      readMFString( pBackground->bottomUrl );
    }
    else if ( token == "frontUrl" )
    {
      readMFString( pBackground->frontUrl );
    }
    else if ( token == "leftUrl" )
    {
      readMFString( pBackground->leftUrl );
    }
    else if ( token == "rightUrl" )
    {
      readMFString( pBackground->rightUrl );
    }
    else if ( token == "topUrl" )
    {
      readMFString( pBackground->topUrl );
    }
    else if ( token == "skyAngle" )
    {
      readMFFloat( pBackground->skyAngle );
    }
    else if ( token == "skyColor" )
    {
      pBackground->skyColor.clear();
      readMFColor( pBackground->skyColor );
    }
    else
    {
      onUnknownToken( "Background", token );
    }
    token = getNextToken();
  }

  return( pBackground );
}

vrml::BillboardSharedPtr WRLLoader::readBillboard( const string &nodeName )
{
  vrml::BillboardSharedPtr pBillboard = vrml::Billboard::create();
  pBillboard->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "axisOfRotation" )
    {
      readSFVec3f( pBillboard->axisOfRotation, getNextToken() );
    }
    else if ( token == "children" )
    {
      readMFNode( pBillboard );
    }
    else
    {
      onUnknownToken( "Billboard", token );
    }
    token = getNextToken();
  }

  return( pBillboard );
}

BoxSharedPtr WRLLoader::readBox( const string &nodeName )
{
  BoxSharedPtr pBox = Box::create();
  pBox->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "size" )
    {
      readSFVec3f( pBox->size, getNextToken() );
    }
    else
    {
      onUnknownToken( "Box", token );
    }
    token = getNextToken();
  }

  return( pBox );
}

CollisionSharedPtr WRLLoader::readCollision( const string &nodeName )
{
  CollisionSharedPtr pCollision = Collision::create();
  pCollision->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "children" )
    {
      readMFNode( pCollision );
    }
    else if ( token == "collide" )
    {
      readSFBool( pCollision->collide );
    }
    else if ( token == "proxy" )
    {
      readSFNode( pCollision, pCollision->proxy, getNextToken() );
    }
    else
    {
      onUnknownToken( "Collision", token );
    }
    token = getNextToken();
  }

  return( pCollision );
}

ColorSharedPtr WRLLoader::readColor( const string &nodeName )
{
  ColorSharedPtr pColor = Color::create();
  pColor->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "color" )
    {
      readMFColor( pColor->color );
    }
    else
    {
      onUnknownToken( "Color", token );
    }
    token = getNextToken();
  }

  return( pColor );
}

ColorInterpolatorSharedPtr WRLLoader::readColorInterpolator( const string &nodeName )
{
  ColorInterpolatorSharedPtr pColorInterpolator = ColorInterpolator::create();
  pColorInterpolator->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "key" )
    {
      readMFFloat( pColorInterpolator->key );
    }
    else if ( token == "keyValue" )
    {
      readMFColor( pColorInterpolator->keyValue );
    }
    else
    {
      onUnknownToken( "ColorInterpolator", token );
    }
    token = getNextToken();
  }

  DP_ASSERT( ( pColorInterpolator->keyValue.size() % pColorInterpolator->key.size() ) == 0 );

  return( pColorInterpolator );
}

ConeSharedPtr WRLLoader::readCone( const string &nodeName )
{
  ConeSharedPtr pCone = Cone::create();
  pCone->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "bottom" )
    {
      readSFBool( pCone->bottom );
    }
    else if ( token == "bottomRadius" )
    {
      readSFFloat( pCone->bottomRadius, getNextToken() );
    }
    else if ( token == "height" )
    {
      readSFFloat( pCone->height, getNextToken() );
    }
    else if ( token == "side" )
    {
      readSFBool( pCone->side );
    }
    else
    {
      onUnknownToken( "Cone", token );
    }
    token = getNextToken();
  }

  return( pCone );
}

CoordinateSharedPtr WRLLoader::readCoordinate( const string &nodeName )
{
  CoordinateSharedPtr pCoordinate = Coordinate::create();
  pCoordinate->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "point" )
    {
      readMFVec3f( pCoordinate->point );
    }
    else
    {
      onUnknownToken( "Coordinate", token );
    }
    token = getNextToken();
  }

  return( pCoordinate );
}

CoordinateInterpolatorSharedPtr WRLLoader::readCoordinateInterpolator( const string &nodeName )
{
  CoordinateInterpolatorSharedPtr pCoordinateInterpolator = CoordinateInterpolator::create();
  pCoordinateInterpolator->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "key" )
    {
      readMFFloat( pCoordinateInterpolator->key );
    }
    else if ( token == "keyValue" )
    {
      readMFVec3f( pCoordinateInterpolator->keyValue );
    }
    else
    {
      onUnknownToken( "CoordinateInterpolator", token );
    }
    token = getNextToken();
  }

  DP_ASSERT( ( pCoordinateInterpolator->keyValue.size() % pCoordinateInterpolator->key.size() ) == 0 );

  return( pCoordinateInterpolator );
}

CylinderSharedPtr WRLLoader::readCylinder( const string &nodeName )
{
  CylinderSharedPtr pCylinder = Cylinder::create();
  pCylinder->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "bottom" )
    {
      readSFBool( pCylinder->bottom );
    }
    else if ( token == "height" )
    {
      readSFFloat( pCylinder->height, getNextToken() );
    }
    else if ( token == "radius" )
    {
      readSFFloat( pCylinder->radius, getNextToken() );
    }
    else if ( token == "side" )
    {
      readSFBool( pCylinder->side );
    }
    else if ( token == "top" )
    {
      readSFBool( pCylinder->top );
    }
    else
    {
      onUnknownToken( "Cylinder", token );
    }
    token = getNextToken();
  }

  return( pCylinder );
}

CylinderSensorSharedPtr WRLLoader::readCylinderSensor( const string &nodeName )
{
  CylinderSensorSharedPtr pCylinderSensor = CylinderSensor::create();
  pCylinderSensor->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "autoOffset" )
    {
      readSFBool( pCylinderSensor->autoOffset );
    }
    else if ( token == "diskAngle" )
    {
      readSFFloat( pCylinderSensor->diskAngle, getNextToken() );
    }
    else if ( token == "enabled" )
    {
      readSFBool( pCylinderSensor->enabled );
    }
    else if ( token == "maxAngle" )
    {
      readSFFloat( pCylinderSensor->maxAngle, getNextToken() );
    }
    else if ( token == "minAngle" )
    {
      readSFFloat( pCylinderSensor->minAngle, getNextToken() );
    }
    else if ( token == "offset" )
    {
      readSFFloat( pCylinderSensor->offset, getNextToken() );
    }
    else
    {
      onUnknownToken( "CylinderSensor", token );
    }
    token = getNextToken();
  }

  onUnsupportedToken( "VRMLLoader", "CylinderSensor" );
  pCylinderSensor.reset();

  return( pCylinderSensor );
}

DirectionalLightSharedPtr WRLLoader::readDirectionalLight( const string &nodeName )
{
  DirectionalLightSharedPtr pDirectionalLight = DirectionalLight::create();
  pDirectionalLight->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "ambientIntensity" )
    {
      readSFFloat( pDirectionalLight->ambientIntensity, getNextToken() );
    }
    else if ( token == "color" )
    {
      readSFColor( pDirectionalLight->color, getNextToken() );
    }
    else if ( token == "direction" )
    {
      readSFVec3f( pDirectionalLight->direction, getNextToken() );
    }
    else if ( token == "intensity" )
    {
      readSFFloat( pDirectionalLight->intensity, getNextToken() );
    }
    else if ( token == "on" )
    {
      readSFBool( pDirectionalLight->on );
    }
    else
    {
      onUnknownToken( "DirectionalLight", token );
    }
    token = getNextToken();
  }

  return( pDirectionalLight );
}

ElevationGridSharedPtr WRLLoader::readElevationGrid( const string &nodeName )
{
  bool killit = false;

  ElevationGridSharedPtr pElevationGrid = ElevationGrid::create();
  pElevationGrid->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "color" )
    {
      readSFNode( pElevationGrid, pElevationGrid->color, getNextToken() );
    }
    else if ( token == "normal" )
    {
      readSFNode( pElevationGrid, pElevationGrid->normal, getNextToken() );
    }
    else if ( token == "texCoord" )
    {
      readSFNode( pElevationGrid, pElevationGrid->texCoord, getNextToken() );
    }
    else if ( token == "height" )
    {
      readMFFloat( pElevationGrid->height );
    }
    else if ( token == "ccw" )
    {
      readSFBool( pElevationGrid->ccw );
    }
    else if ( token == "colorPerVertex" )
    {
      readSFBool( pElevationGrid->colorPerVertex );
    }
    else if ( token == "creaseAngle" )
    {
      readSFFloat( pElevationGrid->creaseAngle, getNextToken() );
    }
    else if ( token == "normalPerVertex" )
    {
      readSFBool( pElevationGrid->normalPerVertex );
    }
    else if ( token == "solid" )
    {
      readSFBool( pElevationGrid->solid );
    }
    else if ( token == "xDimension" )
    {
      readSFInt32( pElevationGrid->xDimension, getNextToken() );
      if ( pElevationGrid->xDimension <= 1 )
      {
        onInvalidValue( pElevationGrid->xDimension, "ElevationGrid", token );
        killit = true;
      }
    }
    else if ( token == "xSpacing" )
    {
      readSFFloat( pElevationGrid->xSpacing, getNextToken() );
      if ( pElevationGrid->xSpacing <= 0.0f )
      {
        onInvalidValue( pElevationGrid->xSpacing, "ElevationGrid", token );
        killit = true;
      }
    }
    else if ( token == "zDimension" )
    {
      readSFInt32( pElevationGrid->zDimension, getNextToken() );
      if ( pElevationGrid->zDimension <= 1 )
      {
        onInvalidValue( pElevationGrid->zDimension, "ElevationGrid", token );
        killit = true;
      }
    }
    else if ( token == "zSpacing" )
    {
      readSFFloat( pElevationGrid->zSpacing, getNextToken() );
      if ( pElevationGrid->zSpacing <= 0.0f )
      {
        onInvalidValue( pElevationGrid->zSpacing, "ElevationGrid", token );
        killit = true;
      }
    }
    else
    {
      onUnknownToken( "ElevationGrid", token );
    }
    token = getNextToken();
  }

  if ( pElevationGrid->height.size() != ( pElevationGrid->xDimension * pElevationGrid->zDimension ) )
  {
    onIncompatibleValues( (int) pElevationGrid->height.size(), (int) pElevationGrid->xDimension * pElevationGrid->zDimension,
                          "ElevationGrid", "height.size", "xDimension * zDimension" );
    killit = true;
  }

  if ( killit )
  {
    pElevationGrid.reset();
  }

  return( pElevationGrid );
}

void  WRLLoader::readEXTERNPROTO( void )
{
  onUnsupportedToken( "VRMLLoader", "EXTERNPROTO" );
  m_PROTONames.insert( getNextToken() );    //  PrototypeName
  ignoreBlock( "[", "]", getNextToken() );  //  PrototypeDeclaration
  MFString  mfString;
  readMFString( mfString );
}

ExtrusionSharedPtr WRLLoader::readExtrusion( const string &nodeName )
{
  ExtrusionSharedPtr pExtrusion = Extrusion::create();
  pExtrusion->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "beginCap" )
    {
      readSFBool( pExtrusion->beginCap );
    }
    else if ( token == "ccw" )
    {
      readSFBool( pExtrusion->ccw );
    }
    else if ( token == "convex" )
    {
      readSFBool( pExtrusion->convex );
    }
    else if ( token == "creaseAngle" )
    {
      readSFFloat( pExtrusion->creaseAngle, getNextToken() );
    }
    else if ( token == "crossSection" )
    {
      readMFVec2f( pExtrusion->crossSection );
    }
    else if ( token == "endCap" )
    {
      readSFBool( pExtrusion->endCap );
    }
    else if ( token == "orientation" )
    {
      readMFRotation( pExtrusion->orientation );
    }
    else if ( token == "scale" )
    {
      readMFVec2f( pExtrusion->scale );
    }
    else if ( token == "solid" )
    {
      readSFBool( pExtrusion->solid );
    }
    else if ( token == "spine" )
    {
      readMFVec3f( pExtrusion->spine );
    }
    else
    {
      onUnknownToken( "Extrusion", token );
    }
    token = getNextToken();
  }

  onUnsupportedToken( "VRMLLoader", "Extrusion" );
  pExtrusion.reset();

  return( pExtrusion );
}

FogSharedPtr WRLLoader::readFog( const string &nodeName )
{
  FogSharedPtr pFog = Fog::create();
  pFog->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "color" )
    {
      readSFColor( pFog->color, getNextToken() );
    }
    else if ( token == "fogType" )
    {
      readSFString( pFog->fogType, getNextToken() );
    }
    else if ( token == "visibilityRange" )
    {
      readSFFloat( pFog->visibilityRange, getNextToken() );
    }
    else
    {
      onUnknownToken( "Fog", token );
    }
    token = getNextToken();
  }

  onUnsupportedToken( "VRMLLoader", "Fog" );
  pFog.reset();

  return( pFog );
}

FontStyleSharedPtr WRLLoader::readFontStyle( const string &nodeName )
{
  FontStyleSharedPtr pFontStyle = FontStyle::create();
  pFontStyle->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "family" )
    {
      readMFString( pFontStyle->family );
    }
    else if ( token == "horizontal" )
    {
      readSFBool( pFontStyle->horizontal );
    }
    else if ( token == "justify" )
    {
      readMFString( pFontStyle->justify );
    }
    else if ( token == "language" )
    {
      readSFString( pFontStyle->language, getNextToken() );
    }
    else if ( token == "leftToRight" )
    {
      readSFBool( pFontStyle->leftToRight );
    }
    else if ( token == "size" )
    {
      readSFFloat( pFontStyle->size, getNextToken() );
    }
    else if ( token == "spacing" )
    {
      readSFFloat( pFontStyle->spacing, getNextToken() );
    }
    else if ( token == "style" )
    {
      readSFString( pFontStyle->style, getNextToken() );
    }
    else if ( token == "topToBottom" )
    {
      readSFBool( pFontStyle->topToBottom );
    }
    else
    {
      onUnknownToken( "FontStyle", token );
    }
    token = getNextToken();
  }

  return( pFontStyle );
}

vrml::GroupSharedPtr WRLLoader::readGroup( const string &nodeName )
{
  vrml::GroupSharedPtr pGroup = vrml::Group::create();
  pGroup->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "bboxCenter" )
    {
      readSFVec3f( pGroup->bboxCenter, getNextToken() );
    }
    else if ( token == "bboxSize" )
    {
      readSFVec3f( pGroup->bboxSize, getNextToken() );
    }
    else if ( token == "children" )
    {
      readMFNode( pGroup );
    }
    else
    {
      onUnknownToken( "Group", token );
    }
    token = getNextToken();
  }

  return( pGroup );
}


ImageTextureSharedPtr WRLLoader::readImageTexture( const string &nodeName )
{
  ImageTextureSharedPtr pImageTexture = ImageTexture::create();
  pImageTexture->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "url" )
    {
      readMFString( pImageTexture->url );
    }
    else if ( token == "repeatS" )
    {
      readSFBool( pImageTexture->repeatS );
    }
    else if ( token == "repeatT" )
    {
      readSFBool( pImageTexture->repeatT );
    }
    else
    {
      onUnknownToken( "ImageTexture", token );
    }
    token = getNextToken();
  }

  return( pImageTexture );
}

void  WRLLoader::readIndex( vector<SFInt32> &mf )
{
  readMFInt32( mf );
  if ( ( mf.size() == 1 ) && ( mf[0] == -1 ) )
  {
    mf.clear();
  }
}

bool removeCollinearPoint( IndexedFaceSetSharedPtr const& pIndexedFaceSet, unsigned int i0, unsigned int i1, unsigned int i2 )
{
  DP_ASSERT( pIndexedFaceSet->coord.isPtrTo<Coordinate>() );
  CoordinateSharedPtr const& pC = pIndexedFaceSet->coord.staticCast<Coordinate>();
  Vec3f e0 = pC->point[pIndexedFaceSet->coordIndex[i1]] - pC->point[pIndexedFaceSet->coordIndex[i0]];
  Vec3f e1 = pC->point[pIndexedFaceSet->coordIndex[i2]] - pC->point[pIndexedFaceSet->coordIndex[i1]];
  if ( length( e0 ^ e1 ) <= FLT_EPSILON )
  {
    bool remove = true;
    if ( pIndexedFaceSet->color && pIndexedFaceSet->colorPerVertex && ! pIndexedFaceSet->colorIndex.empty() )
    {
      DP_ASSERT( pIndexedFaceSet->color.isPtrTo<Color>() );
      ColorSharedPtr const& pColor = pIndexedFaceSet->color.staticCast<Color>();
      Vec3f dc0 = pColor->color[pIndexedFaceSet->colorIndex[i1]] - pColor->color[pIndexedFaceSet->colorIndex[i0]];
      Vec3f dc1 = pColor->color[pIndexedFaceSet->colorIndex[i2]] - pColor->color[pIndexedFaceSet->colorIndex[i1]];
      remove = ( length( dc0 / length( e0 ) - dc1 / length( e1 ) ) < FLT_EPSILON );
    }
    if ( remove && pIndexedFaceSet->normal && pIndexedFaceSet->normalPerVertex && ! pIndexedFaceSet->normalIndex.empty() )
    {
      DP_ASSERT( pIndexedFaceSet->normal.isPtrTo<Normal>() );
      NormalSharedPtr const& pNormal = pIndexedFaceSet->normal.staticCast<Normal>();
      Vec3f nxn0 = pNormal->vector[pIndexedFaceSet->normalIndex[i0]] ^ pNormal->vector[pIndexedFaceSet->normalIndex[i1]];
      Vec3f nxn1 = pNormal->vector[pIndexedFaceSet->normalIndex[i1]] ^ pNormal->vector[pIndexedFaceSet->normalIndex[i2]];
      float c0 = pNormal->vector[pIndexedFaceSet->normalIndex[i0]] * pNormal->vector[pIndexedFaceSet->normalIndex[i1]];
      float c1 = pNormal->vector[pIndexedFaceSet->normalIndex[i1]] * pNormal->vector[pIndexedFaceSet->normalIndex[i2]];
      remove =    areCollinear( nxn0, nxn1 )
              &&  ( abs( acos( c0 ) / length( e0 ) - acos( c1 ) / length( e1 ) ) < FLT_EPSILON );
    }
    if ( remove && pIndexedFaceSet->texCoord && ! pIndexedFaceSet->texCoordIndex.empty() )
    {
      DP_ASSERT( pIndexedFaceSet->texCoord.isPtrTo<TextureCoordinate>() );
      TextureCoordinateSharedPtr const& pTextureCoordinate = pIndexedFaceSet->texCoord.staticCast<TextureCoordinate>();
      Vec2f dt0 = pTextureCoordinate->point[pIndexedFaceSet->texCoordIndex[i1]] - pTextureCoordinate->point[pIndexedFaceSet->texCoordIndex[i0]];
      Vec2f dt1 = pTextureCoordinate->point[pIndexedFaceSet->texCoordIndex[i2]] - pTextureCoordinate->point[pIndexedFaceSet->texCoordIndex[i1]];
      remove = ( length( dt0 / length( e0 ) - dt1 / length( e1 ) ) < FLT_EPSILON );
    }

    // i0-i1-i2 are collinear -> remove i1
    if ( remove )
    {
      pIndexedFaceSet->coordIndex.erase( pIndexedFaceSet->coordIndex.begin() + i1 );
      if ( pIndexedFaceSet->color && pIndexedFaceSet->colorPerVertex && ! pIndexedFaceSet->colorIndex.empty() )
      {
        pIndexedFaceSet->colorIndex.erase( pIndexedFaceSet->colorIndex.begin() + i1 );
      }
      if ( pIndexedFaceSet->normal && pIndexedFaceSet->normalPerVertex && ! pIndexedFaceSet->normalIndex.empty() )
      {
        pIndexedFaceSet->normalIndex.erase( pIndexedFaceSet->normalIndex.begin() + i1 );
      }
      if ( pIndexedFaceSet->texCoord && ! pIndexedFaceSet->texCoordIndex.empty() )
      {
        pIndexedFaceSet->texCoordIndex.erase( pIndexedFaceSet->texCoordIndex.begin() + i1 );
      }
      return( true );
    }
  }
  return( false );
}

bool removeRedundantPoint( IndexedFaceSetSharedPtr const& pIndexedFaceSet, unsigned int i0, unsigned int i1 )
{
  DP_ASSERT( pIndexedFaceSet->coord.isPtrTo<Coordinate>() );
  CoordinateSharedPtr const& pC = pIndexedFaceSet->coord.staticCast<Coordinate>();
  if (    ( pIndexedFaceSet->coordIndex[i0] == pIndexedFaceSet->coordIndex[i1] )
      ||  ( length( pC->point[pIndexedFaceSet->coordIndex[i1]] - pC->point[pIndexedFaceSet->coordIndex[i0]] ) < FLT_EPSILON ) )
  {
    bool remove = true;
    if ( pIndexedFaceSet->color && pIndexedFaceSet->colorPerVertex && ! pIndexedFaceSet->colorIndex.empty() )
    {
      DP_ASSERT( pIndexedFaceSet->color.isPtrTo<Color>() );
      ColorSharedPtr const& pColor = pIndexedFaceSet->color.staticCast<Color>();
      remove =    ( pIndexedFaceSet->colorIndex[i0] == pIndexedFaceSet->colorIndex[i1] )
               || ( length( pColor->color[pIndexedFaceSet->colorIndex[i1]] - pColor->color[pIndexedFaceSet->colorIndex[i0]] ) < FLT_EPSILON );
      DP_ASSERT( remove );    // never encountered this
    }
    if ( remove && pIndexedFaceSet->normal && pIndexedFaceSet->normalPerVertex && ! pIndexedFaceSet->normalIndex.empty() )
    {
      DP_ASSERT( pIndexedFaceSet->normal.isPtrTo<Normal>() );
      NormalSharedPtr const& pNormal = pIndexedFaceSet->normal.staticCast<Normal>();
      remove =    ( pIndexedFaceSet->normalIndex[i0] == pIndexedFaceSet->normalIndex[i1] )
               || ( length( pNormal->vector[pIndexedFaceSet->normalIndex[i1]] - pNormal->vector[pIndexedFaceSet->normalIndex[i0]] ) < FLT_EPSILON );
    }
    if ( remove && pIndexedFaceSet->texCoord && ! pIndexedFaceSet->texCoordIndex.empty() )
    {
      DP_ASSERT( pIndexedFaceSet->texCoord.isPtrTo<TextureCoordinate>() );
      TextureCoordinateSharedPtr const& pTextureCoordinate = pIndexedFaceSet->texCoord.staticCast<TextureCoordinate>();
      remove =    ( pIndexedFaceSet->texCoordIndex[i0] == pIndexedFaceSet->texCoordIndex[i1] )
               || ( length( pTextureCoordinate->point[pIndexedFaceSet->texCoordIndex[i1]] - pTextureCoordinate->point[pIndexedFaceSet->texCoordIndex[i0]] ) < FLT_EPSILON );
      DP_ASSERT( remove );    // never encountered this
    }

    if ( remove )
    {
      // two times the same index or the same position -> remove one of them
      pIndexedFaceSet->coordIndex.erase( pIndexedFaceSet->coordIndex.begin() + i1 );
      if ( pIndexedFaceSet->color && pIndexedFaceSet->colorPerVertex && ! pIndexedFaceSet->colorIndex.empty() )
      {
        pIndexedFaceSet->colorIndex.erase( pIndexedFaceSet->colorIndex.begin() + i1 );
      }
      if ( pIndexedFaceSet->normal && pIndexedFaceSet->normalPerVertex && ! pIndexedFaceSet->normalIndex.empty() )
      {
        pIndexedFaceSet->normalIndex.erase( pIndexedFaceSet->normalIndex.begin() + i1 );
      }
      if ( pIndexedFaceSet->texCoord && ! pIndexedFaceSet->texCoordIndex.empty() )
      {
        pIndexedFaceSet->texCoordIndex.erase( pIndexedFaceSet->texCoordIndex.begin() + i1 );
      }
      return( true );
    }
  }
  return( false );
}

void removeInvalidFace( IndexedFaceSetSharedPtr const& pIndexedFaceSet, size_t i, size_t j, unsigned int numberOfFaces )
{
  pIndexedFaceSet->coordIndex.erase( pIndexedFaceSet->coordIndex.begin() + i
                                   , pIndexedFaceSet->coordIndex.begin() + i + j + 1 );
  if ( ! pIndexedFaceSet->colorIndex.empty() )
  {
    if ( pIndexedFaceSet->colorPerVertex )
    {
      pIndexedFaceSet->colorIndex.erase( pIndexedFaceSet->colorIndex.begin() + i
                                       , pIndexedFaceSet->colorIndex.begin() + i + j + 1 );
    }
    else
    {
      DP_ASSERT( numberOfFaces < pIndexedFaceSet->colorIndex.size() );
      pIndexedFaceSet->colorIndex.erase( pIndexedFaceSet->colorIndex.begin() + numberOfFaces );
    }
  }
  if ( ! pIndexedFaceSet->normalIndex.empty() )
  {
    if ( pIndexedFaceSet->normalPerVertex )
    {
      pIndexedFaceSet->normalIndex.erase( pIndexedFaceSet->normalIndex.begin() + i
                                        , pIndexedFaceSet->normalIndex.begin() + i + j + 1 );
    }
    else
    {
      DP_ASSERT( numberOfFaces < pIndexedFaceSet->normalIndex.size() );
      pIndexedFaceSet->normalIndex.erase( pIndexedFaceSet->normalIndex.begin() + numberOfFaces );
    }
  }
  if ( ! pIndexedFaceSet->texCoordIndex.empty() )
  {
    pIndexedFaceSet->texCoordIndex.erase( pIndexedFaceSet->texCoordIndex.begin() + i
                                        , pIndexedFaceSet->texCoordIndex.begin() + i + j + 1 );
  }
}

IndexedFaceSetSharedPtr WRLLoader::readIndexedFaceSet( const string &nodeName )
{
  IndexedFaceSetSharedPtr pIndexedFaceSet = IndexedFaceSet::create();
  pIndexedFaceSet->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "color" )
    {
      readSFNode( pIndexedFaceSet, pIndexedFaceSet->color, getNextToken() );
      if ( pIndexedFaceSet->color )
      {
        if ( !pIndexedFaceSet->color.isPtrTo<Color>() )
        {
          onUnsupportedToken( "IndexedFaceSet.color", pIndexedFaceSet->color->getType() );
          pIndexedFaceSet->color.reset();
        }
        else if ( pIndexedFaceSet->color.staticCast<Color>()->color.empty() )
        {
          onEmptyToken( "IndexedFaceSet", "color" );
          pIndexedFaceSet->color.reset();
        }
      }
    }
    else if ( token == "coord" )
    {
      readSFNode( pIndexedFaceSet, pIndexedFaceSet->coord, getNextToken() );
      if ( pIndexedFaceSet->coord )
      {
        if ( ! pIndexedFaceSet->coord.isPtrTo<Coordinate>() )
        {
          onUnsupportedToken( "IndexedFaceSet.coord", pIndexedFaceSet->coord->getType() );
          pIndexedFaceSet->coord.reset();
        }
        else if ( pIndexedFaceSet->coord.staticCast<Coordinate>()->point.empty() )
        {
          onEmptyToken( "IndexedFaceSet", "coord" );
          pIndexedFaceSet->coord.reset();
        }
      }
    }
    else if ( token == "normal" )
    {
      readSFNode( pIndexedFaceSet, pIndexedFaceSet->normal, getNextToken() );
      if ( pIndexedFaceSet->normal )
      {
        if ( !pIndexedFaceSet->normal.isPtrTo<Normal>() )
        {
          onUnsupportedToken( "IndexedFaceSet.normal", pIndexedFaceSet->normal->getType() );
          pIndexedFaceSet->normal.reset();
        }
        else if ( pIndexedFaceSet->normal.staticCast<Normal>()->vector.empty() )
        {
          onEmptyToken( "IndexedFaceSet", "normal" );
          pIndexedFaceSet->normal.reset();
        }
      }
    }
    else if ( token == "texCoord" )
    {
      readSFNode( pIndexedFaceSet, pIndexedFaceSet->texCoord, getNextToken() );
      if ( pIndexedFaceSet->texCoord )
      {
        if ( ! pIndexedFaceSet->texCoord.isPtrTo<TextureCoordinate>() )
        {
          onUnsupportedToken( "IndexedFaceSet.texCoord", pIndexedFaceSet->texCoord->getType() );
          pIndexedFaceSet->texCoord.reset();
        }
        else if ( pIndexedFaceSet->texCoord.staticCast<TextureCoordinate>()->point.empty() )
        {
          onEmptyToken( "IndexedFaceSet", "texCoord" );
          pIndexedFaceSet->texCoord.reset();
        }
      }
    }
    else if ( token == "ccw" )
    {
      readSFBool( pIndexedFaceSet->ccw );
    }
    else if ( token == "colorIndex" )
    {
      readIndex( pIndexedFaceSet->colorIndex );
    }
    else if ( token == "colorPerVertex" )
    {
      readSFBool( pIndexedFaceSet->colorPerVertex );
    }
    else if ( token == "convex" )
    {
      readSFBool( pIndexedFaceSet->convex );
    }
    else if ( token == "coordIndex" )
    {
      readIndex( pIndexedFaceSet->coordIndex );
    }
    else if ( token == "creaseAngle" )
    {
      readSFFloat( pIndexedFaceSet->creaseAngle, getNextToken() );
    }
    else if ( token == "normalIndex" )
    {
      readIndex( pIndexedFaceSet->normalIndex );
    }
    else if ( token == "normalPerVertex" )
    {
      readSFBool( pIndexedFaceSet->normalPerVertex );
    }
    else if ( token == "solid" )
    {
      readSFBool( pIndexedFaceSet->solid );
    }
    else if ( token == "texCoordIndex" )
    {
      readIndex( pIndexedFaceSet->texCoordIndex );
    }
    else
    {
      onUnknownToken( "IndexedFaceSet", token );
    }
    token = getNextToken();
  }

  if ( pIndexedFaceSet->coord && ! pIndexedFaceSet->coordIndex.empty() )
  {
    //  if there's no -1 at the end, add one
    if ( pIndexedFaceSet->coordIndex[pIndexedFaceSet->coordIndex.size()-1] != -1 )
    {
      pIndexedFaceSet->coordIndex.push_back( -1 );
    }

    // count the number of faces
    unsigned int numberOfFaces = 0;
    int maxIndex = -1;
    for ( size_t i=0 ; i<pIndexedFaceSet->coordIndex.size() ; i++ )
    {
      if ( pIndexedFaceSet->coordIndex[i] == -1 )
      {
        numberOfFaces++;
      }
      else if ( maxIndex < pIndexedFaceSet->coordIndex[i] )
      {
        maxIndex = pIndexedFaceSet->coordIndex[i];
      }
    }

    // make sure color information is correct
    if ( pIndexedFaceSet->color )
    {
      //  if there's no -1 at the end, add one
      if (    pIndexedFaceSet->colorPerVertex && ! pIndexedFaceSet->colorIndex.empty()
          &&  pIndexedFaceSet->colorIndex[pIndexedFaceSet->colorIndex.size()-1] != -1 )
      {
        pIndexedFaceSet->colorIndex.push_back( -1 );
      }

      if ( pIndexedFaceSet->colorPerVertex )
      {
        // first test on non-empty colorIndex
        if ( ! pIndexedFaceSet->colorIndex.empty() )
        {
          if ( pIndexedFaceSet->colorIndex.size() < pIndexedFaceSet->coordIndex.size() )
          {
            DP_ASSERT( pIndexedFaceSet->coordIndex.size() <= INT_MAX );
            DP_ASSERT( pIndexedFaceSet->colorIndex.size() <= INT_MAX );
            onIncompatibleValues( static_cast<int>(pIndexedFaceSet->coordIndex.size())
                                , static_cast<int>(pIndexedFaceSet->colorIndex.size())
                                , "IndexedFaceSet", "coordIndex.size", "colorIndex.size" );
            pIndexedFaceSet->colorIndex.clear();
          }
          else
          {
            for ( size_t i=0 ; i<pIndexedFaceSet->coordIndex.size() ; i++ )
            {
              if ( ( pIndexedFaceSet->coordIndex[i] < 0 ) ^ ( pIndexedFaceSet->colorIndex[i] < 0 ) )
              {
                onIncompatibleValues( pIndexedFaceSet->coordIndex[i], pIndexedFaceSet->colorIndex[i]
                                    , "IndexedFaceSet", "coordIndex", "colorIndex" );
                pIndexedFaceSet->colorIndex.clear();
                break;
              }
            }
          }
        }
        // retest colorIndex on emptiness: might be cleared above
        if ( pIndexedFaceSet->colorIndex.empty() )
        {
          DP_ASSERT( pIndexedFaceSet->color.staticCast<Color>()->color.size() <= INT_MAX );
          int maxColorIndex = dp::checked_cast<int>(pIndexedFaceSet->color.staticCast<Color>()->color.size());
          if ( maxColorIndex <= maxIndex )
          {
            onIncompatibleValues( maxIndex, maxColorIndex, "IndexedFaceSet", "coordIndex.size", "colors.max" );
            pIndexedFaceSet->color.reset();
          }
        }
      }
      else
      {
        // first test on non-empty colorIndex
        if ( ! pIndexedFaceSet->colorIndex.empty() )
        {
          if ( pIndexedFaceSet->colorIndex.size() < numberOfFaces )
          {
            DP_ASSERT( pIndexedFaceSet->colorIndex.size() <= INT_MAX );
            onIncompatibleValues( numberOfFaces, static_cast<int>(pIndexedFaceSet->colorIndex.size())
                                , "IndexedFaceSet", "faces.size", "colorIndex.size" );
            pIndexedFaceSet->colorIndex.clear();
          }
          else
          {
            for ( unsigned int i=0 ; i < numberOfFaces ; i++ )
            {
              if ( pIndexedFaceSet->colorIndex[i] < 0 )
              {
                onInvalidValue( pIndexedFaceSet->colorIndex[i], "IndexedFaceSet", "colorIndex" );
                pIndexedFaceSet->colorIndex.clear();
                break;
              }
            }
          }
        }
        // retest colorIndex on emptiness: might be cleared above
        if ( pIndexedFaceSet->colorIndex.empty() )
        {
          if ( pIndexedFaceSet->color.staticCast<Color>()->color.size() < numberOfFaces )
          {
            onIncompatibleValues( numberOfFaces
                                , dp::checked_cast<int>(pIndexedFaceSet->color.staticCast<Color>()->color.size())
                                , "IndexedFaceSet", "faces.size", "colors.size" );
            pIndexedFaceSet->color.reset();
          }
        }
      }
    }

    // make sure normal information is correct
    if ( pIndexedFaceSet->normal )
    {
      //  if there's no -1 at the end, add one
      if (    pIndexedFaceSet->normalPerVertex && ! pIndexedFaceSet->normalIndex.empty()
          &&  pIndexedFaceSet->normalIndex[pIndexedFaceSet->normalIndex.size()-1] != -1 )
      {
        pIndexedFaceSet->normalIndex.push_back( -1 );
      }

      if ( pIndexedFaceSet->normalPerVertex )
      {
        // first test on non-empty normalIndex
        if ( ! pIndexedFaceSet->normalIndex.empty() )
        {
          if ( pIndexedFaceSet->normalIndex.size() < pIndexedFaceSet->coordIndex.size() )
          {
            DP_ASSERT( pIndexedFaceSet->coordIndex.size() <= INT_MAX );
            DP_ASSERT( pIndexedFaceSet->normalIndex.size() <= INT_MAX );
            onIncompatibleValues( static_cast<int>(pIndexedFaceSet->coordIndex.size())
                                , static_cast<int>(pIndexedFaceSet->normalIndex.size())
                                , "IndexedFaceSet", "coordIndex.size", "normalIndex.size" );
            pIndexedFaceSet->normalIndex.clear();
          }
          else
          {
            for ( size_t i=0 ; i<pIndexedFaceSet->coordIndex.size() ; i++ )
            {
              if ( ( pIndexedFaceSet->coordIndex[i] < 0 ) ^ ( pIndexedFaceSet->normalIndex[i] < 0 ) )
              {
                onIncompatibleValues( pIndexedFaceSet->coordIndex[i], pIndexedFaceSet->normalIndex[i]
                                    , "IndexedFaceSet", "coordIndex", "normalIndex" );
                pIndexedFaceSet->normalIndex.clear();
                break;
              }
            }
          }
        }
        // retest normalIndex on emptiness: might be cleared above
        if ( pIndexedFaceSet->normalIndex.empty() )
        {
          int maxNormalIndex = dp::checked_cast<int>(pIndexedFaceSet->normal.staticCast<Normal>()->vector.size());
          if ( maxNormalIndex <= maxIndex )
          {
            onIncompatibleValues( maxIndex, maxNormalIndex, "IndexedFaceSet", "coordIndex.max", "normals.size" );
            pIndexedFaceSet->normal.reset();
          }
        }
      }
      else
      {
        // first test on non-empty normalIndex
        if ( ! pIndexedFaceSet->normalIndex.empty() )
        {
          if ( pIndexedFaceSet->normalIndex.size() < numberOfFaces )
          {
            DP_ASSERT( pIndexedFaceSet->normalIndex.size() <= INT_MAX );
            onIncompatibleValues( numberOfFaces, static_cast<int>(pIndexedFaceSet->normalIndex.size())
                                , "IndexedFaceSet", "faces.size", "normalIndex.size" );
            pIndexedFaceSet->normalIndex.clear();
          }
          else
          {
            for ( unsigned int i=0 ; i < numberOfFaces ; i++ )
            {
              if ( pIndexedFaceSet->normalIndex[i] < 0 )
              {
                onInvalidValue( pIndexedFaceSet->normalIndex[i], "IndexedFaceSet", "normalIndex" );
                pIndexedFaceSet->normalIndex.clear();
                break;
              }
            }
          }
        }
        // retest normalIndex on emptiness: might be cleared above
        if ( pIndexedFaceSet->normalIndex.empty() )
        {
          if ( pIndexedFaceSet->normal.staticCast<Normal>()->vector.size() < numberOfFaces )
          {
            onIncompatibleValues( numberOfFaces
                                , dp::checked_cast<int>(pIndexedFaceSet->normal.staticCast<Normal>()->vector.size())
                                , "IndexedFaceSet", "faces.size", "normals.size" );
            pIndexedFaceSet->normal.reset();
          }
        }
      }
    }

    if ( pIndexedFaceSet->texCoord )
    {
      // first test on non-empty texCoordIndex
      if ( ! pIndexedFaceSet->texCoordIndex.empty() )
      {
        //  if there's no -1 at the end, add one
        if ( pIndexedFaceSet->texCoordIndex[pIndexedFaceSet->texCoordIndex.size()-1] != -1 )
        {
          pIndexedFaceSet->texCoordIndex.push_back( -1 );
        }
        if ( pIndexedFaceSet->texCoordIndex.size() < pIndexedFaceSet->coordIndex.size() )
        {
          DP_ASSERT( pIndexedFaceSet->coordIndex.size() <= INT_MAX );
          DP_ASSERT( pIndexedFaceSet->texCoordIndex.size() <= INT_MAX );
          onIncompatibleValues( static_cast<int>(pIndexedFaceSet->coordIndex.size())
                              , static_cast<int>(pIndexedFaceSet->texCoordIndex.size())
                              , "IndexedFaceSet", "coordIndex.size", "texCoordIndex.size" );
          pIndexedFaceSet->texCoordIndex.clear();
        }
        else
        {
          for ( size_t i=0 ; i<pIndexedFaceSet->coordIndex.size() ; i++ )
          {
            if ( ( pIndexedFaceSet->coordIndex[i] < 0 ) ^ ( pIndexedFaceSet->texCoordIndex[i] < 0 ) )
            {
              onIncompatibleValues( pIndexedFaceSet->coordIndex[i], pIndexedFaceSet->texCoordIndex[i]
                                  , "IndexedFaceSet", "coordIndex", "texCoordIndex" );
              pIndexedFaceSet->texCoordIndex.clear();
              break;
            }
          }
        }
      }
      // retest texCoordIndex on emptiness: might be cleared above
      if ( pIndexedFaceSet->texCoordIndex.empty() )
      {
        int maxTexCoordIndex = dp::checked_cast<int>(pIndexedFaceSet->texCoord.staticCast<TextureCoordinate>()->point.size());
        if ( maxTexCoordIndex <= maxIndex )
        {
          onIncompatibleValues( maxIndex, maxTexCoordIndex, "IndexedFaceSet", "coordIndex.max", "texCoord.size" );
          pIndexedFaceSet->texCoord.reset();
        }
      }
    }

    //  filter invalid indices
    int numberOfPoints = dp::checked_cast<int>(pIndexedFaceSet->coord.staticCast<Coordinate>()->point.size());
    int numberOfColors = pIndexedFaceSet->color ? dp::checked_cast<int>(pIndexedFaceSet->color.staticCast<Color>()->color.size()) : 0;
    int numberOfNormals = pIndexedFaceSet->normal ? dp::checked_cast<int>(pIndexedFaceSet->normal.staticCast<Normal>()->vector.size()) : 0;
    int numberOfTexCoords = pIndexedFaceSet->texCoord ? dp::checked_cast<int>(pIndexedFaceSet->texCoord.staticCast<TextureCoordinate>()->point.size()) : 0;
    for ( unsigned int i=0 ; i<pIndexedFaceSet->coordIndex.size() ; )
    {
      if ( pIndexedFaceSet->coordIndex[i] != -1 )
      {
        bool removeIndex = false;
        if ( numberOfPoints <= pIndexedFaceSet->coordIndex[i] )
        {
          onIncompatibleValues( pIndexedFaceSet->coordIndex[i], numberOfPoints, "IndexedFaceSet"
                              , "max( coordIndex )", "coord.size" );
          removeIndex = true;
        }
        if ( pIndexedFaceSet->colorPerVertex && ! pIndexedFaceSet->colorIndex.empty() )
        {
          DP_ASSERT( pIndexedFaceSet->colorIndex[i] != -1 );
          if ( numberOfColors <= pIndexedFaceSet->colorIndex[i] )
          {
            onIncompatibleValues( pIndexedFaceSet->colorIndex[i], numberOfColors, "IndexedFaceSet"
                                , "max( colorIndex )", "color.size" );
            removeIndex = true;
          }
        }
        if ( pIndexedFaceSet->normalPerVertex && ! pIndexedFaceSet->normalIndex.empty() )
        {
          DP_ASSERT( pIndexedFaceSet->normalIndex[i] != -1 );
          if ( numberOfNormals <= pIndexedFaceSet->normalIndex[i] )
          {
            onIncompatibleValues( pIndexedFaceSet->normalIndex[i], numberOfNormals, "IndexedFaceSet"
                                , "max( normalIndex )", "normal.size" );
            removeIndex = true;
          }
        }
        if ( ! pIndexedFaceSet->texCoordIndex.empty() )
        {
          DP_ASSERT( pIndexedFaceSet->texCoordIndex[i] != -1 );
          if ( numberOfTexCoords <= pIndexedFaceSet->texCoordIndex[i] )
          {
            onIncompatibleValues( pIndexedFaceSet->texCoordIndex[i], numberOfTexCoords, "IndexedFaceSet"
                                , "max( texCoordIndex )", "texCoord.size" );
            removeIndex = true;
          }
        }
        if ( removeIndex )
        {
          pIndexedFaceSet->coordIndex.erase( pIndexedFaceSet->coordIndex.begin() + i );
          if ( pIndexedFaceSet->colorPerVertex && ! pIndexedFaceSet->colorIndex.empty() )
          {
            pIndexedFaceSet->colorIndex.erase( pIndexedFaceSet->colorIndex.begin() + i );
          }
          if ( pIndexedFaceSet->normalPerVertex && ! pIndexedFaceSet->normalIndex.empty() )
          {
            pIndexedFaceSet->normalIndex.erase( pIndexedFaceSet->normalIndex.begin() + i );
          }
          if ( ! pIndexedFaceSet->texCoordIndex.empty() )
          {
            pIndexedFaceSet->texCoordIndex.erase( pIndexedFaceSet->texCoordIndex.begin() + i );
          }
        }
        else
        {
          i++;
        }
      }
      else
      {
        // assert that all other index arrays also have a -1 here
        DP_ASSERT(    ! pIndexedFaceSet->colorPerVertex
                    ||  pIndexedFaceSet->colorIndex.empty()
                    ||  ( pIndexedFaceSet->colorIndex[i] == -1 ) );
        DP_ASSERT(    ! pIndexedFaceSet->normalPerVertex
                    ||  pIndexedFaceSet->normalIndex.empty()
                    ||  ( pIndexedFaceSet->normalIndex[i] == -1 ) );
        DP_ASSERT(    pIndexedFaceSet->texCoordIndex.empty()
                    ||  ( pIndexedFaceSet->texCoordIndex[i] == -1 ) );
        i++;
      }
    }
    DP_ASSERT( !pIndexedFaceSet->coordIndex.empty() );

    //  filter invalid faces
    numberOfFaces = 0;
    for ( size_t i=0 ; i<pIndexedFaceSet->coordIndex.size() ; )
    {
      bool removeFace = false;
      // scan for next -1
      unsigned int j;
      for ( j=0 ; pIndexedFaceSet->coordIndex[i+j] != -1 ; j++ )
        ;

      if ( j < 3 )
      {
        onUnsupportedToken( "IndexedFaceSet", "n-gonal coordIndex with less than 3 coords" );
        removeFace = true;
      }
      if (    ! pIndexedFaceSet->colorPerVertex
          &&  ! pIndexedFaceSet->colorIndex.empty()
          &&  ( numberOfColors <= pIndexedFaceSet->colorIndex[numberOfFaces] ) )
      {
        onIncompatibleValues( pIndexedFaceSet->colorIndex[i], numberOfColors, "IndexedFaceSet"
                            , "max( colorIndex )", "color.size" );
        removeFace = true;
      }
      if (    ! pIndexedFaceSet->normalPerVertex
          &&  ! pIndexedFaceSet->normalIndex.empty()
          &&  ( numberOfNormals <= pIndexedFaceSet->normalIndex[numberOfFaces] ) )
      {
        onIncompatibleValues( pIndexedFaceSet->normalIndex[i], numberOfNormals, "IndexedFaceSet"
                            , "max( normalIndex )", "normal.size" );
        removeFace = true;
      }
      if (    ! pIndexedFaceSet->texCoordIndex.empty()
          &&  ( numberOfTexCoords <= pIndexedFaceSet->texCoordIndex[numberOfFaces] ) )
      {
        onIncompatibleValues( pIndexedFaceSet->texCoordIndex[i], numberOfTexCoords, "IndexedFaceSet"
                            , "max( texCoordIndex )", "texCoord.size" );
        removeFace = true;
      }
      if ( removeFace )
      {
        removeInvalidFace( pIndexedFaceSet, i, j, numberOfFaces );
      }
      else
      {
        i += j + 1;
        numberOfFaces++;
      }
    }

    //  some clean ups with invalid faces
    DP_ASSERT( pIndexedFaceSet->coordIndex.size() <= UINT_MAX );
    DP_ASSERT( ! ( pIndexedFaceSet->color && pIndexedFaceSet->colorPerVertex && ! pIndexedFaceSet->colorIndex.empty() ) || pIndexedFaceSet->colorIndex[0] != -1 );
    DP_ASSERT( ! ( pIndexedFaceSet->normal && pIndexedFaceSet->normalPerVertex && ! pIndexedFaceSet->normalIndex.empty() ) || pIndexedFaceSet->normalIndex[0] != -1 );
    DP_ASSERT( ! ( pIndexedFaceSet->texCoord && ! pIndexedFaceSet->texCoordIndex.empty() ) || pIndexedFaceSet->texCoordIndex[0] != -1 );
    bool removed = false;
    for ( unsigned int i0=0, i=1 ; i<pIndexedFaceSet->coordIndex.size() ; i++ )
    {
      DP_ASSERT( pIndexedFaceSet->coordIndex[i-1] != -1 );
      if ( pIndexedFaceSet->coordIndex[i] == -1 )
      {
        DP_ASSERT( ! ( pIndexedFaceSet->color && pIndexedFaceSet->colorPerVertex && ! pIndexedFaceSet->colorIndex.empty() ) || pIndexedFaceSet->colorIndex[i] == -1 );
        DP_ASSERT( ! ( pIndexedFaceSet->normal && pIndexedFaceSet->normalPerVertex && ! pIndexedFaceSet->normalIndex.empty() ) || pIndexedFaceSet->normalIndex[i] == -1 );
        DP_ASSERT( ! ( pIndexedFaceSet->texCoord && ! pIndexedFaceSet->texCoordIndex.empty() ) || pIndexedFaceSet->texCoordIndex[i] == -1 );

        // end of face -> check against closed loop
        if ( ( i0 < i-1 ) && (    removeRedundantPoint( pIndexedFaceSet, i0, i-1 )
                              ||  ( ( i0 < i-2 ) && (     removeCollinearPoint( pIndexedFaceSet, i-2, i-1, i0 )
                                                      ||  removeCollinearPoint( pIndexedFaceSet, i-1, i0, i0+1 ) ) ) ) )
        {
          onUnsupportedToken( "IndexedFaceSet", "redundant point: removed" );
          // a redundant last point has been removed -> i now is the start of the next face, no additional advance of i
          i0 = i;
          removed = true;
        }
        else
        {
          // no redundant point has been removed -> i+1 now is the start of the next face, advance i one more
          i0 = i+1;
          i++;
        }

#if !defined( NDEBUG )
        if ( i0 < pIndexedFaceSet->coordIndex.size() )
        {
          DP_ASSERT( ! ( pIndexedFaceSet->color && pIndexedFaceSet->colorPerVertex && ! pIndexedFaceSet->colorIndex.empty() ) || pIndexedFaceSet->colorIndex[i0] != -1 );
          DP_ASSERT( ! ( pIndexedFaceSet->normal && pIndexedFaceSet->normalPerVertex && ! pIndexedFaceSet->normalIndex.empty() ) || pIndexedFaceSet->normalIndex[i0] != -1 );
          DP_ASSERT( ! ( pIndexedFaceSet->texCoord && ! pIndexedFaceSet->texCoordIndex.empty() ) || pIndexedFaceSet->texCoordIndex[i0] != -1 );
        }
#endif
      }
      else
      {
        DP_ASSERT( ! ( pIndexedFaceSet->color && pIndexedFaceSet->colorPerVertex && ! pIndexedFaceSet->colorIndex.empty() ) || pIndexedFaceSet->colorIndex[i] != -1 );
        DP_ASSERT( ! ( pIndexedFaceSet->normal && pIndexedFaceSet->normalPerVertex && ! pIndexedFaceSet->normalIndex.empty() ) || pIndexedFaceSet->normalIndex[i] != -1 );
        DP_ASSERT( ! ( pIndexedFaceSet->texCoord && ! pIndexedFaceSet->texCoordIndex.empty() ) || pIndexedFaceSet->texCoordIndex[i] != -1 );
        if (    removeRedundantPoint( pIndexedFaceSet, i-1, i )
            ||  ( ( i0 < i-1 ) && removeCollinearPoint( pIndexedFaceSet, i-2, i-1, i ) ) )
        {
          onUnsupportedToken( "IndexedFaceSet", "redundant point: removed" );
          i--;    // removed a redundant point -> back i by one
          removed = true;
        }
      }
    }

    // check again for faces with less than 3 vertices
    if ( removed )
    {
      numberOfFaces = 0;
      for ( size_t i=0 ; i<pIndexedFaceSet->coordIndex.size() ; )
      {
        bool removeFace = false;
        // scan for next -1
        unsigned int j;
        for ( j=0 ; pIndexedFaceSet->coordIndex[i+j] != -1 ; j++ )
          ;

        if ( j < 3 )
        {
          onUnsupportedToken( "IndexedFaceSet", "n-gonal coordIndex with less than 3 coords" );
          removeInvalidFace( pIndexedFaceSet, i, j, numberOfFaces );
        }
        else
        {
          i += j + 1;
          numberOfFaces++;
        }
      }
    }

    DP_ASSERT(    pIndexedFaceSet->colorIndex.empty()
                ||  ( pIndexedFaceSet->colorPerVertex
                      ? ( pIndexedFaceSet->colorIndex.size() == pIndexedFaceSet->coordIndex.size() )
                      : ( pIndexedFaceSet->colorIndex.size() == numberOfFaces ) ) );
    DP_ASSERT(    pIndexedFaceSet->normalIndex.empty()
                ||  ( pIndexedFaceSet->normalPerVertex
                      ? ( pIndexedFaceSet->normalIndex.size() == pIndexedFaceSet->coordIndex.size() )
                      : ( pIndexedFaceSet->normalIndex.size() == numberOfFaces ) ) );
    DP_ASSERT(    pIndexedFaceSet->texCoordIndex.empty()
                ||  ( pIndexedFaceSet->texCoordIndex.size() == pIndexedFaceSet->coordIndex.size() ) );
  }

  // check again, this might have changed above!
  if ( pIndexedFaceSet->coord && pIndexedFaceSet->coordIndex.empty() )
  {
    pIndexedFaceSet.reset();
  }

  return( pIndexedFaceSet );
}

IndexedLineSetSharedPtr WRLLoader::readIndexedLineSet( const string &nodeName )
{
  IndexedLineSetSharedPtr pIndexedLineSet = IndexedLineSet::create();
  pIndexedLineSet->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "color" )
    {
      readSFNode( pIndexedLineSet, pIndexedLineSet->color, getNextToken() );
      if ( pIndexedLineSet->color && ! pIndexedLineSet->color.isPtrTo<Color>() )
      {
        onUnsupportedToken( "IndexedLineSet.color", pIndexedLineSet->color->getType() );
        pIndexedLineSet->color.reset();
      }
    }
    else if ( token == "coord" )
    {
      readSFNode( pIndexedLineSet, pIndexedLineSet->coord, getNextToken() );
      if ( pIndexedLineSet->coord && ! pIndexedLineSet->coord.isPtrTo<Coordinate>() )
      {
        onUnsupportedToken( "IndexedLineSet.coord", pIndexedLineSet->coord->getType() );
        pIndexedLineSet->coord.reset();
      }
    }
    else if ( token == "colorIndex" )
    {
      readIndex( pIndexedLineSet->colorIndex );
    }
    else if ( token == "colorPerVertex" )
    {
      readSFBool( pIndexedLineSet->colorPerVertex );
    }
    else if ( token == "coordIndex" )
    {
      readIndex( pIndexedLineSet->coordIndex );
    }
    else
    {
      onUnknownToken( "IndexedLineSet", token );
    }
    token = getNextToken();
  }

  return( pIndexedLineSet );
}

InlineSharedPtr WRLLoader::readInline( const string &nodeName )
{
  InlineSharedPtr pInline = Inline::create();
  pInline->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "bboxCenter" )
    {
      readSFVec3f( pInline->bboxCenter, getNextToken() );
    }
    else if ( token == "bboxSize" )
    {
      readSFVec3f( pInline->bboxSize, getNextToken() );
    }
    else if ( token == "url" )
    {
      readMFString( pInline->url );
    }
    else
    {
      onUnknownToken( "Inline", token );
    }
    token = getNextToken();
  }

  // map multiply ref'ed Inlines on the same object
  std::map<MFString,InlineSharedPtr>::const_iterator it = m_inlines.find( pInline->url );
  if ( it == m_inlines.end() )
  {
    it = m_inlines.insert( make_pair( pInline->url, pInline ) ).first;
  }
  else
  {
    DP_ASSERT( it->second->bboxCenter == pInline->bboxCenter );
    DP_ASSERT( it->second->bboxSize   == pInline->bboxSize );
  }

  return( it->second );
}

vrml::LODSharedPtr WRLLoader::readLOD( const string &nodeName )
{
  vrml::LODSharedPtr pLOD = vrml::LOD::create();
  pLOD->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "level" )
    {
      readMFNode( pLOD );
    }
    else if ( token == "center" )
    {
      readSFVec3f( pLOD->center, getNextToken() );
    }
    else if ( token == "range" )
    {
      readMFFloat( pLOD->range );
    }
    else
    {
      onUnknownToken( "LOD", token );
    }
    token = getNextToken();
  }

  return( pLOD );
}

vrml::MaterialSharedPtr WRLLoader::readMaterial( const string &nodeName )
{
  vrml::MaterialSharedPtr pMaterial = vrml::Material::create();
  pMaterial->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "ambientIntensity" )
    {
      readSFFloat( pMaterial->ambientIntensity, getNextToken() );
    }
    else if ( token == "diffuseColor" )
    {
      readSFColor( pMaterial->diffuseColor, getNextToken() );
    }
    else if ( token == "emissiveColor" )
    {
      readSFColor( pMaterial->emissiveColor, getNextToken() );
    }
    else if ( token == "shininess" )
    {
      readSFFloat( pMaterial->shininess, getNextToken() );
    }
    else if ( token == "specularColor" )
    {
      readSFColor( pMaterial->specularColor, getNextToken() );
    }
    else if ( token == "transparency" )
    {
      readSFFloat( pMaterial->transparency, getNextToken() );
    }
    else
    {
      onUnknownToken( "Material", token );
    }
    token = getNextToken();
  }

  return( pMaterial );
}

void  WRLLoader::readMFNode( vrml::GroupSharedPtr const& fatherNode )
{
  SFNode n;
  string & token = getNextToken();
  if ( token == "[" )
  {
    token = getNextToken();
    while ( token != "]" )
    {
      readSFNode( fatherNode, n, token );
      if ( n )
      {
        fatherNode->children.push_back( n );
      }
      token = getNextToken();
      DP_ASSERT( !token.empty() );
    }
  }
  else
  {
    readSFNode( fatherNode, n, token );
    if ( n )
    {
      fatherNode->children.push_back( n );
    }
  }
}

template<typename SFType>
void  WRLLoader::readMFType( vector<SFType> &mf, void (WRLLoader::*readSFType)( SFType &sf, string &token ) )
{
  SFType  sf;
  string & token = getNextToken();
  if ( token == "[" )
  {
    token= getNextToken();
    while ( token != "]" )
    {
      (this->*readSFType)( sf, token );
      mf.push_back( sf );
      token = getNextToken();
    }
  }
  else
  {
    (this->*readSFType)( sf, token );
    mf.push_back( sf );
  }
}

MovieTextureSharedPtr WRLLoader::readMovieTexture( const string &nodeName )
{
  MovieTextureSharedPtr pMovieTexture = MovieTexture::create();
  pMovieTexture->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "loop" )
    {
      readSFBool( pMovieTexture->loop );
    }
    else if ( token == "speed" )
    {
      readSFFloat( pMovieTexture->speed, getNextToken() );
    }
    else if ( token == "startTime" )
    {
      readSFTime( pMovieTexture->startTime );
    }
    else if ( token == "stopTime" )
    {
      readSFTime( pMovieTexture->stopTime );
    }
    else if ( token == "url" )
    {
      readMFString( pMovieTexture->url );
    }
    else if ( token == "repeatS" )
    {
      readSFBool( pMovieTexture->repeatS );
    }
    else if ( token == "repeatT" )
    {
      readSFBool( pMovieTexture->repeatT );
    }
    else
    {
      onUnknownToken( "MovieTexture", token );
    }
    token = getNextToken();
  }

  return( pMovieTexture );
}

NavigationInfoSharedPtr WRLLoader::readNavigationInfo( const string &nodeName )
{
  NavigationInfoSharedPtr pNavigationInfo = NavigationInfo::create();
  pNavigationInfo->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "avatarSize" )
    {
      readMFFloat( pNavigationInfo->avatarSize );
    }
    else if ( token == "headlight" )
    {
      readSFBool( pNavigationInfo->headlight );
    }
    else if ( token == "speed" )
    {
      readSFFloat( pNavigationInfo->speed, getNextToken() );
    }
    else if ( token == "type" )
    {
      readMFString( pNavigationInfo->type );
    }
    else if ( token == "visibilityLimit" )
    {
      readSFFloat( pNavigationInfo->visibilityLimit, getNextToken() );
    }
    else
    {
      onUnknownToken( "NavigationInfo", token );
    }
    token = getNextToken();
  }

  onUnsupportedToken( "VRMLLoader", "NavigationInfo" );
  pNavigationInfo.reset();

  return( pNavigationInfo );
}

NormalSharedPtr WRLLoader::readNormal( const string &nodeName )
{
  NormalSharedPtr pNormal = Normal::create();
  pNormal->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "vector" )
    {
      readMFVec3f( pNormal->vector );
      for ( size_t i=0 ; i<pNormal->vector.size() ; i++ )
      {
        pNormal->vector[i].normalize();
      }
    }
    else
    {
      onUnknownToken( "Normal", token );
    }
    token = getNextToken();
  }

  return( pNormal );
}

NormalInterpolatorSharedPtr WRLLoader::readNormalInterpolator( const string &nodeName )
{
  NormalInterpolatorSharedPtr pNormalInterpolator = NormalInterpolator::create();
  pNormalInterpolator->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "key" )
    {
      readMFFloat( pNormalInterpolator->key );
    }
    else if ( token == "keyValue" )
    {
      readMFVec3f( pNormalInterpolator->keyValue );
    }
    else
    {
      onUnknownToken( "NormalInterpolator", token );
    }
    token = getNextToken();
  }

  DP_ASSERT( ( pNormalInterpolator->keyValue.size() % pNormalInterpolator->key.size() ) == 0 );

  return( pNormalInterpolator );
}

OrientationInterpolatorSharedPtr WRLLoader::readOrientationInterpolator( const string &nodeName )
{
  OrientationInterpolatorSharedPtr pOrientationInterpolator = OrientationInterpolator::create();
  pOrientationInterpolator->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "key" )
    {
      readMFFloat( pOrientationInterpolator->key );
    }
    else if ( token == "keyValue" )
    {
      readMFRotation( pOrientationInterpolator->keyValue );
    }
    else
    {
      onUnknownToken( "OrientationInterpolator", token );
    }
    token = getNextToken();
  }

  DP_ASSERT( pOrientationInterpolator->key.size() == pOrientationInterpolator->keyValue.size() );

  return( pOrientationInterpolator );
}

PixelTextureSharedPtr WRLLoader::readPixelTexture( const string &nodeName )
{
  PixelTextureSharedPtr pPixelTexture = PixelTexture::create();
  pPixelTexture->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "image" )
    {
      readSFImage( pPixelTexture->image );
    }
    else if ( token == "repeatS" )
    {
      readSFBool( pPixelTexture->repeatS );
    }
    else if ( token == "repeatT" )
    {
      readSFBool( pPixelTexture->repeatT );
    }
    else
    {
      onUnknownToken( "PixelTexture", token );
    }
    token = getNextToken();
  }

  return( pPixelTexture );
}

PlaneSensorSharedPtr WRLLoader::readPlaneSensor( const string &nodeName )
{
  PlaneSensorSharedPtr pPlaneSensor = PlaneSensor::create();
  pPlaneSensor->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "autoOffset" )
    {
      readSFBool( pPlaneSensor->autoOffset );
    }
    else if ( token == "enabled" )
    {
      readSFBool( pPlaneSensor->enabled );
    }
    else if ( token == "maxPosition" )
    {
      readSFVec2f( pPlaneSensor->maxPosition, getNextToken() );
    }
    else if ( token == "minPosition" )
    {
      readSFVec2f( pPlaneSensor->minPosition, getNextToken() );
    }
    else if ( token == "offset" )
    {
      readSFVec3f( pPlaneSensor->offset, getNextToken() );
    }
    else
    {
      onUnknownToken( "PlaneSensor", token );
    }
    token = getNextToken();
  }

  onUnsupportedToken( "VRMLLoader", "PlaneSensor" );
  pPlaneSensor.reset();

  return( pPlaneSensor );
}

vrml::PointLightSharedPtr WRLLoader::readPointLight( const string &nodeName )
{
  vrml::PointLightSharedPtr pPointLight = vrml::PointLight::create();
  pPointLight->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "ambientIntensity" )
    {
      readSFFloat( pPointLight->ambientIntensity, getNextToken() );
    }
    else if ( token == "attenuation" )
    {
      readSFVec3f( pPointLight->attenuation, getNextToken() );
    }
    else if ( token == "color" )
    {
      readSFColor( pPointLight->color, getNextToken() );
    }
    else if ( token == "intensity" )
    {
      readSFFloat( pPointLight->intensity, getNextToken() );
    }
    else if ( token == "location" )
    {
      readSFVec3f( pPointLight->location, getNextToken() );
    }
    else if ( token == "on" )
    {
      readSFBool( pPointLight->on );
    }
    else if ( token == "radius" )
    {
      readSFFloat( pPointLight->radius, getNextToken() );
    }
    else
    {
      onUnknownToken( "PointLight", token );
    }
    token = getNextToken();
  }

  return( pPointLight );
}

PointSetSharedPtr WRLLoader::readPointSet( const string &nodeName )
{
  PointSetSharedPtr pPointSet = PointSet::create();
  pPointSet->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "color" )
    {
      readSFNode( pPointSet, pPointSet->color, getNextToken() );
    }
    else if ( token == "coord" )
    {
      readSFNode( pPointSet, pPointSet->coord, getNextToken() );
    }
    else
    {
      onUnknownToken( "PointSet", token );
    }
    token = getNextToken();
  }

  return( pPointSet );
}

PositionInterpolatorSharedPtr WRLLoader::readPositionInterpolator( const string &nodeName )
{
  PositionInterpolatorSharedPtr pPositionInterpolator = PositionInterpolator::create();
  pPositionInterpolator->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "key" )
    {
      readMFFloat( pPositionInterpolator->key );
    }
    else if ( token == "keyValue" )
    {
      readMFVec3f( pPositionInterpolator->keyValue );
    }
    else
    {
      onUnknownToken( "PositionInterpolator", token );
    }
    token = getNextToken();
  }

  DP_ASSERT( pPositionInterpolator->key.size() == pPositionInterpolator->keyValue.size() );

  return( pPositionInterpolator );
}

void  WRLLoader::readPROTO( void )
{
  onUnsupportedToken( "VRMLLoader", "PROTO" );
  m_PROTONames.insert( getNextToken() );    //  PrototypeName
  ignoreBlock( "[", "]", getNextToken() );  //  PrototypeDeclaration
  ignoreBlock( "{", "}", getNextToken() );  //  PrototypeDefinition
}

ProximitySensorSharedPtr WRLLoader::readProximitySensor( const string &nodeName )
{
  ProximitySensorSharedPtr pProximitySensor = ProximitySensor::create();
  pProximitySensor->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "center" )
    {
      readSFVec3f( pProximitySensor->center, getNextToken() );
    }
    else if ( token == "size" )
    {
      readSFVec3f( pProximitySensor->size, getNextToken() );
    }
    else if ( token == "enabled" )
    {
      readSFBool( pProximitySensor->enabled );
    }
    else
    {
      onUnknownToken( "ProximitySensor", token );
    }
    token = getNextToken();
  }

  onUnsupportedToken( "VRMLLoader", "ProximitySensor" );
  pProximitySensor.reset();

  return( pProximitySensor );
}

void  WRLLoader::readROUTE( const SFNode currentNode )
{
  bool ok = true;
  string from = getNextToken(); // no reference here!!!
  string & to = getNextToken(); // but reference allowed here.
  if ( to != "TO" )
  {
    onUnexpectedToken( "TO", to );
    ok = false;
  }
  else
  {
    to = getNextToken();
  }

  string  fromName, fromAction;
  SFNode  fromNode;
  if ( ok )
  {
    string::size_type sst = from.find( "." );
    if ( sst == string::npos )
    {
      onUnexpectedToken( "NodeName.eventOutName", from );
      ok = false;
    }
    else
    {
      fromName.assign( from, 0, sst );
      fromAction.assign( from, sst+1, string::npos );
      fromNode = findNode( currentNode, fromName );
    }
  }

  string  toName, toAction;
  SFNode  toNode;
  if ( ok )
  {
    string::size_type sst = to.find( "." );
    if ( sst == string::npos )
    {
      onUnexpectedToken( "NodeName.eventInName", to );
      ok = false;
    }
    else
    {
      toName.assign( to, 0, sst );
      toAction.assign( to, sst+1, string::npos );
      toNode = findNode( currentNode, toName );
    }
  }

  if ( ok )
  {
    if ( ! fromNode )
    {
      onUndefinedToken( "ROUTE nodeOutName", fromName );
    }
    else if ( toAction == "set_scaleOrientation" )
    {
      onUnsupportedToken( "ROUTE", toAction );
    }
    else if (   ( toAction == "set_scale" )
            &&  ! (   fromNode.isPtrTo<PositionInterpolator>()
                  &&  isValidScaling( fromNode.staticCast<PositionInterpolator>() ) ) )
    {
      onInvalidValue( 0, "ROUTE eventInName", toAction );
    }
    else if ( ! toNode )
    {
      onUndefinedToken( "ROUTE nodeInName", toName );
    }
    else
    {
      if ( fromNode.isPtrTo<ColorInterpolator>() )
      {
        if ( fromAction != "value_changed" )
        {
          onUnsupportedToken( "ROUTE eventOutName", fromAction );
        }
        else if ( ! toNode.isPtrTo<Color>() )
        {
          onUnsupportedToken( "ROUTE nodeInName", toNode->getType() );
        }
        else if ( ( toAction == "set_color" ) || ( toAction == "color" ) )
        {
          toNode.staticCast<Color>()->set_color = fromNode.staticCast<ColorInterpolator>();
        }
        else
        {
          onUnsupportedToken( "ROUTE eventInName", toAction );
        }
      }
      else if ( fromNode.isPtrTo<CoordinateInterpolator>() )
      {
        if ( fromAction != "value_changed" )
        {
          onUnsupportedToken( "ROUTE eventOutName", fromAction );
        }
        else if ( ! toNode.isPtrTo<Coordinate>() )
        {
          onUnsupportedToken( "ROUTE nodeInName", toNode->getType() );
        }
        else if ( ( toAction == "set_point" ) || ( toAction == "point" ) )
        {
          toNode.staticCast<Coordinate>()->set_point = fromNode.staticCast<CoordinateInterpolator>();
        }
        else
        {
          onUnsupportedToken( "ROUTE eventInName", toAction );
        }
      }
      else if ( fromNode.isPtrTo<NormalInterpolator>() )
      {
        if ( fromAction != "value_changed" )
        {
          onUnsupportedToken( "ROUTE eventOutName", fromAction );
        }
        else if ( ! toNode.isPtrTo<Normal>() )
        {
          onUnsupportedToken( "ROUTE nodeInName", toNode->getType() );
        }
        else if ( ( toAction == "set_vector" ) || ( toAction == "vector" ) )
        {
          toNode.staticCast<Normal>()->set_vector = fromNode.staticCast<NormalInterpolator>();
        }
        else
        {
          onUnsupportedToken( "ROUTE eventInName", toAction );
        }
      }
      else if ( fromNode.isPtrTo<OrientationInterpolator>() )
      {
        if ( fromAction != "value_changed" )
        {
          onUnsupportedToken( "ROUTE eventOutName", fromAction );
        }
        if ( toNode.isPtrTo<vrml::Transform>() )
        {
          if ( ( toAction == "set_rotation" ) || ( toAction == "rotation" ) )
          {
            toNode.staticCast<vrml::Transform>()->set_rotation = fromNode.staticCast<OrientationInterpolator>();
          }
          else
          {
            onUnsupportedToken( "ROUTE eventInName", toAction );
          }
        }
        else if ( toNode.isPtrTo<Viewpoint>() )
        {
          if ( ( toAction == "set_orientation" ) || ( toAction == "orientation" ) )
          {
            toNode.staticCast<Viewpoint>()->set_orientation = fromNode.staticCast<OrientationInterpolator>();
          }
          else
          {
            onUnsupportedToken( "ROUTE eventInName", toAction );
          }
        }
        else
        {
          onUnsupportedToken( "ROUTE nodeInName", toNode->getType() );
        }
      }
      else if ( fromNode.isPtrTo<PositionInterpolator>() )
      {
        if ( fromAction != "value_changed" )
        {
          onUnsupportedToken( "ROUTE eventOutName", fromAction );
        }
        if ( toNode.isPtrTo<vrml::Transform>() )
        {
          if ( ( toAction == "set_center" ) || ( toAction == "center" ) )
          {
            toNode.staticCast<vrml::Transform>()->set_center = fromNode.staticCast<PositionInterpolator>();
          }
          else if ( ( toAction == "set_scale" ) || ( toAction == "scale" ) )
          {
            toNode.staticCast<vrml::Transform>()->set_scale = fromNode.staticCast<PositionInterpolator>();
          }
          else if ( ( toAction == "set_translation" ) || ( toAction == "translation" ) )
          {
            toNode.staticCast<vrml::Transform>()->set_translation = fromNode.staticCast<PositionInterpolator>();
          }
          else
          {
            onUnsupportedToken( "ROUTE eventInName", toAction );
          }
        }
        else if ( toNode.isPtrTo<Viewpoint>() )
        {
          if ( ( toAction == "set_position" ) || ( toAction == "position" ) )
          {
            toNode.staticCast<Viewpoint>()->set_position = fromNode.staticCast<PositionInterpolator>();
          }
          else
          {
            onUnsupportedToken( "ROUTE eventInName", toAction );
          }
        }
        else
        {
          onUnsupportedToken( "ROUTE nodeInName", toNode->getType() );
        }
      }
      else if ( fromNode.isPtrTo<TimeSensor>() )
      {
        if ( fromAction != "fraction_changed" )
        {
          onUnsupportedToken( "ROUTE eventOutName", fromAction );
        }
        else if ( toNode.isPtrTo<Interpolator>() )
        {
          if ( ( toAction == "set_fraction" ) || ( toAction == "fraction" ) )
          {
            toNode.staticCast<Interpolator>()->set_fraction = fromNode.staticCast<TimeSensor>();
          }
          else
          {
            onUnsupportedToken( "ROUTE eventInName", toAction );
          }
        }
        else
        {
          onUnsupportedToken( "ROUTE nodeInName", toNode->getType() );
        }
      }
      else
      {
        onUnsupportedToken( "ROUTE nodeOutName", fromNode->getType() );
        ok = false;
      }
    }
  }
}

ScalarInterpolatorSharedPtr WRLLoader::readScalarInterpolator( const string &nodeName )
{
  ScalarInterpolatorSharedPtr pScalarInterpolator = ScalarInterpolator::create();
  pScalarInterpolator->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "key" )
    {
      readMFFloat( pScalarInterpolator->key );
    }
    else if ( token == "keyValue" )
    {
      readMFFloat( pScalarInterpolator->keyValue );
    }
    else
    {
      onUnknownToken( "ScalarInterpolator", token );
    }
    token = getNextToken();
  }

  DP_ASSERT( pScalarInterpolator->key.size() == pScalarInterpolator->keyValue.size() );

  onUnsupportedToken( "VRMLLoader", "ScalarInterpolator" );
  pScalarInterpolator.reset();

  return( pScalarInterpolator );
}

ScriptSharedPtr WRLLoader::readScript( const string &nodeName )
{
#if 0
  DP_ASSERT( false );
  ScriptSharedPtr pScript = Script::create();
  pScript->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "url" )
    {
      DP_ASSERT( false );
      readMFString( pScript->url );
    }
    else if ( token == "directOutput" )
    {
      DP_ASSERT( false );
      readSFBool( pScript->directOutput );
    }
    else if ( token == "mustEvaluate" )
    {
      DP_ASSERT( false );
      readSFBool( pScript->mustEvaluate );
    }
    else
    {
      onUnknownToken( "Script", token );
    }
    token = getNextToken();
  }

  return( pScript );
#else
  onUnsupportedToken( "VRMLLoader", "Script" );
  ignoreBlock( "{", "}", getNextToken() );
  return( ScriptSharedPtr::null );
#endif
}

void  WRLLoader::ignoreBlock( const string &open, const string &close, string &token )
{
  onUnexpectedToken( open, token );
  token = getNextToken();
  while ( token != close )
  {
    if ( token == "{" )
    {
      ignoreBlock( "{", "}", token );
    }
    else if ( token == "[" )
    {
      ignoreBlock( "[", "]", token );
    }
    token = getNextToken();
  }
}

void  WRLLoader::readSFBool( SFBool &b )
{
  b = ( getNextToken() == "TRUE" );
}

void  WRLLoader::readSFColor( SFColor &c, string &token )
{
  readSFVec3f( c, token );
  c[0] = clamp( c[0], 0.0f, 1.0f );
  c[1] = clamp( c[1], 0.0f, 1.0f );
  c[2] = clamp( c[2], 0.0f, 1.0f );
}

void  WRLLoader::readSFFloat( SFFloat &f, string &token )
{
  f = _atof( token );
}

void  WRLLoader::readSFImage( SFImage &image )
{
  readSFInt32( image.width, getNextToken() );
  readSFInt32( image.height, getNextToken() );
  readSFInt32( image.numComponents, getNextToken() );
  switch( image.numComponents )
  {
    case 1 :
      image.pixelsValues = new SFInt8[image.width*image.height];
      for ( int i=0 ; i<image.height; i++ )
      {
        for ( int j=0 ; j<image.width ; j++ )
        {
          readSFInt8( image.pixelsValues[i*image.width+j] );
        }
      }
      break;
    case 3 :
    case 4 :
      {
        SFInt32 * pv = new SFInt32[image.width*image.height];
        for ( int i=0 ; i<image.height; i++ )
        {
          for ( int j=0 ; j<image.width ; j++ )
          {
            readSFInt32( pv[i*image.width+j], getNextToken() );
          }
        }
        image.pixelsValues = (SFInt8*) pv;
      }
      break;
    default:
      onInvalidValue( image.numComponents, "SFImage", "numComponents" );
      break;
  }

  onUnsupportedToken( "VRMLLoader", "SFImage" );
}

void  WRLLoader::readSFInt8( SFInt8 &i )
{
  i = atoi( getNextToken().c_str() );
}

void  WRLLoader::readSFInt32( SFInt32 &i, string &token )
{
  i = atoi( token.c_str() );
}

void  WRLLoader::readSFNode( const SFNode fatherNode, SFNode &n, string &token )
{
  m_openNodes.push_back( fatherNode );

  if ( token == "DEF" )
  {
    string nodeName = getNextToken();   // no reference here!!
    n = getNode( nodeName, getNextToken() );
    if ( n )
    {
      m_defNodes[nodeName] = n;
    }
  }
  else if ( token == "NULL" )
  {
    n = SFNode::null;
  }
  else if ( token == "USE" )
  {
    string & nodeName = getNextToken();
    if ( m_defNodes.find( nodeName ) != m_defNodes.end() )
    {
      n = m_defNodes[nodeName];
    }
  }
  else
  {
    n = getNode( "", token );
  }

  DP_ASSERT( m_openNodes[m_openNodes.size()-1] == fatherNode );
  m_openNodes.pop_back();
}

void  WRLLoader::readSFRotation( SFRotation &r, string &token )
{
  SFVec3f axis;
  SFFloat angle;
  readSFVec3f( axis, token );
  readSFFloat( angle, getNextToken() );
  r = SFRotation( axis, angle );
}

void  WRLLoader::readSFString( SFString &s, string &token )
{
  if ( token[0] == '"' )
  {
    s = ( token.length() > 1 ) ? &token[1] : token = getNextToken();
    while ( token[token.length()-1] != '"' )
    {
      token = getNextToken();
      s += " ";
      s += token;
    }
    s.replace( s.find( "\"" ), 1, "" );
  }
  else
  {
    s = token;
  }
}

void  WRLLoader::readSFTime( SFTime &t )
{
  t = _atof( getNextToken() );
}

void  WRLLoader::readSFVec2f( SFVec2f &v, string &token )
{
  readSFFloat( v[0], token );
  readSFFloat( v[1], getNextToken() );
}

void  WRLLoader::readSFVec3f( SFVec3f &v, string &token )
{
  readSFFloat( v[0], token );
  readSFFloat( v[1], getNextToken() );
  readSFFloat( v[2], getNextToken() );
}

vrml::ShapeSharedPtr WRLLoader::readShape( const string &nodeName )
{
  vrml::ShapeSharedPtr pShape = vrml::Shape::create();
  pShape->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "appearance" )
    {
      readSFNode( pShape, pShape->appearance, getNextToken() );
      if ( pShape->appearance && ! pShape->appearance.isPtrTo<Appearance>() )
      {
        onUnsupportedToken( "Shape.appearance", pShape->appearance->getType() );
        pShape->appearance.reset();
      }
    }
    else if ( token == "geometry" )
    {
      readSFNode( pShape, pShape->geometry, getNextToken() );
      if ( ! pShape->geometry )
      {
        onEmptyToken( "Shape", "geometry" );
      }
      else if ( ! pShape->geometry.isPtrTo<vrml::Geometry>() )
      {
        onUnsupportedToken( "Shape.geometry", pShape->geometry->getType() );
        pShape->geometry.reset();
      }
    }
    else
    {
      onUnknownToken( "Shape", token );
    }
    token = getNextToken();
  }

  return( pShape );
}

SoundSharedPtr WRLLoader::readSound( const string &nodeName )
{
  SoundSharedPtr pSound = Sound::create();
  pSound->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "direction" )
    {
      readSFVec3f( pSound->direction, getNextToken() );
    }
    else if ( token == "intensity" )
    {
      readSFFloat( pSound->intensity, getNextToken() );
    }
    else if ( token == "location" )
    {
      readSFVec3f( pSound->location, getNextToken() );
    }
    else if ( token == "maxBack" )
    {
      readSFFloat( pSound->maxBack, getNextToken() );
    }
    else if ( token == "maxFront" )
    {
      readSFFloat( pSound->maxFront, getNextToken() );
    }
    else if ( token == "minBack" )
    {
      readSFFloat( pSound->minBack, getNextToken() );
    }
    else if ( token == "minFront" )
    {
      readSFFloat( pSound->minFront, getNextToken() );
    }
    else if ( token == "priority" )
    {
      readSFFloat( pSound->priority, getNextToken() );
    }
    else if ( token == "source" )
    {
      readSFNode( pSound, pSound->source, getNextToken() );
    }
    else if ( token == "spatialize" )
    {
      readSFBool( pSound->spatialize );
    }
    else
    {
      onUnknownToken( "Sound", token );
    }
    token = getNextToken();
  }

  onUnsupportedToken( "VRMLLoader", "Sound" );
  pSound.reset();

  return( pSound );
}

SphereSharedPtr WRLLoader::readSphere( const string &nodeName )
{
  SphereSharedPtr pSphere = Sphere::create();
  pSphere->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "radius" )
    {
      readSFFloat( pSphere->radius, getNextToken() );
    }
    else
    {
      onUnknownToken( "Sphere", token );
    }
    token = getNextToken();
  }

  return( pSphere );
}

SphereSensorSharedPtr WRLLoader::readSphereSensor( const string &nodeName )
{
  SphereSensorSharedPtr pSphereSensor = SphereSensor::create();
  pSphereSensor->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "autoOffset" )
    {
      readSFBool( pSphereSensor->autoOffset );
    }
    else if ( token == "enabled" )
    {
      readSFBool( pSphereSensor->enabled );
    }
    else if ( token == "offset" )
    {
      readSFRotation( pSphereSensor->offset, getNextToken() );
    }
    else
    {
      onUnknownToken( "SphereSensor", token );
    }
    token = getNextToken();
  }

  onUnsupportedToken( "VRMLLoader", "SphereSensor" );
  pSphereSensor.reset();

  return( pSphereSensor );
}

vrml::SpotLightSharedPtr WRLLoader::readSpotLight( const string &nodeName )
{
  vrml::SpotLightSharedPtr pSpotLight = vrml::SpotLight::create();
  pSpotLight->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "ambientIntensity" )
    {
      readSFFloat( pSpotLight->ambientIntensity, getNextToken() );
    }
    else if ( token == "attenuation" )
    {
      readSFVec3f( pSpotLight->attenuation, getNextToken() );
    }
    else if ( token == "beamWidth" )
    {
      readSFFloat( pSpotLight->beamWidth, getNextToken() );
    }
    else if ( token == "color" )
    {
      readSFColor( pSpotLight->color, getNextToken() );
    }
    else if ( token == "cutOffAngle" )
    {
      readSFFloat( pSpotLight->cutOffAngle, getNextToken() );
    }
    else if ( token == "direction" )
    {
      readSFVec3f( pSpotLight->direction, getNextToken() );
    }
    else if ( token == "intensity" )
    {
      readSFFloat( pSpotLight->intensity, getNextToken() );
    }
    else if ( token == "location" )
    {
      readSFVec3f( pSpotLight->location, getNextToken() );
    }
    else if ( token == "on" )
    {
      readSFBool( pSpotLight->on );
    }
    else if ( token == "radius" )
    {
      readSFFloat( pSpotLight->radius, getNextToken() );
    }
    else
    {
      onUnknownToken( "SpotLight", token );
    }
    token = getNextToken();
  }

  return( pSpotLight );
}

void  WRLLoader::readStatements( void )
{
  string & token = getNextToken();
  while ( !token.empty() )
  {
    if ( token == "EXTERNPROTO" )
    {
      readEXTERNPROTO();
    }
    else if ( token == "PROTO" )
    {
      readPROTO();
    }
    else if ( token == "ROUTE" )
    {
      readROUTE( SFNode::null );
    }
    else
    {
      SFNode  n;
      readSFNode( SFNode::null, n, token );
      if ( n )
      {
        m_topLevelGroup->children.push_back( n );
      }
    }
    token = getNextToken();
  }
}

vrml::SwitchSharedPtr WRLLoader::readSwitch( const string &nodeName )
{
  vrml::SwitchSharedPtr pSwitch = vrml::Switch::create();
  pSwitch->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "choice" )
    {
      readMFNode( pSwitch );
    }
    else if ( token == "whichChoice" )
    {
      readSFInt32( pSwitch->whichChoice, getNextToken() );
    }
    else
    {
      onUnknownToken( "Switch", token );
    }
    token = getNextToken();
  }

  return( pSwitch );
}

TextSharedPtr WRLLoader::readText( const string &nodeName )
{
  TextSharedPtr pText = Text::create();
  pText->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "string" )
    {
      readMFString( pText->string );
    }
    else if ( token == "fontStyle" )
    {
      readSFNode( pText, pText->fontStyle, getNextToken() );
    }
    else if ( token == "length" )
    {
      readMFFloat( pText->length );
    }
    else if ( token == "maxExtent" )
    {
      readSFFloat( pText->maxExtent, getNextToken() );
    }
    else
    {
      onUnknownToken( "Text", token );
    }
    token = getNextToken();
  }

  onUnsupportedToken( "VRMLLoader", "Text" );
  pText.reset();

  return( pText );
}

TextureCoordinateSharedPtr WRLLoader::readTextureCoordinate( const string &nodeName )
{
  TextureCoordinateSharedPtr pTextureCoordinate = TextureCoordinate::create();
  pTextureCoordinate->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "point" )
    {
      readMFVec2f( pTextureCoordinate->point );
    }
    else
    {
      onUnknownToken( "TextureCoordinate", token );
    }
    token = getNextToken();
  }

  return( pTextureCoordinate );
}

TextureTransformSharedPtr WRLLoader::readTextureTransform( const string &nodeName )
{
  TextureTransformSharedPtr pTextureTransform = TextureTransform::create();
  pTextureTransform->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "center" )
    {
      readSFVec2f( pTextureTransform->center, getNextToken() );
    }
    else if ( token == "rotation" )
    {
      readSFFloat( pTextureTransform->rotation, getNextToken() );
    }
    else if ( token == "scale" )
    {
      readSFVec2f( pTextureTransform->scale, getNextToken() );
    }
    else if ( token == "translation" )
    {
      readSFVec2f( pTextureTransform->translation, getNextToken() );
    }
    else
    {
      onUnknownToken( "TextureTransform", token );
    }
    token = getNextToken();
  }

  return( pTextureTransform );
}

TimeSensorSharedPtr WRLLoader::readTimeSensor( const string &nodeName )
{
  TimeSensorSharedPtr pTimeSensor = TimeSensor::create();
  pTimeSensor->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "cycleInterval" )
    {
      readSFTime( pTimeSensor->cycleInterval );
    }
    else if ( token == "enabled" )
    {
      readSFBool( pTimeSensor->enabled );
    }
    else if ( token == "loop" )
    {
      readSFBool( pTimeSensor->loop );
    }
    else if ( token == "startTime" )
    {
      readSFTime( pTimeSensor->startTime );
    }
    else if ( token == "stopTime" )
    {
      readSFTime( pTimeSensor->stopTime );
    }
    else
    {
      onUnknownToken( "TimeSensor", token );
    }
    token = getNextToken();
  }

  return( pTimeSensor );
}

TouchSensorSharedPtr WRLLoader::readTouchSensor( const string &nodeName )
{
  TouchSensorSharedPtr pTouchSensor = TouchSensor::create();
  pTouchSensor->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "enabled" )
    {
      readSFBool( pTouchSensor->enabled );
    }
    else
    {
      onUnknownToken( "TouchSensor", token );
    }
    token = getNextToken();
  }

  onUnsupportedToken( "VRMLLoader", "TouchSensor" );
  pTouchSensor.reset();

  return( pTouchSensor );
}

vrml::TransformSharedPtr WRLLoader::readTransform( const string &nodeName )
{
  vrml::TransformSharedPtr pTransform = vrml::Transform::create();
  pTransform->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "center" )
    {
      readSFVec3f( pTransform->center, getNextToken() );
    }
    else if ( token == "children" )
    {
      readMFNode( pTransform );
    }
    else if ( token == "rotation" )
    {
      readSFRotation( pTransform->rotation, getNextToken() );
    }
    else if ( token == "scale" )
    {
      readSFVec3f( pTransform->scale, getNextToken() );
    }
    else if ( token == "scaleOrientation" )
    {
      readSFRotation( pTransform->scaleOrientation, getNextToken() );
    }
    else if ( token == "translation" )
    {
      readSFVec3f( pTransform->translation, getNextToken() );
    }
    else if ( token == "ROUTE" )
    {
      readROUTE( pTransform );
    }
    else
    {
      onUnknownToken( "Transform", token );
    }
    token = getNextToken();
  }

  if ( ! isValidScaling( pTransform->scale ) )
  {
    pTransform.reset();
  }

  return( pTransform );
}

ViewpointSharedPtr WRLLoader::readViewpoint( const string &nodeName )
{
  ViewpointSharedPtr pViewpoint = Viewpoint::create();
  pViewpoint->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "fieldOfView" )
    {
      readSFFloat( pViewpoint->fieldOfView, getNextToken() );
      DP_ASSERT( ( 0.0f < pViewpoint->fieldOfView ) && ( pViewpoint->fieldOfView < PI ) );
    }
    else if ( token == "jump" )
    {
      readSFBool( pViewpoint->jump );
    }
    else if ( token == "orientation" )
    {
      readSFRotation( pViewpoint->orientation, getNextToken() );
    }
    else if ( token == "position" )
    {
      readSFVec3f( pViewpoint->position, getNextToken() );
    }
    else if ( token == "description" )
    {
      readSFString( pViewpoint->description, getNextToken() );
    }
    else
    {
      onUnknownToken( "Viewpoint", token );
    }
    token = getNextToken();
  }

  return( pViewpoint );
}

VisibilitySensorSharedPtr WRLLoader::readVisibilitySensor( const string &nodeName )
{
  VisibilitySensorSharedPtr pVisibilitySensor = VisibilitySensor::create();
  pVisibilitySensor->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "center" )
    {
      readSFVec3f( pVisibilitySensor->center, getNextToken() );
    }
    else if ( token == "enabled" )
    {
      readSFBool( pVisibilitySensor->enabled );
    }
    else if ( token == "size" )
    {
      readSFVec3f( pVisibilitySensor->size, getNextToken() );
    }
    else
    {
      onUnknownToken( "VisibilitySensor", token );
    }
    token = getNextToken();
  }

  onUnsupportedToken( "VRMLLoader", "VisibilitySensor" );
  pVisibilitySensor.reset();

  return( pVisibilitySensor );
}

WorldInfoSharedPtr WRLLoader::readWorldInfo( const string &nodeName )
{
  WorldInfoSharedPtr pWorldInfo = WorldInfo::create();
  pWorldInfo->setName( nodeName );

  string & token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "info" )
    {
      readMFString( pWorldInfo->info );
    }
    else if ( token == "title" )
    {
      readSFString( pWorldInfo->title, getNextToken() );
    }
    else
    {
      onUnknownToken( "WorldInfo", token );
    }
    token = getNextToken();
  }

  onUnsupportedToken( "VRMLLoader", "WorldInfo" );
  pWorldInfo.reset();

  return( pWorldInfo );
}

void WRLLoader::setNextToken( void )
{
  if ( ( m_nextTokenEnd == string::npos ) && !m_eof )
  {
    getNextLine();
    m_nextTokenEnd = 0;
  }
  if ( !m_eof )
  {
    DP_ASSERT( !m_currentString.empty() );
    do
    {
      m_nextTokenStart = findNotDelimiter( m_currentString, m_nextTokenEnd );   // find_first_not_of is slower!
      if ( ( m_nextTokenStart == string::npos ) || ( m_currentString[m_nextTokenStart] == '#' ) )
      {
        getNextLine();
        m_nextTokenEnd = m_eof ? string::npos : 0;
      }
      else
      {
        m_nextTokenEnd = findDelimiter( m_currentString, m_nextTokenStart );    // find_first_of is slower!
      }
    } while ( m_nextTokenEnd == 0 );
  }
}

bool  WRLLoader::testWRLVersion( const string &filename )
{
  m_lineNumber = 0;
  bool ok = getNextLine();
  if ( ok )
  {
    if ( m_currentString.compare( 0, 15, "#VRML V2.0 utf8" ) == 0 )
    {
      m_currentString.clear();
      m_nextTokenEnd = string::npos;
      setNextToken();
    }
    else if ( ( m_currentString.compare( 0, 16, "#VRML V1.0 ascii" ) == 0 )
          ||  ( m_currentString.compare( 0, 4, "#X3D" ) == 0 ) )
    {
      while ( iscntrl( m_currentString.back() ) )
      {
        m_currentString.pop_back();
      }
      std::ostringstream message;
      message << "WRLLoader: unsupported VRML version <" << m_currentString << "> in file <" << filename << ">" << std::endl;
      throw std::runtime_error( message.str().c_str() );
    }
    else
    {
      std::ostringstream message;
      message << "WRLLoader: the file <" << filename << "> is not a valid VRML file" << std::endl;
      throw std::runtime_error( message.str().c_str() );
    }
  }
  else if ( callback() )
  {
    callback()->onUnexpectedEndOfFile( m_lineNumber );
  }
  return( ok );
}
