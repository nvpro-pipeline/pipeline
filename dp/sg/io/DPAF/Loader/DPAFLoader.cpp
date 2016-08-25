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


#include <cstdio> // some include between this line and DPAFLoader.h confuses gcc so that cstdio cannot be included anymore.
#include <dp/Exception.h>
#include <dp/fx/EffectLibrary.h>
#if defined(HAVE_HALF_FLOAT)
#include <dp/math/half.h>
#endif
#include <dp/sg/core/Config.h>
#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/BufferHost.h>
#include <dp/sg/core/ClipPlane.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/MatrixCamera.h>
#include <dp/sg/core/ParallelCamera.h>
#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/algorithm/AppTraverser.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/sg/io/PlugInterfaceID.h>
#include <dp/sg/io/IO.h>
#include <dp/util/File.h>
#include <dp/util/Locale.h>
#include "DPAFLoader.h"
#include <utility>

using namespace dp::sg::core;
using namespace dp::math;
using namespace dp::util;
using std::make_pair;
using std::map;
using std::pair;
using std::string;
using std::vector;

const UPITID PITID_SCENE_LOADER(UPITID_SCENE_LOADER, UPITID_VERSION); // plug-in type
UPIID PIID_DP_SCENE_LOADER = UPIID(".DPAF", PITID_SCENE_LOADER);

#if defined( _WIN32 )
// is this necessary??
BOOL APIENTRY DllMain(HANDLE hModule, DWORD reason, LPVOID lpReserved)
{
  if (reason == DLL_PROCESS_ATTACH)
  {
    // initialize supported Plug Interface ID
    PIID_DP_SCENE_LOADER = UPIID(".DPAF", PITID_SCENE_LOADER);
    int i=0;
  }

  return TRUE;
}
#elif defined( LINUX )
void lib_init()
{
  int i=0;
}
#endif

bool getPlugInterface(const UPIID& piid, dp::util::PlugInSharedPtr & pi)
{
  if (piid==PIID_DP_SCENE_LOADER)
  {
    pi = DPAFLoader::create();
    return( !!pi );
  }
  return false;
}

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();

  piids.push_back(PIID_DP_SCENE_LOADER);
}

DPAFLoaderSharedPtr DPAFLoader::create()
{
  return( std::shared_ptr<DPAFLoader>( new DPAFLoader() ) );
}

DPAFLoader::DPAFLoader()
: m_line(0)
, m_tokenizer( std::string(), boost::char_separator<char>( " ,;\t\n" ) )
{
  m_tokenizer.assign( m_currentLine );        // need to assign a valid string (not a temporary created one)
  m_tokenizerIterator = m_tokenizer.end();    // to get an iterator the end
}

DPAFLoader::~DPAFLoader()
{
#if !defined(NDEBUG)
  assertClean();
#endif
}

void  DPAFLoader::cleanup( void )
{
  m_billboards.clear();
  m_buffers.clear();
  m_geoNodes.clear();
  m_groups.clear();
  m_indexSets.clear();
  m_lightSources.clear();
  m_LODs.clear();
  m_matrixCameras.clear();
  m_objects.clear();
  m_parallelCameras.clear();
  m_parameterGroupData.clear();
  m_perspectiveCameras.clear();
  m_pipelineData.clear();
  m_primitives.clear();
  m_quadMeshes.clear();
  m_samplers.clear();
  m_switches.clear();
  m_textureImages.clear();
  m_transforms.clear();
  m_vertexAttributeSets.clear();

  m_scene.reset();

  m_currentLine.clear();
  m_currentString.clear();
  m_ifs.clear();
  m_line = 0;

  m_fileFinder.clear();
}

#if !defined(NDEBUG)
void DPAFLoader::assertClean()
{
  DP_ASSERT( m_billboards.empty() );
  DP_ASSERT( m_buffers.empty() );
  DP_ASSERT( m_geoNodes.empty() );
  DP_ASSERT( m_groups.empty() );
  DP_ASSERT( m_indexSets.empty() );
  DP_ASSERT( m_lightSources.empty() );
  DP_ASSERT( m_LODs.empty() );
  DP_ASSERT( m_matrixCameras.empty() );
  DP_ASSERT( m_objects.empty() );
  DP_ASSERT( m_parallelCameras.empty() );
  DP_ASSERT( m_parameterGroupData.empty() );
  DP_ASSERT( m_perspectiveCameras.empty() );
  DP_ASSERT( m_pipelineData.empty() );
  DP_ASSERT( m_primitives.empty() );
  DP_ASSERT( m_quadMeshes.empty() );
  DP_ASSERT( m_samplers.empty() );
  DP_ASSERT( m_switches.empty() );
  DP_ASSERT( m_textureImages.empty() );
  DP_ASSERT( m_transforms.empty() );
  DP_ASSERT( m_vertexAttributeSets.empty() );

  DP_ASSERT( ! m_scene );

  DP_ASSERT( m_currentLine.empty() );
  DP_ASSERT( m_currentString.empty() );
  DP_ASSERT( m_line == 0 );
}
#endif

SceneSharedPtr DPAFLoader::load( const string& filename, dp::util::FileFinder const& fileFinder, dp::sg::ui::ViewStateSharedPtr & viewState )
{
#if !defined(NDEBUG)
  assertClean();
#endif

  if ( !dp::util::fileExists(filename) )
  {
    throw dp::FileNotFoundException( filename );
  }

  // set the locale temporarily to the default "C" to make atof behave predictably
  dp::util::Locale tl("C");

  // private copy of the search paths
  m_fileFinder = fileFinder;

  // run the importer
  try
  {
    m_scene = import( filename, viewState );
    if ( !m_scene && callback() )
    {
      callback()->onFileEmpty( filename );
    }
  }
  catch( ... )
  {
    // note: stack unwinding doesn't consider heap objects!
    cleanup();
    throw;
  }

  SceneSharedPtr scene = m_scene;  // get the scene locally, before it's cleaned up
  cleanup();

  if ( !scene )
  {
    throw std::runtime_error( "Failed to load file <" + filename + ">. Scene is empty" );
  }

  return( scene );                   // and return that scene
}

bool DPAFLoader::getNextLine( void )
{
  getline( m_ifs, m_currentString );
  m_currentLine = m_currentString;
  m_line++;
  return( m_ifs.good() );
}

std::string DPAFLoader::getNextToken( void )
{
  onUnexpectedEndOfFile( ( m_tokenizerIterator == m_tokenizer.end() ) && m_ifs.eof() );

  while ( ( m_tokenizerIterator == m_tokenizer.end() ) && !m_ifs.eof() )
  {
    getNextLine();
    m_tokenizer.assign( m_currentLine );
    m_tokenizerIterator = m_tokenizer.begin();
  }

  return( ( m_tokenizerIterator != m_tokenizer.end() ) ? *m_tokenizerIterator++ : "" );
}

SceneSharedPtr DPAFLoader::import( const string &filename, dp::sg::ui::ViewStateSharedPtr & viewState )
{
  m_ifs.open( filename.c_str() );
  if ( m_ifs.good() )
  {
    if ( testDPVersion() )
    {
      string token(getNextToken());
      while ( !token.empty() )
      {
        string name = readName( getNextToken() );
        // cut temporary counters from object name
        string objName(name.substr(0, name.rfind("_dpaf")));

        if ( token == "Billboard" )
        {
          storeNamedObject(name, m_billboards, readBillboard( objName.c_str(), name ));
        }
        else if ( token == "Buffer" )
        {
          m_buffers[name] = readBuffer(); // buffer is not an Object. storeNamedObject cannot be used.
        }
        else if ( token == "GeoNode" )
        {
          storeNamedObject(name, m_geoNodes, readGeoNode( objName.c_str() ));
        }
        else if ( token == "Group" )
        {
          storeNamedObject(name, m_groups, readGroup( objName.c_str(), name ));
        }
        else if ( token == "IndexSet" )
        {
          storeNamedObject( name, m_indexSets, readIndexSet( objName.c_str() ) );
        }
        else if ( token == "LightSource" )
        {
          storeNamedObject(name, m_lightSources, readLightSource( objName.c_str() ));
        }
        else if ( token == "LOD" )
        {
          storeNamedObject(name, m_LODs, readLOD( objName.c_str(), name ));
        }
        else if ( token == "MatrixCamera" )
        {
          storeNamedObject(name, m_matrixCameras, readMatrixCamera( objName.c_str() ));
        }
        else if ( token == "ParallelCamera" )
        {
          storeNamedObject(name, m_parallelCameras, readParallelCamera( objName.c_str() ));
        }
        else if ( token == "ParameterGroupData" )
        {
          storeNamedObject(name, m_parameterGroupData, readParameterGroupData( objName.c_str() ));
        }
        else if ( token == "PipelineData" )
        {
          storeNamedObject(name, m_pipelineData, readPipelineData( objName.c_str() ));
        }
        else if ( token == "PerspectiveCamera" )
        {
          storeNamedObject(name, m_perspectiveCameras, readPerspectiveCamera( objName.c_str() ));
        }
        else if ( token == "Primitive" )
        {
          storeNamedObject(name, m_primitives, readPrimitive( objName.c_str() ));
        }
        else if ( token == "Sampler" )
        {
          storeNamedObject( name, m_samplers, readSampler( objName ) );
        }
        else if ( token == "Scene" )
        {
          DP_ASSERT( !m_scene );
          onUnexpectedToken( "{", name );
          m_scene = readScene();
        }
        else if ( token == "Switch" )
        {
          storeNamedObject(name, m_switches, readSwitch( objName.c_str(), name ));
        }
        else if ( token == "TextureHost" )
        {
          storeTextureHost( name, readTextureHost( objName.c_str() ) );
        }
        else if ( token == "Transform" )
        {
          storeNamedObject(name, m_transforms, readTransform( objName.c_str(), name ));
        }
        else if ( token == "VertexAttributeSet" )
        {
          storeNamedObject(name, m_vertexAttributeSets, readVertexAttributeSet( objName.c_str() ));
        }
        else if ( token == "ViewState" )
        {
          onUnexpectedToken( "{", name );
          viewState = readViewState();
        }
        else
        {
          onUnknownToken( "Node Type", token );
        }

        token = getNextToken();
      }

    }
    m_ifs.close();
  }
  return( m_scene );
}

bool  DPAFLoader::onIncompatibleValues( int value0, int value1, const string &node, const string &field0, const string &field1 ) const
{
  return( callback() ? callback()->onIncompatibleValues( m_line, node, field0, value0, field1, value1) : true );
}

template<typename T> bool  DPAFLoader::onInvalidValue( T value, const string &node, const string &field ) const
{
  return( callback() ? callback()->onInvalidValue( m_line, node, field, value ) : true );
}

bool  DPAFLoader::onEmptyToken( const string &tokenType, const string &token ) const
{
  return( callback() ? callback()->onEmptyToken( m_line, tokenType, token ) : true );
}

bool  DPAFLoader::onFileNotFound( const string &file ) const
{
  return( callback() ? callback()->onFileNotFound( file ) : true );
}

bool  DPAFLoader::onFilesNotFound( bool found, const vector<string> &files ) const
{
  return( callback() && !found ? callback()->onFilesNotFound( files ) : true );
}

void  DPAFLoader::onUnexpectedEndOfFile( bool error ) const
{
  if ( callback() && error )
  {
    callback()->onUnexpectedEndOfFile( m_line );
  }
}

void  DPAFLoader::onUnexpectedToken( const string &expected, const string &token ) const
{
  if ( callback() && ( expected != token ) )
  {
    callback()->onUnexpectedToken( m_line, expected, token );
  }
}

void  DPAFLoader::onUnknownToken( const string &context, const string &token ) const
{
  if ( callback() )
  {
    callback()->onUnknownToken( m_line, context, token );
  }
}

bool  DPAFLoader::onUndefinedToken( const string &context, const string &token ) const
{
  return( callback() ? callback()->onUndefinedToken( m_line, context, token ) : true );
}

bool  DPAFLoader::onUnsupportedToken( const string &context, const string &token ) const
{
  return( callback() ? callback()->onUnsupportedToken( m_line, context, token ) : true );
}

BillboardSharedPtr DPAFLoader::readBillboard( const char *name, const std::string & extName )
{
  BillboardSharedPtr billboard;

  string token = getNextToken();

  if ( token == "{" )
  {
    billboard = Billboard::create();
    token = getNextToken();
    while ( token != "}" )
    {
      if ( ! readGroupToken( billboard, token, extName ) )
      {
        if ( token == "alignment" )
        {
          billboard->setAlignment( readAlignment() );
        }
        else if ( token == "rotationAxis" )
        {
          billboard->setRotationAxis( readVector<3,float>( getNextToken() ) );
        }
        else
        {
          onUnknownToken( "Billboard", token );
        }
      }
      token = getNextToken();
    }
    billboard->setName( name );
  }
  else if ( m_billboards.find( token ) != m_billboards.end() )
  {
    billboard = std::static_pointer_cast<dp::sg::core::Billboard>(m_billboards[token]->clone());
    billboard->setName( name );
  }
  else
  {
    onUndefinedToken( "Billboard", token );
  }

  return( billboard );
}

bool  DPAFLoader::readBool( const std::string & token )
{
  return( ( token.empty() ? getNextToken() : token ) == "TRUE" );
}

BufferSharedPtr DPAFLoader::readBuffer()
{
  std::string token = getNextToken();

  std::vector<unsigned char> data;
  BufferSharedPtr buffer;
  if ( token == "{" )
  {
    do {
      token = getNextToken();
      if (token == "type")
      {
        token = getNextToken();
        DP_ASSERT( token == "host" && "host buffers only supported at the moment" );
        buffer = BufferHost::create();
      }
      else if ( token == "data" )
      {
        data.clear();
        readScalarArray<unsigned char>( getNextToken(), data );
      }
      else if (token == "}")
      {
        // just ignore this token
      }
      else
      {
        onUnknownToken( "Billboard", token );
      }
    } while ( token != "}" );
  }
  else
  {
    onUnknownToken( "Billboard", token );
  }

  DP_ASSERT(buffer);
  buffer->setSize( data.size() );
  buffer->setData( 0, data.size(), &data[0] );
  return buffer;
}

bool  DPAFLoader::readCameraToken( CameraSharedPtr const& camera, string & token )
{
  bool b = true;
  if ( token == "direction" )
  {
    m_cameraDirection = readVector<3,float>( getNextToken() );
    m_cameraDirection.normalize();
  }
  else if ( token == "headLights" )
  {
    token = getNextToken();
    onUnexpectedToken( "[", token );
    token = getNextToken();
    while ( token != "]" )
    {
      string lightName = readName( token );
      if ( m_lightSources.find( lightName ) != m_lightSources.end() )
      {
        camera->addHeadLight( m_lightSources[lightName] );
      }
      else
      {
        onUndefinedToken( "Camera.headLights", lightName );
      }
      token = getNextToken();
    }
  }
  else if ( token == "position" )
  {
    camera->setPosition( readVector<3,float>( getNextToken() ) );
  }
  else if ( token == "focusDistance" )
  {
    camera->setFocusDistance( readScalar<float>( getNextToken() ) );
  }
  else if ( token == "upVector" )
  {
    m_cameraUpVector = readVector<3,float>( getNextToken() );
    m_cameraUpVector.normalize();
  }
  else
  {
    b = readObjectToken( camera, token );
  }
  return( b );
}

bool  DPAFLoader::readFrustumCameraToken( FrustumCameraSharedPtr const& camera, string & token )
{
  bool b = true;
  if ( token == "farDistance" )
  {
    camera->setFarDistance( readScalar<float>( getNextToken() ) );
  }
  else if ( token == "nearDistance" )
  {
    camera->setNearDistance( readScalar<float>( getNextToken() ) );
  }
  else if ( token == "windowOffset" )
  {
    camera->setWindowOffset( readVector<2,float>( getNextToken() ) );
  }
  else if ( token == "windowSize" )
  {
    camera->setWindowSize( readVector<2,float>( getNextToken() ) );
  }
  else
  {
    b = readCameraToken( camera, token );
  }
  return( b );
}

SamplerSharedPtr DPAFLoader::readSampler( const string & name )
{
  SamplerSharedPtr sampler;

  string token = getNextToken();

  if ( token == "{" )
  {
    unsigned int cf = 0;
    TextureTarget target = TextureTarget::UNSPECIFIED;
    string textureName;

    sampler = Sampler::create();
    token = getNextToken();
    while ( token != "}" )
    {
      if ( token == "borderColor" )
      {
        sampler->setBorderColor( readVector<4,float>( getNextToken() ) );
      }
      else if ( token == "compareMode" )
      {
        sampler->setCompareMode( readTextureCompareMode() );
      }
      else if ( token == "creationFlags" )
      {
        cf = readScalar<unsigned int>( getNextToken() );
      }
      else if ( token == "magFilterMode" )
      {
        sampler->setMagFilterMode( readTextureMagFilterMode() );
      }
      else if ( token == "minFilterMode" )
      {
        sampler->setMinFilterMode( readTextureMinFilterMode() );
      }
      else if ( token == "textureImage" )
      {
        // textureImage is either the name of a TextureHost, which should be in m_textureImages, or a file name, which should be found in m_searchPaths
        textureName = readName( getNextToken() );
        std::map<std::string,dp::sg::core::TextureHostSharedPtr>::const_iterator it = m_textureImages.find( textureName );
        if ( it != m_textureImages.end() )
        {
          sampler->setTexture( it->second );
        }
        else
        {
          std::string foundFile = m_fileFinder.find( textureName );
          if ( !foundFile.empty() )
          {
            TextureHostSharedPtr textureHost = dp::sg::io::loadTextureHost( foundFile, m_fileFinder );
            if ( textureHost )
            {
              TextureTarget texTarget = textureHost->getTextureTarget();
              // try to set proper Target before assigning
              if ( texTarget == TextureTarget::UNSPECIFIED )
              {
                texTarget = determineTextureTarget( textureHost );
                textureHost->setTextureTarget( target != TextureTarget::UNSPECIFIED ? target : texTarget );
              }
              sampler->setTexture( textureHost );
            }
          }
          else
          {
            onFileNotFound( textureName );
          }
        }
      }
      else if ( token == "textureTarget" )
      {
        target = readTextureTarget();
      }
      else if ( token == "wrapMode" )
      {
        sampler->setWrapMode( TexWrapCoordAxis::S, readTextureWrapMode() );
        sampler->setWrapMode( TexWrapCoordAxis::T, readTextureWrapMode() );
        sampler->setWrapMode( TexWrapCoordAxis::R, readTextureWrapMode() );
      }
      else
      {
        onUnknownToken( "Sampler", token );
      }
      token = getNextToken();
    }
    sampler->setName( name );
  }
  else if ( m_samplers.find( token ) != m_samplers.end() )
  {
    sampler = std::static_pointer_cast<dp::sg::core::Sampler>(m_samplers[token]->clone());
    sampler->setName( name );
  }
  else
  {
    onUndefinedToken( "Sampler", token );
  }

  return( sampler );
}

TextureCompareMode  DPAFLoader::readTextureCompareMode()
{
  string token = getNextToken();
  TextureCompareMode tcm = TextureCompareMode::NONE;
  if ( token == "NONE" )
  {
    tcm = TextureCompareMode::NONE;
  }
  else if ( token == "R_TO_TEXTURE" )
  {
    tcm = TextureCompareMode::R_TO_TEXTURE;
  }
  else
  {
    onUnknownToken( "TextureCompareMode", token );
    DP_ASSERT( !"Unknown texture compare mode" );
  }
  return( tcm );
}

TextureMagFilterMode  DPAFLoader::readTextureMagFilterMode( void )
{
  std::string token = getNextToken();
  TextureMagFilterMode tmfm = TextureMagFilterMode::NEAREST;
  if ( token == "NEAREST" )
  {
    tmfm = TextureMagFilterMode::NEAREST;
  }
  else if ( token == "LINEAR" )
  {
    tmfm = TextureMagFilterMode::LINEAR;
  }
  else
  {
    onUnknownToken( "TextureMagFilterMode", token );
    DP_ASSERT( !"Unknown texture mag filter mode" );
  }
  return( tmfm );
}

TextureMinFilterMode  DPAFLoader::readTextureMinFilterMode( void )
{
  std::string token = getNextToken();
  TextureMinFilterMode tmfm = TextureMinFilterMode::NEAREST;
  if ( token == "NEAREST" )
  {
    tmfm = TextureMinFilterMode::NEAREST;
  }
  else if ( token == "LINEAR" )
  {
    tmfm = TextureMinFilterMode::LINEAR;
  }
  else if ( token == "LINEAR_MIPMAP_LINEAR" )
  {
    tmfm = TextureMinFilterMode::LINEAR_MIPMAP_LINEAR;
  }
  else if ( token == "NEAREST_MIPMAP_NEAREST" )
  {
    tmfm = TextureMinFilterMode::NEAREST_MIPMAP_NEAREST;
  }
  else if ( token == "NEAREST_MIPMAP_LINEAR" )
  {
    tmfm = TextureMinFilterMode::NEAREST_MIPMAP_LINEAR;
  }
  else if ( token == "LINEAR_MIPMAP_NEAREST" )
  {
    tmfm = TextureMinFilterMode::LINEAR_MIPMAP_NEAREST;
  }
  else
  {
    onUnknownToken( "TextureMinFilterMode", token );
    DP_ASSERT( !"Unknown texture min filter mode" );
  }
  return( tmfm );
}

TextureWrapMode DPAFLoader::readTextureWrapMode( void )
{
  std::string token = getNextToken();
  TextureWrapMode twm = TextureWrapMode::REPEAT;
  if ( token == "CLAMP" )
  {
    twm = TextureWrapMode::CLAMP;
  }
  else if ( token == "CLAMP_TO_BORDER" )
  {
    twm = TextureWrapMode::CLAMP_TO_BORDER;
  }
  else if ( token == "CLAMP_TO_EDGE" )
  {
    twm = TextureWrapMode::CLAMP_TO_EDGE;
  }
  else if ( ( token == "MIRROR" ) || ( token == "MIRROR_REPEAT" ) )
  {
    twm = TextureWrapMode::MIRROR_REPEAT;
  }
  else if ( token == "MIRROR_CLAMP" )
  {
    twm = TextureWrapMode::MIRROR_CLAMP;
  }
  else if ( token == "MIRROR_CLAMP_TO_BORDER" )
  {
    twm = TextureWrapMode::MIRROR_CLAMP_TO_BORDER;
  }
  else if ( token == "MIRROR_CLAMP_TO_EDGE" )
  {
    twm = TextureWrapMode::MIRROR_CLAMP_TO_EDGE;
  }
  else if ( token == "REPEAT" )
  {
    twm = TextureWrapMode::REPEAT;
  }
  else
  {
    onUnknownToken( "TextureWrapMode", token );
    DP_ASSERT( !"Unknown texture wrap mode" );
  }
  return( twm );
}

NodeSharedPtr DPAFLoader::readChild( const string & token )
{
  NodeSharedPtr child;
  string name = readName( token );
  if ( m_billboards.find( name ) != m_billboards.end() )
  {
    child = m_billboards[name];
  }
  else if ( m_geoNodes.find( name ) != m_geoNodes.end() )
  {
    child = m_geoNodes[name];
  }
  else if ( m_groups.find( name ) != m_groups.end() )
  {
    child = m_groups[name];
  }
  else if ( m_lightSources.find( name ) != m_lightSources.end() )
  {
    child = m_lightSources[name];
  }
  else if ( m_LODs.find( name ) != m_LODs.end() )
  {
    child = m_LODs[name];
  }
  else if ( m_switches.find( name ) != m_switches.end() )
  {
    child = m_switches[name];
  }
  else if ( m_transforms.find( name ) != m_transforms.end() )
  {
    child = m_transforms[name];
  }
  else
  {
    onUndefinedToken( "child", name );
  }
  return( child );
}

LightSourceSharedPtr DPAFLoader::readLightSource( const string & name )
{
  LightSourceSharedPtr lightSource;

  string token = getNextToken();

  if ( token == "{" )
  {
    lightSource = LightSource::create();
    token = getNextToken();
    while ( token != "}" )
    {
      if ( ! readLightSourceToken( lightSource, token ) )
      {
        onUnknownToken( "LightSource", token );
      }
      token = getNextToken();
    };
    lightSource->setName( name );
  }
  else if ( m_lightSources.find( token ) != m_lightSources.end() )
  {
    lightSource = std::static_pointer_cast<dp::sg::core::LightSource>(m_lightSources[token]->clone());
    lightSource->setName( name );
  }
  else
  {
    onUndefinedToken( "LightSource", token );
  }

  return( lightSource );
}

LightSourceSharedPtr DPAFLoader::readLightSourceReferences( const string & token )
{
  LightSourceSharedPtr lightSource;
  string name = readName( token );
  if ( m_lightSources.find( name ) != m_lightSources.end() )
  {
    lightSource = m_lightSources[name];
  }
  else
  {
    onUndefinedToken( "lightSource", name );
  }
  return( lightSource );
}

void  DPAFLoader::readChildren( GroupSharedPtr const& group )
{
  std::string token = getNextToken();
  onUnexpectedToken( "[", token );
  token = getNextToken();
  while ( token != "]" )
  {
    group->addChild( readChild( token ) );
    token = getNextToken();
  }
}

dp::DataType DPAFLoader::readType( const char *token )
{
  std::string t( token ? token : getNextToken() );
  return( static_cast<dp::DataType>( atoi( t.c_str() ) ) );
}

Billboard::Alignment DPAFLoader::readAlignment()
{
  Billboard::Alignment ba;
  string token( getNextToken() );
  if ( token == "AXIS" )
  {
    ba = Billboard::Alignment::AXIS;
  }
  else if ( token == "SCREEN" )
  {
    ba = Billboard::Alignment::SCREEN;
  }
  else if ( token == "VIEWER" )
  {
    ba = Billboard::Alignment::VIEWER;
  }
  else
  {
    onUnknownToken( "Billboard::Alignment", token );
    ba = Billboard::Alignment::VIEWER;
  }
  return( ba );
}

void DPAFLoader::readEnumArray( const string & t, vector<int> & values, const dp::fx::ParameterSpec & ps )
{
  const dp::fx::EnumSpecSharedPtr & enumSpec = ps.getEnumSpec();
  DP_ASSERT( enumSpec );
  onUnexpectedToken( "[", t );
  string token = getNextToken();
  while ( token != "]" )
  {
    values.push_back( enumSpec->getValue( token ) );
    token = getNextToken();
  }
}

template<unsigned int m, unsigned int n>
Matmnt<m,n,char> DPAFLoader::readEnumMatrix( const string & t, const dp::fx::ParameterSpec & ps )
{
  DP_ASSERT( t == "(" );
  Matmnt<m,n,char> mat;
  unsigned int i=0;
  string token = getNextToken();
  while ( token != ")" )
  {
    DP_ASSERT( i < m );
    mat[i++] = readEnumVector<n>( token, ps );
    token = getNextToken();
  }
  DP_ASSERT( i == m );
  return( mat );
}

template<unsigned int n>
Vecnt<n,char> DPAFLoader::readEnumVector( const string & t, const dp::fx::ParameterSpec & ps )
{
  const dp::fx::EnumSpecSharedPtr & enumSpec = ps.getEnumSpec();
  DP_ASSERT( enumSpec );
  onUnexpectedToken( "(", t );
  Vecnt<n,char> v;
  unsigned int i=0;
  string token = getNextToken();
  while ( token != ")" )
  {
    DP_ASSERT( i < n );
    v[i++] = enumSpec->getValue( token );
    token = getNextToken();
  }
  DP_ASSERT( i == n );
  return( v );
}

GeoNodeSharedPtr DPAFLoader::readGeoNode( const char *name )
{
  GeoNodeSharedPtr geoNode;

  std::string token = getNextToken();

  if ( token == "{" )
  {
    geoNode = GeoNode::create();
    token = getNextToken();
    while ( token != "}" )
    {
      if ( ! readNodeToken( geoNode, token ) )
      {
        if ( token == "primitive" )
        {
          token = getNextToken();
          string name = readName( token );
          if ( m_primitives.find( name ) != m_primitives.end() )
          {
            geoNode->setPrimitive( m_primitives[name] );
          }
          else
          {
            onUndefinedToken( "GeoNode.drawables", name );
          }
        }
        else if ( token == "materialPipeline" )
        {
          token = getNextToken();
          string name = readName( token );

          if ( m_pipelineData.find( name ) != m_pipelineData.end() )
          {
            geoNode->setMaterialPipeline( m_pipelineData[name] );
          }
          else if( name != "NULL" )
          {
            onUndefinedToken( "GeoNode.materialPipeline", name );
          }
        }
        else
        {
          onUnknownToken( "GeoNode", token );
        }
      }
      token = getNextToken();
    };
    geoNode->setName( name );
  }
  else if ( m_geoNodes.find( token ) != m_geoNodes.end() )
  {
    geoNode = std::static_pointer_cast<dp::sg::core::GeoNode>(m_geoNodes[token]->clone());
    geoNode->setName( name );
  }
  else
  {
    onUndefinedToken( "GeoNode", token );
  }

  return( geoNode );
}

GroupSharedPtr DPAFLoader::readGroup( const char *name, const std::string & extName )
{
  GroupSharedPtr group;

  std::string token = getNextToken();

  if ( token == "{" )
  {
    group = Group::create();
    token = getNextToken();
    while ( token != "}" )
    {
      if ( ! readGroupToken( group, token, extName ) )
      {
        onUnknownToken( "Group", token );
      }
      token = getNextToken();
    };
    group->setName( name );
  }
  else if ( m_groups.find( token ) != m_groups.end() )
  {
    group = std::static_pointer_cast<dp::sg::core::Group>(m_groups[token]->clone());
    group->setName( name );
  }
  else
  {
    onUndefinedToken( "Group", token );
  }

  return( group );
}

bool  DPAFLoader::readGroupToken( GroupSharedPtr const& group, const string & token, const string & extName )
{
  bool b = true;
  if ( token == "children" )
  {
    readChildren( group );
  }
  else if ( token == "clipPlanes" )
  {
    string token = getNextToken();
    onUnexpectedToken( "[", token );
    token = getNextToken();
    unsigned int count = 0;
    while ( token != "]" )
    {
      ClipPlaneSharedPtr plane( ClipPlane::create() );
      plane->setNormal(readVector<3,float>( token ));
      plane->setOffset(readScalar<float>( getNextToken() ));
      plane->setEnabled(readBool());

      group->addClipPlane( plane );

      token = getNextToken();
      count++;
    }
  }
  else
  {
    b = readNodeToken( group, token );
  }
  return( b );
}

void DPAFLoader::readImages( TextureHostSharedPtr const& th )
{
  string token = getNextToken();
  onUnexpectedToken( "[", token );
  token = getNextToken();
  while ( token != "]" )
  {
    Image::PixelFormat pf = Image::PixelFormat::UNKNOWN;
    Image::PixelDataType pt = Image::PixelDataType::UNKNOWN;
    unsigned int width(1), height(1), depth(1);
    unsigned char * pixels(NULL);
    vector<const void *> mipmaps;

    onUnexpectedToken( "{", token );
    token = getNextToken();
    while ( token != "}" )
    {
      if ( token == "depth" )
      {
        depth = readScalar<unsigned int>( getNextToken() );
      }
      else if ( token == "height" )
      {
        height = readScalar<unsigned int>( getNextToken() );
      }
      else if ( token == "mipmaps" )
      {
        readMipmaps( width, height, depth, numberOfComponents( pf ), pt, mipmaps );
      }
      else if ( token == "pixelFormat" )
      {
        pf = readPixelFormat();
      }
      else if ( token == "pixelType" )
      {
        pt = readPixelType();
      }
      else if ( token == "pixels" )
      {
        DP_ASSERT( ( pf != Image::PixelFormat::UNKNOWN ) && ( pt != Image::PixelDataType::UNKNOWN ) );
        pixels = readPixels( width * height * depth * numberOfComponents( pf ), pt );
      }
      else if ( token == "width" )
      {
        width = readScalar<unsigned int>( getNextToken() );
      }
      else
      {
        onUnknownToken( "TextureHost", token );
      }
      token = getNextToken();
    }
    DP_ASSERT( pixels );
    // disable coverity errors for pixel==NULL
    // coverity[var_deref_model]
    th->createImage( width, height, depth, pf, pt, pixels, mipmaps );

    delete[] pixels;
    for ( size_t i=0 ; i<mipmaps.size() ; i++ )
    {
      // data was allocated as unsigned char *
      delete reinterpret_cast<const unsigned char *>( mipmaps[i] );
    }

    token = getNextToken();
  }
}

bool  DPAFLoader::readLightSourceToken( LightSourceSharedPtr const& light, const string &token )
{
  bool b = true;
  if ( token == "enabled" )
  {
    light->setEnabled( readBool() );
  }
  else if ( token == "lightPipeline" )
  {
    string name = readName( getNextToken() );
    if ( m_pipelineData.find( name ) != m_pipelineData.end() )
    {
      light->setLightPipeline( m_pipelineData[name] );
    }
    else
    {
      onUndefinedToken( "LightSource.lightPipeline", token );
    }
  }
  else if ( token == "shadowCasting" )
  {
    light->setShadowCasting( readBool() );
  }
  else
  {
    b = readNodeToken( light, token );
  }
  return( b );
}

dp::sg::core::PipelineDataSharedPtr DPAFLoader::readPipelineData( const char * name )
{
  dp::sg::core::PipelineDataSharedPtr pipelineData;
  string token = getNextToken();
  if ( token == "{" )
  {
    string annotation;
    unsigned int hints = 0;
    std::string effectSpecName;
    unsigned int notifyChangeBits = 0;
    vector<ParameterGroupDataSharedPtr> parameterGroupData;
    bool transparent = false;

    token = getNextToken();
    while ( token != "}" )
    {
      if ( !readObjectToken( token, annotation, hints ) )
      {
        if ( token == "effectFile" )
        {
          std::string effectFile = readName( getNextToken() );
          DP_VERIFY( dp::fx::EffectLibrary::instance()->loadEffects( effectFile ) );
        }
        else if ( token == "effectSpec" )
        {
          effectSpecName = readName( getNextToken() );
        }
        else if ( token == "notifyChangeBits" )
        {
          notifyChangeBits = readScalar<unsigned int>( getNextToken() );
        }
        else if ( token == "parameterGroupData" )
        {
          token = getNextToken();
          DP_ASSERT( token == "[" );
          token = getNextToken();
          while ( token != "]" )
          {
            string name = readName( token );
            DP_ASSERT( name != "options" );   // "options" should not be encountered any more
            if ( m_parameterGroupData.find( name ) != m_parameterGroupData.end() )
            {
              parameterGroupData.push_back( m_parameterGroupData[name] );
            }
            token = getNextToken();
          }
        }
        else if ( token == "transparent" )
        {
          transparent = readBool( getNextToken() );
        }
        else
        {
          onUnknownToken( "EffectData", token );
        }
      }
      token = getNextToken();
    }

    DP_ASSERT( !effectSpecName.empty() );
    dp::fx::EffectSpecSharedPtr effectSpec = dp::fx::EffectLibrary::instance()->getEffectSpec( effectSpecName );

    DP_ASSERT( effectSpec );
    pipelineData = dp::sg::core::PipelineData::create( effectSpec );
    pipelineData->setName( name );
    if ( !annotation.empty() )
    {
      pipelineData->setAnnotation( annotation );
    }
    if ( hints )
    {
      pipelineData->setHints( hints );
    }
    for ( size_t i=0 ; i<parameterGroupData.size() ; i++ )
    {
      DP_VERIFY( pipelineData->setParameterGroupData( parameterGroupData[i] ) );
    }
    pipelineData->setTransparent( transparent );
  }
  else if ( m_pipelineData.find( token ) != m_pipelineData.end() )
  {
    pipelineData = std::static_pointer_cast<dp::sg::core::PipelineData>(m_pipelineData[token]->clone());
    pipelineData->setName( name );
  }
  else
  {
    onUndefinedToken( "EffectData", token );
  }
  return( pipelineData );
}

Image::PixelFormat DPAFLoader::readPixelFormat()
{
  Image::PixelFormat pf = Image::PixelFormat::UNKNOWN;
  std::string token = getNextToken();
  if ( token == "COLOR_INDEX" )
  {
    pf = Image::PixelFormat::COLOR_INDEX;
  }
  else if ( token == "RGB" )
  {
    pf = Image::PixelFormat::RGB;
  }
  else if ( token == "RGBA" )
  {
    pf = Image::PixelFormat::RGBA;
  }
  else if ( token == "BGR" )
  {
    pf = Image::PixelFormat::BGR;
  }
  else if ( token == "BGRA" )
  {
    pf = Image::PixelFormat::BGRA;
  }
  else if ( token == "LUMINANCE" )
  {
    pf = Image::PixelFormat::LUMINANCE;
  }
  else if ( token == "PixelFormat::LUMINANCE_ALPHA" )
  {
    pf = Image::PixelFormat::LUMINANCE_ALPHA;
  }
  else if ( token == "PixelFormat::ALPHA" )
  {
    pf = Image::PixelFormat::ALPHA;
  }
  else if ( token == "PixelFormat::DEPTH_COMPONENT" )
  {
    pf = Image::PixelFormat::DEPTH_COMPONENT;
  }
  else if ( token == "PixelFormat::DEPTH_STENCIL" )
  {
    pf = Image::PixelFormat::DEPTH_STENCIL;
  }
  else if ( token == "PixelFormat::INTEGER_ALPHA" )
  {
    pf = Image::PixelFormat::INTEGER_ALPHA;
  }
  else if ( token == "PixelFormat::INTEGER_LUMINANCE" )
  {
    pf = Image::PixelFormat::INTEGER_LUMINANCE;
  }
  else if ( token == "PixelFormat::INTEGER_LUMINANCE_ALPHA" )
  {
    pf = Image::PixelFormat::INTEGER_LUMINANCE_ALPHA;
  }
  else if ( token == "PixelFormat::INTEGER_RGB" )
  {
    pf = Image::PixelFormat::INTEGER_RGB;
  }
  else if ( token == "PixelFormat::INTEGER_BGR" )
  {
    pf = Image::PixelFormat::INTEGER_BGR;
  }
  else if ( token == "PixelFormat::INTEGER_RGBA" )
  {
    pf = Image::PixelFormat::INTEGER_RGBA;
  }
  else if ( token == "PixelFormat::INTEGER_BGRA" )
  {
    pf = Image::PixelFormat::INTEGER_BGRA;
  }
  else if ( token == "PixelFormat::COMPRESSED_LUMINANCE_LATC1" )
  {
    pf = Image::PixelFormat::COMPRESSED_LUMINANCE_LATC1;
  }
  else if ( token == "PixelFormat::COMPRESSED_SIGNED_LUMINANCE_LATC1" )
  {
    pf = Image::PixelFormat::COMPRESSED_SIGNED_LUMINANCE_LATC1;
  }
  else if ( token == "PixelFormat::COMPRESSED_LUMINANCE_ALPHA_LATC2" )
  {
    pf = Image::PixelFormat::COMPRESSED_LUMINANCE_ALPHA_LATC2;
  }
  else if ( token == "PixelFormat::COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2" )
  {
    pf = Image::PixelFormat::COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2;
  }
  else if ( token == "PixelFormat::COMPRESSED_RED_RGTC1" )
  {
    pf = Image::PixelFormat::COMPRESSED_RED_RGTC1;
  }
  else if ( token == "PixelFormat::COMPRESSED_SIGNED_RED_RGTC1" )
  {
    pf = Image::PixelFormat::COMPRESSED_SIGNED_RED_RGTC1;
  }
  else if ( token == "PixelFormat::COMPRESSED_RG_RGTC2" )
  {
    pf = Image::PixelFormat::COMPRESSED_RG_RGTC2;
  }
  else if ( token == "PixelFormat::COMPRESSED_SIGNED_RG_RGTC2" )
  {
    pf = Image::PixelFormat::COMPRESSED_SIGNED_RG_RGTC2;
  }
  else if ( token == "PixelFormat::COMPRESSED_RGB_DXT1" )
  {
    pf = Image::PixelFormat::COMPRESSED_RGB_DXT1;
  }
  else if ( token == "PixelFormat::COMPRESSED_RGBA_DXT1" )
  {
    pf = Image::PixelFormat::COMPRESSED_RGBA_DXT1;
  }
  else if ( token == "PixelFormat::COMPRESSED_RGBA_DXT3" )
  {
    pf = Image::PixelFormat::COMPRESSED_RGBA_DXT3;
  }
  else if ( token == "PixelFormat::COMPRESSED_RGBA_DXT5" )
  {
    pf = Image::PixelFormat::COMPRESSED_RGBA_DXT5;
  }
  else if ( token == "PixelFormat::COMPRESSED_SRGB_DXT1" )
  {
    pf = Image::PixelFormat::COMPRESSED_SRGB_DXT1;
  }
  else if ( token == "PixelFormat::COMPRESSED_SRGBA_DXT1" )
  {
    pf = Image::PixelFormat::COMPRESSED_SRGBA_DXT1;
  }
  else if ( token == "PixelFormat::COMPRESSED_SRGBA_DXT3" )
  {
    pf = Image::PixelFormat::COMPRESSED_SRGBA_DXT3;
  }
  else if ( token == "PixelFormat::COMPRESSED_SRGBA_DXT5" )
  {
    pf = Image::PixelFormat::COMPRESSED_SRGBA_DXT5;
  }
  else
  {
    onUnknownToken( "Image::PixelFormat", token );
  }
  return( pf );
}

Image::PixelDataType DPAFLoader::readPixelType()
{
  Image::PixelDataType pt = Image::PixelDataType::UNKNOWN;
  std::string token = getNextToken();
  if ( token == "BYTE" )
  {
    pt = Image::PixelDataType::BYTE;
  }
  else if ( token == "UNSIGNED_BYTE" )
  {
    pt = Image::PixelDataType::UNSIGNED_BYTE;
  }
  else if ( token == "SHORT" )
  {
    pt = Image::PixelDataType::SHORT;
  }
  else if ( token == "UNSIGNED_SHORT" )
  {
    pt = Image::PixelDataType::UNSIGNED_SHORT;
  }
  else if ( token == "INT" )
  {
    pt = Image::PixelDataType::INT;
  }
  else if ( token == "UNSIGNED_INT" )
  {
    pt = Image::PixelDataType::UNSIGNED_INT;
  }
  else if ( token == "FLOAT" )
  {
    pt = Image::PixelDataType::FLOAT;
  }
  else if ( token == "HALF" )
  {
    pt = Image::PixelDataType::HALF;
  }
  else
  {
    onUnknownToken( "Image::PixelDataType", token );
  }
  return( pt );
}

template<typename T>
void DPAFLoader::readPixelComponent( const std::string & token, T & pc )
{
  pc = readScalar<T>( token );
}

#if defined(HAVE_HALF_FLOAT)
template<>
void DPAFLoader::readPixelComponent( const string & token, half & pc )
{
  DP_ASSERT( false );
  pc = half(readScalar<float>( token ));
}
#endif

template<typename T>
unsigned char * DPAFLoader::readPixels( unsigned int nov )
{
  T * pixels = new T[nov];
  std::string token = getNextToken();
  onUnexpectedToken( "[", token );
  token = getNextToken();
  for ( unsigned int i=0 ; i<nov && ( token != "]" ) ; i++ )
  {
    readPixelComponent( token, pixels[i] );
    token = getNextToken();
  }
  onUnexpectedToken( "]", token );
  return( (unsigned char *)pixels );
}

unsigned char * DPAFLoader::readPixels( unsigned int nov,
                                        Image::PixelDataType pt )
{
  unsigned char * p(NULL);
  switch( pt )
  {
    case Image::PixelDataType::BYTE :
      p = readPixels<char>( nov );
      break;
    case Image::PixelDataType::UNSIGNED_BYTE :
      p = readPixels<unsigned char>( nov );
      break;
    case Image::PixelDataType::SHORT :
      p = readPixels<short>( nov );
      break;
    case Image::PixelDataType::UNSIGNED_SHORT :
      p = readPixels<unsigned short>( nov );
      break;
    case Image::PixelDataType::INT :
      p = readPixels<int>( nov );
      break;
    case Image::PixelDataType::UNSIGNED_INT :
      p = readPixels<unsigned int>( nov );
      break;
    case Image::PixelDataType::FLOAT :
      p = readPixels<float>( nov );
      break;
#if defined(HAVE_HALF_FLOAT)
    case Image::PixelDataType::HALF :
      p = readPixels<half>( nov );
      break;
#endif
    default :
      DP_ASSERT( false );
      break;
  }
  return( p );
}

LODSharedPtr DPAFLoader::readLOD( const char *name, const std::string & extName )
{
  LODSharedPtr lod;

  std::string token = getNextToken();

  if ( token == "{" )
  {
    lod = LOD::create();
    token = getNextToken();
    while ( token != "}" )
    {
      if ( ! readGroupToken( lod, token, extName ) )
      {
        if ( token == "center" )
        {
          lod->setCenter( readVector<3,float>( getNextToken() ) );
        }
        else if ( token == "ranges" )
        {
          vector<float> ranges;
          readScalarArray<float>( getNextToken(), ranges );
          lod->setRanges( &ranges[0], dp::checked_cast<unsigned int>(ranges.size()) );
        }
        else
        {
          onUnknownToken( "LOD", token );
        }
      }
      token = getNextToken();
    }
    lod->setName( name );
  }
  else if ( m_LODs.find( token ) != m_LODs.end() )
  {
    lod = std::static_pointer_cast<dp::sg::core::LOD>(m_LODs[token]->clone());
    lod->setName( name );
  }
  else
  {
    onUndefinedToken( "LOD", token );
  }

  return( lod );
}

template<unsigned int m, unsigned int n, typename T>
Matmnt<m,n,T> DPAFLoader::readMatrix( const string & t )
{
  DP_ASSERT( t == "(" );
  Matmnt<m,n,T> mat;
  unsigned int i=0;
  string token = getNextToken();
  while ( token != ")" )
  {
    DP_ASSERT( i < m );
    mat[i++] = readVector<n,T>( token );
    token = getNextToken();
  }
  DP_ASSERT( i == m );
  return( mat );
}

MatrixCameraSharedPtr DPAFLoader::readMatrixCamera( const char *name )
{
  MatrixCameraSharedPtr camera;

  string token = getNextToken();
  Mat44f projection(cIdentity44f), inverse(cIdentity44f), worldToView(cIdentity44f), viewToWorld(cIdentity44f);
  bool   p(false), i(false), wtv(false), vtw(false);

  if ( token == "{" )
  {
    camera = MatrixCamera::create();
    token = getNextToken();
    while ( token != "}" )
    {
      if ( ! readCameraToken( camera, token ) )
      {
        if ( token == "inverseMatrix" )
        {
          inverse = readMatrix<4,4,float>( getNextToken() );
          i = true;
        }
        else if ( token == "projectionMatrix" )
        {
          projection = readMatrix<4,4,float>( getNextToken() );
          p = true;
        }
        else if ( token == "worldToViewMatrix" )
        {
          worldToView = readMatrix<4,4,float>( getNextToken() );
          wtv = true;
        }
        else if ( token == "viewToWorldMatrix" )
        {
          viewToWorld = readMatrix<4,4,float>( getNextToken() );
          vtw = true;
        }
        else
        {
          onUnknownToken( "MatrixCamera", token );
        }
      }
      token = getNextToken();
    };
    if ( p && i )
    {
      DP_ASSERT( isIdentity( projection * inverse ) );
    }
    else if ( p )
    {
      inverse = projection;
      DP_VERIFY( inverse.invert() );
    }
    else if ( i )
    {
      projection = inverse;
      DP_VERIFY( projection.invert() );
    }
    camera->setMatrices( projection, inverse );
    camera->setName( name );

    if ( wtv && vtw )
    {
      DP_ASSERT( isIdentity( worldToView * viewToWorld ) );
    }
    else if ( wtv )
    {
      viewToWorld = worldToView;
      DP_VERIFY( viewToWorld.invert() );
    }
    else if ( vtw )
    {
      worldToView = viewToWorld;
      DP_VERIFY( worldToView.invert() );
    }
    camera->setWorldToViewMatrix( worldToView );
    camera->setViewToWorldMatrix( viewToWorld );
  }
  else if ( m_matrixCameras.find( token ) != m_matrixCameras.end() )
  {
    camera = std::static_pointer_cast<dp::sg::core::MatrixCamera>(m_matrixCameras[token]->clone());
    camera->setName( name );
  }
  else
  {
    onUndefinedToken( "MatrixCamera", token );
  }

  return( camera );
}

void DPAFLoader::readMipmaps( unsigned int width, unsigned int height, unsigned int depth
                             , unsigned int noc, Image::PixelDataType pt, vector<const void *> & mipmaps )
{
  std::string token = getNextToken();
  onUnexpectedToken( "{", token );
  token = getNextToken();
  while ( token != "}" )
  {
    DP_ASSERT( 1 < width * height * depth );
    width = ( width == 1 ) ? 1 : width / 2;
    height = ( height == 1 ) ? 1 : height / 2;
    depth = ( depth == 1 ) ? 1 : depth / 2;
    mipmaps.push_back( readPixels( width * height * depth * noc, pt ) );
    token = getNextToken();
  }
}

string  DPAFLoader::readName( const string & t )
{
  string name;
  string token = t;
  if ( token[0] == '"' )
  {
    name = &token[1];
    while ( token[token.length()-1] != '"' )
    {
      token = getNextToken();
      name += " ";
      name += token;
    }
    name.replace( name.find( "\"" ), 1, "" );
  }
  else
  {
    name = token;
  }

  return name;
}

bool DPAFLoader::readObjectToken( const string & token, std::string & annotation, unsigned int & hints )
{
  annotation.clear();
  hints = 0;

  if ( token == "annotation" )
  {
    annotation = readName( getNextToken() );
    return true;
  }
  else if ( token == "hints" )
  {
    hints = readScalar<unsigned int>( getNextToken() );
    return true;
  }

  return false;
}

bool DPAFLoader::readObjectToken( ObjectSharedPtr const& object, const string & token )
{
  std::string annotation;
  unsigned int hints;

  bool result = readObjectToken( token, annotation, hints );

  if( result )
  {
    if( annotation.size() )
    {
      object->setAnnotation( annotation );
    }
    else if( hints )
    {
      object->setHints( hints );
    }
  }

  return result;
}

bool  DPAFLoader::readNodeToken( NodeSharedPtr const& node, const string & token )
{
  // there are no specific flags in node any more, but leave this
  // method here in case we add some more some day
  return readObjectToken( node, token );
}

ParallelCameraSharedPtr DPAFLoader::readParallelCamera( const char *name )
{
  ParallelCameraSharedPtr camera;

  m_cameraDirection = Vec3f( 0.0f, 0.0f, -1.0f );
  m_cameraUpVector = Vec3f( 0.0f, 1.0f, 0.0f );

  string token = getNextToken();

  if ( token == "{" )
  {
    camera = ParallelCamera::create();
    token = getNextToken();
    while ( token != "}" )
    {
      if ( ! readFrustumCameraToken( camera, token ) )
      {
        onUnknownToken( "ParallelCamera", token );
      }
      token = getNextToken();
    };
    m_cameraUpVector.orthonormalize( m_cameraDirection );
    camera->setOrientation( m_cameraDirection, m_cameraUpVector );
    camera->setName( name );
  }
  else if ( m_parallelCameras.find( token ) != m_parallelCameras.end() )
  {
    camera = std::static_pointer_cast<dp::sg::core::ParallelCamera>(m_parallelCameras[token]->clone());
    camera->setName( name );
  }
  else
  {
    onUndefinedToken( "ParallelCamera", token );
  }

  return( camera );
}

void DPAFLoader::readParameter( ParameterGroupDataSharedPtr const& pgd, dp::fx::ParameterGroupSpec::iterator it )
{
  unsigned int type = it->first.getType();
  if ( type & dp::fx::PT_SCALAR_TYPE_MASK )
  {
    switch( type & dp::fx::PT_SCALAR_TYPE_MASK )
    {
      case dp::fx::PT_BOOL :
        readParameterT<bool>( pgd, it );
        break;
      case dp::fx::PT_ENUM :
        readParameterEnum( pgd, it );
        break;
      case dp::fx::PT_INT8 :
        readParameterT<char>( pgd, it );
        break;
      case dp::fx::PT_UINT8 :
        readParameterT<unsigned char>( pgd, it );
        break;
      case dp::fx::PT_INT16 :
        readParameterT<short>( pgd, it );
        break;
      case dp::fx::PT_UINT16 :
        readParameterT<unsigned short>( pgd, it );
        break;
      case dp::fx::PT_FLOAT32 :
        readParameterT<float>( pgd, it );
        break;
      case dp::fx::PT_INT32 :
        readParameterT<int>( pgd, it );
        break;
      case dp::fx::PT_UINT32 :
        readParameterT<unsigned int>( pgd, it );
        break;
      case dp::fx::PT_FLOAT64 :
        readParameterT<double>( pgd, it );
        break;
      case dp::fx::PT_INT64 :
        readParameterT<long long>( pgd, it );
        break;
      case dp::fx::PT_UINT64 :
        readParameterT<unsigned long long>( pgd, it );
        break;
      default :
        DP_ASSERT( false );
        break;
    }
  }
  else
  {
    DP_ASSERT( type & dp::fx::PT_POINTER_TYPE_MASK );
    string name = readName( getNextToken() );
    switch( type & dp::fx::PT_POINTER_TYPE_MASK )
    {
      case dp::fx::PT_BUFFER_PTR :
        if ( m_buffers.find( name ) != m_buffers.end() )
        {
          pgd->setParameter( it, m_buffers[name] );
        }
        else
        {
          onUnknownToken( "ParameterGroupData.buffer", name );
        }
        break;
      case dp::fx::PT_SAMPLER_PTR :
        if ( m_samplers.find( name ) != m_samplers.end() )
        {
          pgd->setParameter( it, m_samplers[name] );
        }
        else
        {
          onUnknownToken( "ParameterGroupData.sampler", name );
        }
        break;
      default :
        DP_ASSERT( false );
        break;
    }
  }
}

void DPAFLoader::readParameterEnum( ParameterGroupDataSharedPtr const& pgd, dp::fx::ParameterGroupSpec::iterator it )
{
  unsigned int type = it->first.getType();
  DP_ASSERT( type & dp::fx::PT_SCALAR_TYPE_MASK );
  if ( type & dp::fx::PT_SCALAR_MODIFIER_MASK )
  {
    switch( type & dp::fx::PT_SCALAR_MODIFIER_MASK )
    {
      case dp::fx::PT_VECTOR2 :
        readParameterEnumVN<2>( pgd, it );
        break;
      case dp::fx::PT_VECTOR3 :
        readParameterEnumVN<3>( pgd, it );
        break;
      case dp::fx::PT_VECTOR4 :
        readParameterEnumVN<4>( pgd, it );
        break;
      case dp::fx::PT_MATRIX2x2 :
        readParameterEnumMN<2,2>( pgd, it );
        break;
      case dp::fx::PT_MATRIX2x3 :
        readParameterEnumMN<2,3>( pgd, it );
        break;
      case dp::fx::PT_MATRIX2x4 :
        readParameterEnumMN<2,4>( pgd, it );
        break;
      case dp::fx::PT_MATRIX3x2 :
        readParameterEnumMN<3,2>( pgd, it );
        break;
      case dp::fx::PT_MATRIX3x3 :
        readParameterEnumMN<3,3>( pgd, it );
        break;
      case dp::fx::PT_MATRIX3x4 :
        readParameterEnumMN<3,4>( pgd, it );
        break;
      case dp::fx::PT_MATRIX4x2 :
        readParameterEnumMN<4,2>( pgd, it );
        break;
      case dp::fx::PT_MATRIX4x3 :
        readParameterEnumMN<4,3>( pgd, it );
        break;
      case dp::fx::PT_MATRIX4x4 :
        readParameterEnumMN<4,4>( pgd, it );
        break;
    }
  }
  else
  {
    string token = getNextToken();
    if ( it->first.getArraySize() )
    {
      vector<int> values;
      readEnumArray( token, values, it->first );
      pgd->setParameterArray( it, values );
    }
    else
    {
      DP_ASSERT( it->first.getEnumSpec() );
      pgd->setParameter( it, it->first.getEnumSpec()->getValue( token ) );
    }
  }
}

template<unsigned int m, unsigned int n>
void DPAFLoader::readParameterEnumMN( ParameterGroupDataSharedPtr const& pgd, dp::fx::ParameterGroupSpec::iterator it )
{
  string token = getNextToken();
  if ( it->first.getArraySize() )
  {
    vector<Matmnt<m,n,char> > values;
    DP_ASSERT( token == "{" );
    token = getNextToken();
    while ( token != "}" )
    {
      values.push_back( readEnumMatrix<m,n>( token, it->first ) );
      token = getNextToken();
    }
    pgd->setParameterArray( it, values );
  }
  else
  {
    pgd->setParameter( it, readEnumMatrix<m,n>( token, it->first ) );
  }
}

template<unsigned int n>
void DPAFLoader::readParameterEnumVN( ParameterGroupDataSharedPtr const& pgd, dp::fx::ParameterGroupSpec::iterator it )
{
  string token = getNextToken();
  if ( it->first.getArraySize() )
  {
    vector<Vecnt<n,char> > values;
    DP_ASSERT( token == "[" );
    token = getNextToken();
    while ( token != "]" )
    {
      values.push_back( readEnumVector<n>( token, it->first ) );
      token = getNextToken();
    }
    pgd->setParameterArray( it, values );
  }
  else
  {
    pgd->setParameter( it, readEnumVector<n>( token, it->first ) );
  }
}

ParameterGroupDataSharedPtr DPAFLoader::readParameterGroupData( const char * name )
{
  ParameterGroupDataSharedPtr parameterGroupData;
  string token = getNextToken();
  if ( token == "{" )
  {
    string annotation;
    unsigned int hints = 0;
    dp::fx::ParameterGroupSpecSharedPtr pgs;
#if !defined(NDEBUG)
    bool encounteredEffectFile = false;
#endif

    token = getNextToken();
    while ( token != "}" )
    {
      if ( !readObjectToken( token, annotation, hints ) )
      {
        if ( token == "effectFile" )
        {
          DP_ASSERT( !encounteredEffectFile );
#if !defined(NDEBUG)
          encounteredEffectFile = true;
#endif
          std::string effectFile = readName( getNextToken() );
          DP_VERIFY( dp::fx::EffectLibrary::instance()->loadEffects( effectFile, m_fileFinder ) );
        }
        else if ( token == "parameterGroupSpec" )
        {
          DP_ASSERT( encounteredEffectFile );
          string effectName = readName( getNextToken() );
          string groupName = readName( getNextToken() );
          const dp::fx::EffectSpecSharedPtr & effectSpec = dp::fx::EffectLibrary::instance()->getEffectSpec( effectName );
          if ( effectSpec )
          {
            dp::fx::EffectSpec::ParameterGroupSpecsContainer::const_iterator it = effectSpec->findParameterGroupSpec( groupName );
            if ( it != effectSpec->endParameterGroupSpecs() )
            {
              pgs = *it;
              parameterGroupData = ParameterGroupData::create( pgs );
            }
          }
        }
        else if ( token == "parameters" )
        {
          DP_ASSERT( encounteredEffectFile );
          DP_ASSERT( parameterGroupData );

          token = getNextToken();
          DP_ASSERT( token == "[" );
          token = getNextToken();
          while ( token != "]" )
          {
            string name = readName( token );
            dp::fx::ParameterGroupSpec::iterator it = pgs->findParameterSpec( name );
            DP_ASSERT( it != pgs->endParameterSpecs() );
            readParameter( parameterGroupData, it );
            token = getNextToken();
          }
        }
        else
        {
          onUnknownToken( "ParameterGroupData", token );
        }
      }
      token = getNextToken();
    }
    DP_ASSERT( parameterGroupData );
    parameterGroupData->setName( name );
    if ( !annotation.empty() )
    {
      parameterGroupData->setAnnotation( annotation );
    }
    if ( hints )
    {
      parameterGroupData->setHints( hints );
    }
  }
  else if ( m_parameterGroupData.find( token ) != m_parameterGroupData.end() )
  {
    parameterGroupData = std::static_pointer_cast<dp::sg::core::ParameterGroupData>(m_parameterGroupData[token]->clone());
    parameterGroupData->setName( name );
  }
  else
  {
    onUndefinedToken( "ParameterGroupData", token );
  }

  return( parameterGroupData );
}

template<unsigned int m, unsigned int n, typename T>
void DPAFLoader::readParameterMNT( ParameterGroupDataSharedPtr const& pgd, dp::fx::ParameterGroupSpec::iterator it )
{
  string token = getNextToken();
  if ( it->first.getArraySize() )
  {
    vector<Matmnt<m,n,T> > values;
    DP_ASSERT( token == "{" );
    token = getNextToken();
    while ( token != "}" )
    {
      values.push_back( readMatrix<m,n,T>( token ) );
      token = getNextToken();
    }
    pgd->setParameterArray( it, values );
  }
  else
  {
    pgd->setParameter( it, readMatrix<m,n,T>( token ) );
  }
}

template<typename T>
void DPAFLoader::readParameterT( ParameterGroupDataSharedPtr const& pgd, dp::fx::ParameterGroupSpec::iterator it )
{
  unsigned int type = it->first.getType();
  DP_ASSERT( type & dp::fx::PT_SCALAR_TYPE_MASK );
  if ( type & dp::fx::PT_SCALAR_MODIFIER_MASK )
  {
    switch( type & dp::fx::PT_SCALAR_MODIFIER_MASK )
    {
      case dp::fx::PT_VECTOR2 :
        readParameterVNT<2,T>( pgd, it );
        break;
      case dp::fx::PT_VECTOR3 :
        readParameterVNT<3,T>( pgd, it );
        break;
      case dp::fx::PT_VECTOR4 :
        readParameterVNT<4,T>( pgd, it );
        break;
      case dp::fx::PT_MATRIX2x2 :
        readParameterMNT<2,2,T>( pgd, it );
        break;
      case dp::fx::PT_MATRIX2x3 :
        readParameterMNT<2,3,T>( pgd, it );
        break;
      case dp::fx::PT_MATRIX2x4 :
        readParameterMNT<2,4,T>( pgd, it );
        break;
      case dp::fx::PT_MATRIX3x2 :
        readParameterMNT<3,2,T>( pgd, it );
        break;
      case dp::fx::PT_MATRIX3x3 :
        readParameterMNT<3,3,T>( pgd, it );
        break;
      case dp::fx::PT_MATRIX3x4 :
        readParameterMNT<3,4,T>( pgd, it );
        break;
      case dp::fx::PT_MATRIX4x2 :
        readParameterMNT<4,2,T>( pgd, it );
        break;
      case dp::fx::PT_MATRIX4x3 :
        readParameterMNT<4,3,T>( pgd, it );
        break;
      case dp::fx::PT_MATRIX4x4 :
        readParameterMNT<4,4,T>( pgd, it );
        break;
    }
  }
  else
  {
    string token = getNextToken();
    if ( it->first.getArraySize() )
    {
      vector<T> values;
      readScalarArray<T>( token, values );
      pgd->setParameterArray( it, values );
    }
    else
    {
      pgd->setParameter( it, readScalar<T>( token ) );
    }
  }
}

unsigned int DPAFLoader::readParameterType()
{
  unsigned int type = 0;
  string name = getNextToken();
  if ( name == "Buffer" )
  {
    type = dp::fx::PT_BUFFER_PTR;
  }
  else if ( name.find( "Sampler" ) == 0 )
  {
    type = dp::fx::PT_SAMPLER_PTR;
  }
  else
  {
    static pair<string,unsigned int> prefixToType[] =
    {
      make_pair( string( "bool" ), dp::fx::PT_BOOL ),
      make_pair( string( "enum" ), dp::fx::PT_ENUM ),
      make_pair( string( "char" ), dp::fx::PT_INT8 ),
      make_pair( string( "uchar" ), dp::fx::PT_UINT8 ),
      make_pair( string( "short" ), dp::fx::PT_INT16 ),
      make_pair( string( "ushort" ), dp::fx::PT_UINT16 ),
      make_pair( string( "float" ), dp::fx::PT_FLOAT32 ),
      make_pair( string( "int" ), dp::fx::PT_INT32 ),
      make_pair( string( "uint" ), dp::fx::PT_UINT32 ),
      make_pair( string( "double" ), dp::fx::PT_FLOAT64 ),
      make_pair( string( "longlong" ), dp::fx::PT_INT64 ),
      make_pair( string( "ulonglong" ), dp::fx::PT_UINT64 )
    };
    static const unsigned int prefixN = sizeof(prefixToType)/sizeof(pair<string,unsigned int>);
    string::size_type idx = name.find_first_of( "0123456789" );
    string prefix = name.substr( 0, idx );
    for ( unsigned int i=0 ; i<prefixN ; i++ )
    {
      if ( prefix == prefixToType[i].first )
      {
        type = prefixToType[i].second;
        break;
      }
    }
    DP_ASSERT( type & dp::fx::PT_SCALAR_TYPE_MASK );

    if ( idx != string::npos )
    {
      static pair<string,unsigned int> postfixToType[] =
      {
        make_pair( string( "2" ), dp::fx::PT_VECTOR2 ),
        make_pair( string( "3" ), dp::fx::PT_VECTOR3 ),
        make_pair( string( "4" ), dp::fx::PT_VECTOR4 ),
        make_pair( string( "2x2" ), dp::fx::PT_MATRIX2x2 ),
        make_pair( string( "2x3" ), dp::fx::PT_MATRIX2x3 ),
        make_pair( string( "2x4" ), dp::fx::PT_MATRIX2x4 ),
        make_pair( string( "3x2" ), dp::fx::PT_MATRIX3x2 ),
        make_pair( string( "3x3" ), dp::fx::PT_MATRIX3x3 ),
        make_pair( string( "3x4" ), dp::fx::PT_MATRIX3x4 ),
        make_pair( string( "4x2" ), dp::fx::PT_MATRIX4x2 ),
        make_pair( string( "4x3" ), dp::fx::PT_MATRIX4x3 ),
        make_pair( string( "4x4" ), dp::fx::PT_MATRIX4x4 ),
      };
      static const unsigned int postfixN = sizeof(postfixToType)/sizeof(pair<string,unsigned int>);
      string postfix = name.substr( idx );
      for ( unsigned int i=0 ; i<postfixN ; i++ )
      {
        if ( postfix == postfixToType[i].first )
        {
          type |= postfixToType[i].second;
          break;
        }
      }
      DP_ASSERT( type & dp::fx::PT_SCALAR_MODIFIER_MASK );
    }
  }
  return( type );
}

template<unsigned int n, typename T>
void DPAFLoader::readParameterVNT( ParameterGroupDataSharedPtr const& pgd, dp::fx::ParameterGroupSpec::iterator it )
{
  string token = getNextToken();
  if ( it->first.getArraySize() )
  {
    vector<Vecnt<n,T> > values;
    DP_ASSERT( token == "[" );
    token = getNextToken();
    while ( token != "]" )
    {
      values.push_back( readVector<n,T>( token ) );
      token = getNextToken();
    }
    pgd->setParameterArray( it, values );
  }
  else
  {
    pgd->setParameter( it, readVector<n,T>( token ) );
  }
}

PerspectiveCameraSharedPtr DPAFLoader::readPerspectiveCamera( const char *name )
{
  PerspectiveCameraSharedPtr camera;

  m_cameraDirection = Vec3f( 0.0f, 0.0f, -1.0f );
  m_cameraUpVector = Vec3f( 0.0f, 1.0f, 0.0f );

  string token = getNextToken();

  if ( token == "{" )
  {
    camera = PerspectiveCamera::create();
    token = getNextToken();
    while ( token != "}" )
    {
      if ( ! readFrustumCameraToken( camera, token ) )
      {
        onUnknownToken( "PerspectiveCamera", token );
      }
      token = getNextToken();
    };
    m_cameraUpVector.orthonormalize( m_cameraDirection );
    camera->setOrientation( m_cameraDirection, m_cameraUpVector );
    camera->setName( name );
  }
  else if ( m_perspectiveCameras.find( token ) != m_perspectiveCameras.end() )
  {
    camera = std::static_pointer_cast<dp::sg::core::PerspectiveCamera>(m_perspectiveCameras[token]->clone());
    camera->setName( name );
  }
  else
  {
    onUndefinedToken( "PerspectiveCamera", token );
  }

  return( camera );
}

IndexSetSharedPtr DPAFLoader::readIndexSet( const char * name )
{
  IndexSetSharedPtr iset;

  string token = getNextToken();

  if ( token == "{" )
  {
    iset = IndexSet::create();
    unsigned int numberOfIndices = 0;
    token = getNextToken();
    while ( token != "}" )
    {
      if ( token == "dataType" )
      {
        unsigned int val = readScalar<unsigned int>( getNextToken() );
        iset->setIndexDataType( static_cast<dp::DataType>(val) );
      }
      else if ( token == "primitiveRestartIndex" )
      {
        unsigned int val = readScalar<unsigned int>( getNextToken() );
        iset->setPrimitiveRestartIndex( val );
      }
      else if ( token == "numberOfIndices" )
      {
        numberOfIndices = readScalar<unsigned int>( getNextToken() );
      }
      else if ( token == "indices" )
      {
        DP_VERIFY( getNextToken() == "[" );

        // assure we know count first, which is stored in userdata
        DP_ASSERT( numberOfIndices );

        BufferHostSharedPtr buffer = BufferHost::create();
        buffer->setSize( dp::getSizeOf( iset->getIndexDataType() ) * numberOfIndices );

        Buffer::DataWriteLock writer( buffer, Buffer::MapMode::WRITE );
        unsigned char * bufPtr = writer.getPtr<unsigned char>();

        // move to first index
        token = getNextToken();

        unsigned int numReadIndices = 0;
        while( token != "]" )
        {
          switch( iset->getIndexDataType() )
          {
            case dp::DataType::UNSIGNED_INT_32:
              *((unsigned int *)bufPtr) = readScalar<unsigned int>( token );
              bufPtr += sizeof( unsigned int );
              break;

            case dp::DataType::UNSIGNED_INT_16:
              *((unsigned short *)bufPtr) = readScalar<unsigned short>( token );
              bufPtr += sizeof( unsigned short );
              break;

            case dp::DataType::UNSIGNED_INT_8:
              *bufPtr++ = readScalar<unsigned char>( token );
              break;
          }

          numReadIndices++;
          token = getNextToken();
        }

        DP_ASSERT( numReadIndices == numberOfIndices );

        iset->setBuffer( buffer, numberOfIndices, iset->getIndexDataType(), iset->getPrimitiveRestartIndex() );
      }
      token = getNextToken();
    }
    iset->setName( name );
  }
  else if ( m_indexSets.find( token ) != m_indexSets.end() )
  {
    iset = std::static_pointer_cast<dp::sg::core::IndexSet>(m_indexSets[token]->clone());
    iset->setName( name );
  }
  else
  {
    onUndefinedToken( "IndexSet", token );
  }

  return( iset );
}

PatchesMode DPAFLoader::readPatchesMode()
{
  string token = getNextToken();
  if ( token == "Triangles" )
  {
    return( PatchesMode::TRIANGLES );
  }
  else if ( token == "Quads" )
  {
    return( PatchesMode::QUADS );
  }
  else if ( token == "Isolines" )
  {
    return( PatchesMode::ISOLINES );
  }
  else if ( token == "Points" )
  {
    return( PatchesMode::POINTS );
  }
  else
  {
    onUndefinedToken( "PatchesMode", token );
    return( PatchesMode::TRIANGLES );
  }
}

PatchesOrdering DPAFLoader::readPatchesOrdering()
{
  string token = getNextToken();
  if ( token == "CCW" )
  {
    return( PatchesOrdering::CCW );
  }
  else if ( token == "CW" )
  {
    return( PatchesOrdering::CW );
  }
  else
  {
    onUndefinedToken( "PatchesOrdering", token );
    return( PatchesOrdering::CCW );
  }
}

PatchesSpacing DPAFLoader::readPatchesSpacing()
{
  string token = getNextToken();
  if ( token == "Equal" )
  {
    return( PatchesSpacing::EQUAL );
  }
  else if ( token == "Even" )
  {
    return( PatchesSpacing::FRACTIONAL_EVEN );
  }
  else if ( token == "Odd" )
  {
    return( PatchesSpacing::FRACTIONAL_ODD );
  }
  else
  {
    onUndefinedToken( "PatchesSpacing", token );
    return( PatchesSpacing::EQUAL );
  }
}

void DPAFLoader::readPrimitiveToken( PrimitiveData & pd, const string & token, const string & primitiveTypeName )
{
  if ( token == "elementCount" )
  {
    pd.elementCount = readScalar<unsigned int>( getNextToken() );
  }
  else if ( token == "elementOffset" )
  {
    pd.elementOffset = readScalar<unsigned int>( getNextToken() );
  }
  else if ( token == "indexSet" )
  {
    string name = readName( getNextToken() );
    std::map<std::string,IndexSetSharedPtr>::const_iterator it = m_indexSets.find( name );
    if ( it != m_indexSets.end() )
    {
      pd.indexSet = it->second;
    }
    else
    {
      onUndefinedToken( primitiveTypeName + ".indexSet", name );
    }
  }
  else if ( token == "instanceCount" )
  {
    pd.instanceCount = readScalar<unsigned int>( getNextToken() );
  }
  else if ( token == "patchesMode" )
  {
    pd.patchesMode = readPatchesMode();
  }
  else if ( token == "patchesOrdering" )
  {
    pd.patchesOrdering = readPatchesOrdering();
  }
  else if ( token == "patchesSpacing" )
  {
    pd.patchesSpacing = readPatchesSpacing();
  }
  else if ( token == "patchesType" )
  {
    pd.patchesType = patchesNameToType( getNextToken() );
  }
  else if ( token == "primitiveType" )
  {
    pd.primitiveType = primitiveNameToType( getNextToken() );
  }
  else if ( token == "vertexAttributeSet" )
  {
    string name = readName( getNextToken() );
    std::map<std::string,VertexAttributeSetSharedPtr>::const_iterator it = m_vertexAttributeSets.find( name );
    if ( it != m_vertexAttributeSets.end() )
    {
      pd.vertexAttributeSet = it->second;
    }
    else
    {
      onUndefinedToken( primitiveTypeName + ".vertexAttributeSet", name );
    }
  }
  else
  {
    onUnknownToken( primitiveTypeName, token );
  }
}

PrimitiveSharedPtr DPAFLoader::readPrimitive( const char *name )
{
  PrimitiveSharedPtr primitive;

  string token = getNextToken();

  if ( token == "{" )
  {
    PrimitiveData primitiveData = { PrimitiveType::UNINITIALIZED, PatchesType::NONE, PatchesMode::TRIANGLES, PatchesSpacing::EQUAL, PatchesOrdering::CCW, ~0u, ~0u, dp::sg::core::IndexSetSharedPtr(), ~0u, dp::sg::core::VertexAttributeSetSharedPtr() };

    token = getNextToken();
    while ( token != "}" )
    {
      readPrimitiveToken( primitiveData, token, "Primitive" );
      token = getNextToken();
    }
    DP_ASSERT( primitiveData.primitiveType != PrimitiveType::UNINITIALIZED );
    if ( primitiveData.primitiveType == PrimitiveType::PATCHES )
    {
      primitive = Primitive::create( primitiveData.patchesType, primitiveData.patchesMode );
    }
    else
    {
      primitive = Primitive::create( primitiveData.primitiveType );
    }
    setPrimitiveData( primitive, primitiveData, name );
  }
  else if ( m_primitives.find( token ) != m_primitives.end() )
  {
    primitive = std::static_pointer_cast<dp::sg::core::Primitive>(m_primitives[token]->clone());
    primitive->setName( name );
  }
  else
  {
    onUndefinedToken( "Primitive", token );
  }

  return( primitive );
}

Quatf DPAFLoader::readQuatf( const string& token )
{
  Quatf q;
  q[0] = readScalar<float>( token );
  q[1] = readScalar<float>( getNextToken() );
  q[2] = readScalar<float>( getNextToken() );
  q[3] = readScalar<float>( getNextToken() );
  q.normalize();
  return( q );
}

template<typename T>
T DPAFLoader::readScalar( const string & token )
{
  if ( std::numeric_limits<T>::is_integer )
  {
    char *endPtr;
    if ( std::numeric_limits<T>::is_signed )
    {
      return( dp::checked_cast<T>( strtol( token.c_str(), &endPtr, 0 ) ) );
    }
    else
    {
      return( dp::checked_cast<T>( strtoul( token.c_str(), &endPtr, 0 ) ) );
    }
  }
  else
  {
    return( static_cast<T>( _atof( token.c_str() ) ) );
  }
}

template<>
bool DPAFLoader::readScalar( const string & token )
{
  return( readBool( token ) );
}

template<typename T>
void DPAFLoader::readScalarArray( const string & t, vector<T> & values )
{
  onUnexpectedToken( "[", t );
  string token = getNextToken();
  while ( token != "]" )
  {
    values.push_back( readScalar<T>( token ) );
    token = getNextToken();
  }
}

SceneSharedPtr DPAFLoader::readScene( void )
{
  SceneSharedPtr scene = Scene::create();
  std::string token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "ambientColor" )
    {
      scene->setAmbientColor( readVector<3,float>( getNextToken() ) );
    }
    else if ( token == "backColor" )
    {
      scene->setBackColor( readVector<4,float>( getNextToken() ) );
    }
    else if ( token == "backImage" )
    {
      // backImage is either the name of a TextureHost, which should be in m_textureImages, or a file name, which should be found in m_searchPaths
      string fileName = readName( getNextToken() );
      std::map<std::string,dp::sg::core::TextureHostSharedPtr>::const_iterator it = m_textureImages.find( fileName );
      if ( it != m_textureImages.end() )
      {
        scene->setBackImage( it->second );
      }
      else
      {
        std::string foundFile = m_fileFinder.find( fileName );
        if ( !foundFile.empty() )
        {
          TextureHostSharedPtr texImg = dp::sg::io::loadTextureHost( foundFile, m_fileFinder );
          if ( texImg )
          {
            scene->setBackImage( texImg );
          }
        }
        else
        {
          onFileNotFound( fileName );
        }
      }
    }
    else if ( token == "cameras" )
    {
      token = getNextToken();
      onUnexpectedToken( "[", token );
      token = getNextToken();
      while ( token != "]" )
      {
        string name = readName( token );
        if ( m_matrixCameras.find( name ) != m_matrixCameras.end() )
        {
          scene->addCamera( m_matrixCameras[name] );
        }
        else if ( m_parallelCameras.find( name ) != m_parallelCameras.end() )
        {
          scene->addCamera( m_parallelCameras[name] );
        }
        else if ( m_perspectiveCameras.find( name ) != m_perspectiveCameras.end() )
        {
          scene->addCamera( m_perspectiveCameras[name] );
        }
        else
        {
          onUndefinedToken( "Scene.cameras", name );
        }
        token = getNextToken();
      }
    }
    else if ( token == "root" )
    {
      scene->setRootNode( readChild( getNextToken() ) );
    }
    else
    {
      onUndefinedToken( "Scene", token );
    }
    token = getNextToken();
  }
  return( scene );
}

SwitchSharedPtr DPAFLoader::readSwitch( const char *name, const std::string & extName )
{
  SwitchSharedPtr sw;

  string token = getNextToken();

  if ( token == "{" )
  {
    sw = Switch::create();
    token = getNextToken();
    vector<unsigned int> actives;
    while ( token != "}" )
    {
      if ( ! readGroupToken( sw, token, extName ) )
      {
        if ( token == "active" )
        {
          actives.clear();
          token = getNextToken();
          onUnexpectedToken( "[", token );
          token = getNextToken();
          while ( token != "]" )
          {
            actives.push_back( readScalar<unsigned int>( token ) );
            token = getNextToken();
          }
        }
        else
        {
          onUnknownToken( "Switch", token );
        }
      }
      token = getNextToken();
    }
    sw->setName( name );
    for ( size_t i=0; i<actives.size(); ++i )
    {
      sw->setActive( actives[i] );
    }
  }
  else if ( m_switches.find( token ) != m_switches.end() )
  {
    sw = std::static_pointer_cast<dp::sg::core::Switch>(m_switches[token]->clone());
    sw->setName( name );
  }
  else
  {
    onUndefinedToken( "Switch", token );
  }

  return( sw );
}

TextureTarget DPAFLoader::readTextureTarget()
{
  TextureTarget target = TextureTarget::UNSPECIFIED;
  std::string token = getNextToken();
  if ( token == "TEXTURE_1D" )
  {
    target = TextureTarget::TEXTURE_1D;
  }
  else if ( token == "TEXTURE_2D" )
  {
    target = TextureTarget::TEXTURE_2D;
  }
  else if ( token == "TEXTURE_3D" )
  {
    target = TextureTarget::TEXTURE_3D;
  }
  else if ( token == "TEXTURE_CUBE" )
  {
    target = TextureTarget::TEXTURE_CUBE;
  }
  else if ( token == "TEXTURE_1D_ARRAY" )
  {
    target = TextureTarget::TEXTURE_1D_ARRAY;
  }
  else if ( token == "TEXTURE_2D_ARRAY" )
  {
    target = TextureTarget::TEXTURE_2D_ARRAY;
  }
  else if ( token == "TEXTURE_RECTANGLE" )
  {
    target = TextureTarget::TEXTURE_RECTANGLE;
  }
  else if ( token == "TEXTURE_CUBE_ARRAY" )
  {
    target = TextureTarget::TEXTURE_CUBE_ARRAY;
  }
  else if ( token == "TEXTURE_BUFFER" )
  {
    target = TextureTarget::TEXTURE_BUFFER;
  }
  else if ( token == "TEXTURE_UNSPECIFIED" )
  {
    target = TextureTarget::UNSPECIFIED;
  }
  else
  {
    onUndefinedToken( "TextureTarget", token );
    DP_ASSERT( !"Unknown texture format" );
  }
  return target;
}

TextureHostSharedPtr DPAFLoader::readTextureHost( const char *name )
{
  TextureHostSharedPtr textureHost;

  string token = getNextToken();

  if ( token == "{" )
  {
    textureHost = TextureHost::create();
    TextureTarget textureTarget = TextureTarget::UNSPECIFIED;

    token = getNextToken();
    while ( token != "}" )
    {
      if ( token == "creationFlags" )
      {
        textureHost->setCreationFlags( readScalar<unsigned int>( getNextToken() ) );
      }
      else if ( token == "textureTarget" )
      {
        textureTarget = readTextureTarget(); // this needs to be delayed until after the images were read
      }
      else if ( token == "images" )
      {
        readImages( textureHost );
      }
      else
      {
        onUnknownToken( "TextureHost", token );
      }
      token = getNextToken();
    }

    textureHost->setTextureTarget( textureTarget );
  }
  else if ( m_textureImages.find( token ) != m_textureImages.end() )
  {
    textureHost = std::static_pointer_cast<dp::sg::core::TextureHost>(m_textureImages[token]->clone());
  }
  else
  {
    onUndefinedToken( "TextureHost", token );
  }

  return( textureHost );
}

TransformSharedPtr DPAFLoader::readTransform( const char *name, const std::string & extName )
{
  TransformSharedPtr transform;

  std::string token = getNextToken();

  if ( token == "{" )
  {
    transform = Transform::create();
    token = getNextToken();
    while ( token != "}" )
    {
      if ( ! readTransformToken( transform, token, extName ) )
      {
        onUnknownToken( "Transform", token );
      }
      token = getNextToken();
    };
    transform->setName( name );
  }
  else if ( m_transforms.find( token ) != m_transforms.end() )
  {
    transform = std::static_pointer_cast<dp::sg::core::Transform>(m_transforms[token]->clone());
    transform->setName( name );
  }
  else
  {
    onUndefinedToken( "Transform", token );
  }

  return( transform );
}

bool  DPAFLoader::readTransformToken( TransformSharedPtr const& transform, const string & token, const std::string & extName )
{
  bool b = true;
  Trafo trafo = transform->getTrafo();

  if ( token == "center" )
  {
    trafo.setCenter( readVector<3,float>( getNextToken() ) );
  }
  else if ( token == "orientation" )
  {
    trafo.setOrientation( readQuatf( getNextToken() ) );
  }
  else if ( token == "scaleOrientation" )
  {
    trafo.setScaleOrientation( readQuatf( getNextToken() ) );
  }
  else if ( token == "scaling" )
  {
    trafo.setScaling( readVector<3,float>( getNextToken() ) );
  }
  else if ( token == "translation" )
  {
    trafo.setTranslation( readVector<3,float>( getNextToken() ) );
  }
  else
  {
    b = readGroupToken( transform, token, extName );
  }
  transform->setTrafo( trafo );
  return( b );
}

template<unsigned int n, typename T>
Vecnt<n,T> DPAFLoader::readVector( const string & t )
{
  DP_ASSERT( t == "(" );
  Vecnt<n,T> v;
  unsigned int i=0;
  string token = getNextToken();
  while ( token != ")" )
  {
    DP_ASSERT( i < n );
    v[i++] = readScalar<T>( token );
    token = getNextToken();
  }
  DP_ASSERT( i == n );
  return( v );
}

void DPAFLoader::readVertexData( unsigned int type, unsigned char * vdata, string token )
{
  onUnexpectedToken( "[", token );
  token = getNextToken();
  while ( token != "]" )
  {
    switch ( dp::DataType(type) )
    {
      case dp::DataType::INT_8:
      case dp::DataType::UNSIGNED_INT_8:
        *vdata = (unsigned char)atoi(token.c_str());
        vdata++;
        break;
      case dp::DataType::INT_16:
      case dp::DataType::UNSIGNED_INT_16:
        *(unsigned short*)vdata = (unsigned short)atoi(token.c_str());
        vdata += sizeof(unsigned short);
        break;
      case dp::DataType::INT_32:
      case dp::DataType::UNSIGNED_INT_32:
        *(unsigned int*)vdata = static_cast<unsigned int>(atoi(token.c_str()));
        vdata += sizeof(unsigned int);
        break;
      case dp::DataType::FLOAT_32:
        *(float*)vdata = _atof(token.c_str());
        vdata += sizeof(float);
        break;
      case dp::DataType::FLOAT_64:
        *(double*)vdata = atof(token.c_str());
        vdata += sizeof(double);
        break;
      default:
        DP_ASSERT( !"unsupported datatype" );
    }
    token = getNextToken();
  }
}

VertexAttributeSetSharedPtr DPAFLoader::readVertexAttributeSet( const char *name )
{
  VertexAttributeSetSharedPtr vas;

  std::string token = getNextToken();

  if ( token == "{" )
  {
    vas = VertexAttributeSet::create();
    token = getNextToken();
    while ( token != "}" )
    {
      if ( ! readVertexAttributeSetToken( vas, token ) )
      {
        onUnknownToken( "VertexAttributeSet", token );
      }
      token = getNextToken();
    }
    vas->setName( name );
  }
  else if ( m_vertexAttributeSets.find( token ) != m_vertexAttributeSets.end() )
  {
    vas = std::static_pointer_cast<dp::sg::core::VertexAttributeSet>(m_vertexAttributeSets[token]->clone());
    vas->setName( name );
  }
  else
  {
    onUndefinedToken( "VertexAttributeSet", token );
  }

  return( vas );
}

bool DPAFLoader::readVertexAttributeSetToken( VertexAttributeSetSharedPtr const& vas, const string & token )
{
  bool b = true;
  string::size_type idx = token.find( "vattrib" );
  if ( idx == 0 )
  {
    VertexAttributeSet::AttributeID attrIndex = static_cast<VertexAttributeSet::AttributeID>(atoi( token.c_str() + strlen( "vattrib" ) ));
    DP_ASSERT(attrIndex < VertexAttributeSet::AttributeID::VERTEX_ATTRIB_COUNT);

    unsigned int size = readScalar<unsigned int>( getNextToken() );
    dp::DataType type = readType();
    BufferSharedPtr buffer = m_buffers[ getNextToken() ];
    unsigned int offset = readScalar<unsigned int>( getNextToken() );
    unsigned int stride = readScalar<unsigned int>( getNextToken() );
    unsigned int count = readScalar<unsigned int>( getNextToken() );

    VertexAttribute va;
    va.setData(size, type, buffer, offset, stride, count);
    vas->setVertexAttribute(attrIndex, va);
  }
  else if ( token == "enableFlags" )
  {
    unsigned int enableFlags = readScalar<unsigned int>( getNextToken() );
    for ( unsigned int i=0; enableFlags & ~((1<<i)-1); ++i )
    {
      if ( enableFlags & (1<<i) )
      {
        vas->setEnabled(static_cast<VertexAttributeSet::AttributeID>(i), true);
      }
    }
  }
  else if ( token == "normalizeFlags" )
  {
    unsigned int normalizeFlags = readScalar<unsigned int>( getNextToken() );
    // only considered for generic attributes!
    normalizeFlags >>= 16;
    for ( unsigned int i=0; normalizeFlags & ~((1<<i)-1); ++i )
    {
      if ( normalizeFlags & (1<<i) )
      {
        vas->setNormalizeEnabled(static_cast<VertexAttributeSet::AttributeID>(i+16), true);
      }
    }
  }
  else
  {
    b = readObjectToken( vas, token );
  }
  return( b );
}

dp::sg::ui::ViewStateSharedPtr DPAFLoader::readViewState( void )
{
  dp::sg::ui::ViewStateSharedPtr viewState = dp::sg::ui::ViewState::create();
  string token = getNextToken();
  while ( token != "}" )
  {
    if ( token == "autoClip" )
    {
      viewState->setAutoClipPlanes( readBool() );
    }
    else if ( token == "camera" )
    {
      string name = readName( getNextToken() );
      if ( m_matrixCameras.find( name ) != m_matrixCameras.end() )
      {
        viewState->setCamera( m_matrixCameras[name] );
      }
      else if ( m_parallelCameras.find( name ) != m_parallelCameras.end() )
      {
        viewState->setCamera( m_parallelCameras[name] );
      }
      else if ( m_perspectiveCameras.find( name ) != m_perspectiveCameras.end() )
      {
        viewState->setCamera( m_perspectiveCameras[name] );
      }
      else
      {
        onUndefinedToken( "ViewState", name );
      }
    }
    else if ( token == "stereoState" )
    {
      // ignore, no longer supported
      readBool();
    }
    else if ( token == "stereoAutomaticEyeDistanceAdjustment" )
    {
      viewState->setStereoAutomaticEyeDistanceAdjustment( readBool() );
    }
    else if ( token == "stereoAutomaticEyeDistanceFactor" )
    {
      viewState->setStereoAutomaticEyeDistanceFactor( readScalar<float>( getNextToken() ) );
    }
    else if ( token == "stereoEyeDistance" )
    {
      viewState->setStereoEyeDistance(readScalar<float>( getNextToken() ));
    }
    else if ( token == "targetDistance" )
    {
      viewState->setTargetDistance(readScalar<float>( getNextToken() ));
    }
    token = getNextToken();
  }
  return( viewState );
}

void DPAFLoader::setPrimitiveData( const PrimitiveSharedPtr & primitive, const PrimitiveData & data, const string & name )
{
  primitive->setIndexSet( data.indexSet );
  primitive->setInstanceCount( data.instanceCount );
  primitive->setVertexAttributeSet( data.vertexAttributeSet );
  primitive->setElementRange( data.elementOffset, data.elementCount );
  if ( primitive->getPrimitiveType() == PrimitiveType::PATCHES )
  {
    primitive->setPatchesOrdering( data.patchesOrdering );
    primitive->setPatchesSpacing( data.patchesSpacing );
  }
  primitive->setName( name );
}

void DPAFLoader::storeTextureHost( const std::string & name, const dp::sg::core::TextureHostSharedPtr & tih )
{
  DP_ASSERT( tih );
  m_textureImages[name] = tih;
}

bool  DPAFLoader::testDPVersion( void )
{
  m_line = 0;
  getNextLine();
  return( m_currentLine == "#DPAF V1.0");
}
