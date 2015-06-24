// Copyright NVIDIA Corporation 2002-2015
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
// XMLLoader.cpp
//

#include <dp/sg/io/PlugInterface.h> // definition of UPITID_VERSION,
#include <dp/sg/io/PlugInterfaceID.h> // definition of UPITID_VERSION, UPITID_SCENE_LOADER, and UPITID_SCENE_SAVER

#include <dp/Exception.h>
#include <dp/sg/core/Config.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/io/IO.h>
#include <dp/sg/io/PlugInterface.h>
#include <dp/sg/io/PlugInterfaceID.h>

#include "XMLLoader.h"

#include <dp/math/Vecnt.h>
#include <dp/math/Quatt.h>

#include <dp/sg/core/Group.h>
#include <dp/sg/core/Transform.h>

#include <dp/util/PlugIn.h>
#include <dp/util/File.h>
#include <dp/util/Tools.h>

// optimizers
#include <dp/sg/algorithm/CombineTraverser.h>
#include <dp/sg/algorithm/UnifyTraverser.h>
#include <dp/sg/algorithm/StrippingTraverser.h>
#include <dp/sg/algorithm/NormalizeTraverser.h>

// stl headers
#include <algorithm>

using namespace dp::sg::core;
using namespace dp::math;

using namespace std;

// define a unique plug-interface ID for SceneLoader
const dp::util::UPITID PITID_SCENE_LOADER(UPITID_SCENE_LOADER, UPITID_VERSION);

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();

  piids.push_back(dp::util::UPIID(".XML", PITID_SCENE_LOADER));
}

bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugInSharedPtr & pi)
{
  const dp::util::UPIID PIID_XML_SCENE_LOADER = dp::util::UPIID(".XML", PITID_SCENE_LOADER);

  if ( piid == PIID_XML_SCENE_LOADER )
  {
    pi = XMLLoader::create();
    return( !!pi );
  }

  return false;
}

XMLLoaderSharedPtr XMLLoader::create()
{
  return( std::shared_ptr<XMLLoader>( new XMLLoader() ) );
}

XMLLoader::XMLLoader()
  : m_viewState(NULL)
{
}

XMLLoader::~XMLLoader()
{
}

#if 0
void
XMLLoader::optimizeGeometry( GeoNodeWeakPtr geode )
{
  UnifyTraverser ut;
  ut.setIgnoreNames( true );
  ut.setUnifyTargets(  UnifyTraverser::UT_STATE_SET | UnifyTraverser::UT_STATE_PASS | UnifyTraverser::UT_STATE_ATTRIBUTE );
  ut.apply( geode );

//  NormalizeTraverser nt;
//  nt.apply( geode );

  CombineTraverser ct;
  ct.setClassesToCombine( CombineTraverser::CC_PRIMITIVE_SET );
  ct.apply( geode );

  ut.setUnifyTargets(  UnifyTraverser::UT_PRIMITIVE_SET | UnifyTraverser::UT_VERTEX_ATTRIBUTE_SET
                     | UnifyTraverser::UT_VERTICES );
  ut.setEpsilon( 0.00001f );
  ut.apply( geode );

//  StrippingTraverser st;
//  st.apply( geode );
}
#endif

SceneSharedPtr
XMLLoader::lookupFile( const string & file )
{
  SceneSharedPtr scene;

  map< string, SceneSharedPtr >::iterator iter = m_fileCache.find( file ); 

  if( iter == m_fileCache.end() )
  {
    // find file
    std::string foundFile = m_fileFinder.find( file );
    if ( !foundFile.empty() )
    {
      dp::sg::ui::ViewStateSharedPtr viewState = dp::sg::io::loadScene( foundFile, m_fileFinder.getSearchPaths() );
      if ( viewState )
      {
        scene = viewState->getScene();
      }

      if ( scene )
      {
        // add it
        m_fileCache[ file ] = scene;
      }
    }
  }
  else
  {
    scene = (*iter).second;
  }

  return scene;
}

void
XMLLoader::buildScene( GroupSharedPtr const& parent, TiXmlDocument & doc, TiXmlNode * node )
{
  if( !node )
  {
    return;
  }

  TiXmlElement * element = node->ToElement();
  const string value = node->Value();

  // skip everything that is not an element
  if( element )
  {
    if( value == "file" )
    {
      //cout << "XMLLoader: Adding File: " << element->GetText() << endl;
      SceneSharedPtr scene = lookupFile( element->GetText() );

      if( scene )
      {
        // see if they positioned it
        const char * pos = element->Attribute( "position" );
        const char * ori = element->Attribute( "orientation" );
        const char * note = element->Attribute( "annotation" );

        NodeSharedPtr root = scene->getRootNode();
        NodeSharedPtr theNode;

        if( pos || ori )
        {
          TransformSharedPtr transH( Transform::create() );
          transH->addChild( root.clone() );
          Trafo trafo;

          if( pos )
          {
            stringstream ss( pos );
            float x, y, z;

            ss >> x;
            ss >> y;
            ss >> z;

            trafo.setTranslation( Vec3f( x, y, z ) );
          }

          if( ori )
          {
            stringstream ss( ori );
            float x, y, z, w;
            int args = 0;

            // read first value
            ss >> x;
            args++;

            if( ss.good() )
            {
              args+=2;
              ss >> y;
              ss >> z;

              if( ss.good() )
              {
                args++;
                ss >> w;
              }
            }

            if( args == 4 )
            {
              trafo.setOrientation( Quatf( x, y, z, w ) ); 
            }
          }

          transH->setTrafo( trafo );

          // reset 'theNode' here
          theNode = transH;
        }
        else
        {
          theNode = root.clone();
        }

        if( note )
        {
          theNode->setAnnotation( note );
        }

        parent->addChild( theNode );
      }
    }
    else if( value == "searchpath" )
    {
      //cout << "XMLLoader: Adding SearchPath: " << element->GetText() << endl;
      m_fileFinder.addSearchPath( element->GetText() );
    }
    else if( value == "env" )
    {
      //cout << "XMLLoader: Adding Env: " << element->GetText() << endl;
#ifdef _WIN32
      putenv( element->GetText() );
#else
      // putenv is no good under unix because the string storage has to
      // remain valid.  we need to use setenv() instead.

      std::stringstream ss( element->GetText() );
      char key[256] = {0}, value[4096] = {0};

      ss.getline( key, 256, '=' );

      if( ss.good() )
      {
        ss.getline( value, 4096 );
      }

      setenv( key, value, 1 );
#endif
    }
  }
}

SceneSharedPtr
XMLLoader::load( std::string const& filename
               , std::vector<std::string> const& searchPaths
               , dp::sg::ui::ViewStateSharedPtr & viewState )
{
  if ( !dp::util::fileExists(filename) )
  {
    throw dp::FileNotFoundException( filename );
  }

  // set locale temporarily to standard "C" locale
  dp::util::TempLocale tl("C");

  TiXmlDocument doc( filename.c_str() );

  m_viewState = viewState.getWeakPtr();

  m_fileFinder.addSearchPaths( searchPaths );

#if 0
  for(unsigned int i=0;i<m_searchPath.size();i++)
  {
    cout << "Incoming Path: " << m_searchPath[i] << endl;
  }
#endif

  SceneSharedPtr hScene;
  if ( !(doc.LoadFile() && doc.FirstChild() ) )
  {
    throw std::runtime_error( std::string( "Failed to load file " + filename ) );
  }

  // create toplevel group
  GroupSharedPtr hGroup(Group::create());
  hGroup->setName( filename );

  // build scene
  TiXmlNode * child = doc.FirstChild();

  do
  {
    buildScene( hGroup, doc, child );
  } while( child = child->NextSibling() ); 

  // create toplevel scene
  hScene = Scene::create();

  // add group as scene's toplevel
  hScene->setRootNode( hGroup );

  m_fileFinder.clear();

  return hScene;
}
