// Copyright NVIDIA Corporation 2013
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


#include <iostream>
#include <tinyxml.h>
#include <dp/fx/EffectLibrary.h>
#include <dp/sg/algorithm/Replace.h>
#include <dp/sg/algorithm/Search.h>
#include <dp/sg/core/nvsg.h>
#include <dp/sg/io/IO.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/util/File.h>

// exchanger sceneFile mappingsFile materialsFile
// e.g.: MaterialExchanger sample.nbf sample_mappings.xml sample_materials.xml
int main( int argc, char *argv[] )
{
  if ( argc != 3 )
  {
    std::cerr << "exchanger usage: exchanger sceneFile mappingsFile materialsFile\n"
              << "  where sceneFile is a DP/SG-loadable scene file (*.nbf, *.dae, ...)\n"
              << "  and mappingsFile is an XML file that gives the mapping of EffectData\n"
              << "  names in the scene to EffectData names in the materialsFile,\n"
              << "  and materialsFile is an XML file with the EffectData to use.\n"
              << "  Replaces all EffectData in the scene with a name like \"*fx\"\n"
              << "  with a material from the materialsFile with the same name without \"fx\"\n";
  }

  std::string sceneFile = argv[1];
  std::string mappingsFile = argv[2];
  std::string materialsFile = argv[3];

  if ( !dp::util::fileExists( sceneFile ) )
  {
    std::cerr << "Could not find sceneFile <" << sceneFile << ">.\n";
    return( -1 );
  }
  if ( !dp::util::fileExists( mappingsFile ) )
  {
    std::cerr << "Could not find mappingsFile <" << mappingsFile << ">.\n";
    return( -1 );
  }
  if ( !dp::util::fileExists( materialsFile ) )
  {
    std::cerr << "Could not find materialsFile <" << materialsFile << ">.\n";
    return( -1 );
  }

  dp::sg::core::nvsgInitialize();
  dp::sg::ui::ViewStateSharedPtr viewState = dp::sg::io::loadScene( sceneFile );

  if ( ! viewState )
  {
    std::cerr << "Could not load sceneFile <" << sceneFile << ">.\n";
    return( -1 );
  }

  dp::sg::core::SceneSharedPtr scene = viewState->getScene();

  if ( ! scene )
  {
    std::cerr << "The sceneFile <" << sceneFile << "> does not contain a Scene.\n";
    return( -1 );
  }

  dp::sg::core::NodeSharedPtr rootNode = scene->getRootNode();

  if ( ! rootNode )
  {
    std::cerr << "The scene in <" << sceneFile << "> does not contain a root node.\n";
    return( -1 );
  }

  if ( ! dp::fx::EffectLibrary::instance()->loadEffects( materialsFile ) )
  {
    std::cerr << "Could not load materialsFile <" << materialsFile << ">.\n";
    return( -1 );
  }

  std::unique_ptr<TiXmlDocument> doc( new TiXmlDocument( mappingsFile.c_str() ) );
  if ( ! doc )
  {
    std::cerr << "Could not open mappingsFile <" << mappingsFile << ">.\n";
    return( -1 );
  }

  if ( ! doc->LoadFile() )
  {
    std::cerr << "Could not load mappingsFile <" << mappingsFile << ">.\n";
    return( -1 );
  }

  TiXmlHandle libraryHandle = doc->FirstChildElement( "library" );   // The required XML root node.
  TiXmlElement * rootElement = libraryHandle.Element();

  if ( ! rootElement )
  {
    std::cerr << "mappingsFile <" << mappingsFile << "> does not contain any mapping.\n";
    return( -1 );
  }

  dp::sg::algorithm::ReplacementMapEffectData replacementMap;
  for ( TiXmlElement * element = rootElement->FirstChildElement() ; element ; element = element->NextSiblingElement() )
  {
    DP_ASSERT( element->Attribute( "from" ) && element->Attribute( "to" ) );
    std::string from( element->Attribute( "from" ) );
    std::string to( element->Attribute( "to" ) );

    dp::fx::SmartEffectData effectData = dp::fx::EffectLibrary::instance()->getEffectData( to );
    if ( ! effectData )
    {
      std::cerr << "Could no get EffectData <" << to << ".\n";
      return( -1 );
    }
    DP_ASSERT( replacementMap.find( from ) == replacementMap.end() );
    replacementMap[from] = dp::sg::core::EffectData::create( effectData );
  }

  dp::sg::algorithm::replaceEffectDatas( scene, replacementMap );

  std::string saveName = dp::util::getFilePath( sceneFile ) + "/" + dp::util::getFileStem( sceneFile ) + "X.nbf";

  if ( ! dp::sg::io::saveScene( saveName, viewState ) )
  {
    std::cerr << "Could not save the converted scene to <" << saveName << ">\n";
    return( -1 );
  }

  dp::sg::core::nvsgTerminate();

  return( 0 );
}

