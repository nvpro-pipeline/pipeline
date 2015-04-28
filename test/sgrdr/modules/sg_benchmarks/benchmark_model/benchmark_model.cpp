// Copyright NVIDIA Corporation 2011
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


#include <test/testfw/manager/Manager.h>
#include "benchmark_model.h"

#include <dp/sg/io/IO.h>

#include <dp/sg/generator/GeoSphereScene.h>
#include <dp/sg/generator/PreviewScene.h>
#include <dp/sg/generator/SimpleScene.h>

#include <boost/program_options.hpp>

using namespace dp;
using namespace sgrdr;

namespace options = boost::program_options;

//Automatically add the test to the module's global test list
REGISTER_TEST("benchmark_model", "tests performance of SceniX models", create_benchmark_model);


Benchmark_model::Benchmark_model()
  : m_renderData(nullptr)
  , m_sgrdr(nullptr)
{
}

Benchmark_model::~Benchmark_model()
{
}

bool Benchmark_model::onInit()
{
  m_renderData = new test::framework::RenderDataSgRdr;
  DP_ASSERT( dynamic_cast<test::framework::SgRdrBackend*>(&(*m_backend)) );
  m_sgrdr = dynamic_cast<test::framework::SgRdrBackend*>(&(*m_backend))->getRenderer();

  dp::sg::ui::ViewStateSharedPtr viewStateHandle = createScene();
  m_renderData->setViewState( viewStateHandle );
  dp::sg::ui::setupDefaultViewState( viewStateHandle );


  return true;  
}

bool Benchmark_model::onRun(unsigned int i)
{
  render(m_renderData, m_displayTarget);

  return true;  
}

bool Benchmark_model::onRunCheck( unsigned int i )
{
  return i < m_repetitions;
}

bool Benchmark_model::onClear()
{
  delete m_renderData;

  return true;
}

dp::sg::ui::ViewStateSharedPtr Benchmark_model::createScene()
{
  dp::sg::core::SceneSharedPtr scene;

  if( m_sceneFileName == "cubes" )
  {
    dp::sg::generator::SimpleScene simpleScene;
    scene = simpleScene.m_sceneHandle;
  }
  else if ( m_sceneFileName == "preview" )
  {
    PreviewScene previewScene;  // Vertex_3f, TexCoord0_3f, Tangent_3f, Binormal_3f, Normal_3f on all five objects. Only front faces visible, no thin geometry.
    scene = previewScene.m_sceneHandle;
  }
  else if ( m_sceneFileName == "geosphere" )
  {
    GeoSphereScene geoSphereScene; // Vertex_3f, Normal_3f. Only front faces visible, no thin geometry.
    scene = geoSphereScene.m_sceneHandle;
  }
  else
  {
    return dp::sg::io::loadScene( m_sceneFileName );
  }

  dp::sg::ui::ViewStateSharedPtr viewStateHandle = dp::sg::ui::ViewState::create();
  viewStateHandle->setSceneTree( dp::sg::xbar::SceneTree::create( scene ) );

  return viewStateHandle;
}

bool Benchmark_model::option( const std::vector<std::string>& optionString )
{
  TestRender::option(optionString);


  options::options_description od("Usage: benchmark_model");
  od.add_options() ( "filename", options::value<std::string>(), "Filename of the model" )
                   ( "repetitions", options::value<unsigned int>()->default_value(32), "How many times the scene should be rendered" )
    ;

  options::basic_parsed_options<char> parsedOpts = options::basic_command_line_parser<char>(optionString).options( od ).allow_unregistered().run();

  options::variables_map optsMap;

  try
  {
    options::store( parsedOpts, optsMap );
  }
  catch( options::invalid_option_value e )
  {
    std::cerr << "Error: Invalid values specified. Exiting program.\n";
    return false;
  }


  if( !optsMap["filename"].empty() )
  {
    m_sceneFileName = optsMap["filename"].as<std::string>();
  }
  else
  {
    std::cerr << "Error: A model file needs to be specified for benchmark_model\n";
    return false;
  }
  
  m_repetitions = optsMap["repetitions"].as<unsigned int>();

  return true;
}

