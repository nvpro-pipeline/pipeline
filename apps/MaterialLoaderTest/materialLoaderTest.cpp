// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <iostream>
#include <vector>
#include <boost/program_options.hpp>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <dp/DP.h>
#include <dp/fx/EffectLibrary.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/TextureFile.h>
#include <dp/sg/generator/MeshGenerator.h>
#include <dp/sg/renderer/rix/gl/SceneRenderer.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/sg/ui/glut/SceneRendererWidget.h>
#include <dp/util/File.h>

dp::fx::Manager getShaderManager( std::string const& name )
{
  static const std::map<std::string, dp::fx::Manager> shaderManager =
  {
    { "rix:ubo140",             dp::fx::Manager::UNIFORM_BUFFER_OBJECT_RIX },
    { "rix:ssbo140",            dp::fx::Manager::SHADER_STORAGE_BUFFER_OBJECT_RIX },
    { "rixfx:uniform",          dp::fx::Manager::UNIFORM },
    { "rixfx:shaderbufferload", dp::fx::Manager::SHADERBUFFER },
    { "rixfx:ubo140",           dp::fx::Manager::UNIFORM_BUFFER_OBJECT_RIX_FX },
    { "rixfx:ssbo140",          dp::fx::Manager::SHADER_STORAGE_BUFFER_OBJECT }
  };
  auto const& smit = shaderManager.find( name );
  return( ( smit == shaderManager.end() ) ? dp::fx::Manager::UNIFORM_BUFFER_OBJECT_RIX : smit->second );
}


class MaterialLoaderTestWidget : public dp::sg::ui::glut::SceneRendererWidget
{
  public:
    MaterialLoaderTestWidget( std::string const& renderEngine, dp::fx::Manager shaderManager, bool saveShaders, std::vector<std::string> const& effectNames, size_t bench );
    virtual ~MaterialLoaderTestWidget();

  protected:
    void virtual paint();

  private:
    size_t                          m_bench;
    std::vector<std::string>        m_effectNames;
    dp::sg::core::GeoNodeSharedPtr  m_geoNode;
    bool                            m_saveShaders;
};

MaterialLoaderTestWidget::MaterialLoaderTestWidget( std::string const& renderEngine, dp::fx::Manager shaderManager, bool saveShaders, std::vector<std::string> const& effectNames, size_t bench )
  : m_bench( bench )
  , m_effectNames( effectNames )
  , m_saveShaders( saveShaders )
{
  dp::sg::core::PrimitiveSharedPtr primitive = dp::sg::generator::createCube();
  primitive->generateTangentSpace();

  m_geoNode = dp::sg::core::GeoNode::create();
  m_geoNode->setPrimitive( primitive );

  dp::sg::core::SceneSharedPtr scene = dp::sg::core::Scene::create();
  scene->setRootNode( m_geoNode );

  dp::sg::ui::ViewStateSharedPtr viewState = dp::sg::ui::ViewState::create();
  viewState->setSceneTree( dp::sg::xbar::SceneTree::create( scene ) );
  dp::sg::ui::setupDefaultViewState( viewState );
  viewState->getCamera()->addHeadLight( dp::sg::core::createStandardPointLight() );

  // use an environment sampler
  DP_VERIFY( dp::fx::EffectLibrary::instance()->loadEffects( dp::home() + "/apps/Viewer/res/viewerEffects.xml" ) );

  dp::sg::core::TextureFileSharedPtr environmentTexture = dp::sg::core::TextureFile::create( dp::home() + "/media/textures/spheremaps/spherical_checker.png", dp::sg::core::TextureTarget::TEXTURE_2D );
  environmentTexture->incrementMipmapUseCount();

  dp::sg::core::SamplerSharedPtr environmentSampler = dp::sg::core::Sampler::create( environmentTexture );
  environmentSampler->setMagFilterMode( dp::sg::core::TextureMagFilterMode::LINEAR );
  environmentSampler->setMinFilterMode( dp::sg::core::TextureMinFilterMode::LINEAR_MIPMAP_LINEAR );
  environmentSampler->setWrapModes( dp::sg::core::TextureWrapMode::REPEAT, dp::sg::core::TextureWrapMode::CLAMP_TO_EDGE, dp::sg::core::TextureWrapMode::CLAMP_TO_EDGE );

  dp::sg::renderer::rix::gl::SceneRendererSharedPtr sceneRenderer = dp::sg::renderer::rix::gl::SceneRenderer::create( renderEngine.c_str(), shaderManager, dp::culling::Mode::AUTO, dp::sg::renderer::rix::gl::TransparencyMode::NONE );
  sceneRenderer->setEnvironmentSampler( environmentSampler );
  sceneRenderer->setRenderTarget( getRenderTarget() );
  sceneRenderer->setEnvironmentRenderingEnabled( true );

  setContinuousUpdate( true );
  setSceneRenderer( sceneRenderer );
  setViewState( viewState );
}

MaterialLoaderTestWidget::~MaterialLoaderTestWidget()
{
}

static std::string domainString( dp::fx::Domain domain )
{
  switch( domain )
  {
    case dp::fx::Domain::VERTEX                  : return( "vert" );
    case dp::fx::Domain::TESSELLATION_CONTROL    : return( "teco" );
    case dp::fx::Domain::TESSELLATION_EVALUATION : return( "teev" );
    case dp::fx::Domain::GEOMETRY                : return( "geom" );
    case dp::fx::Domain::FRAGMENT                : return( "frag" );
    default                                     : return( "xxxx" );
  }
}

void MaterialLoaderTestWidget::paint()
{
  static size_t idx = 0;
  if (idx == m_effectNames.size())
  {
    glutLeaveMainLoop();
  }
  else
  {
    std::cout << "MaterialLoaderTest: using effect <" << m_effectNames[idx] << ">";
    m_geoNode->setMaterialPipeline(dp::sg::core::PipelineData::create(dp::fx::EffectLibrary::instance()->getEffectData(m_effectNames[idx])));

    dp::sg::ui::glut::SceneRendererWidget::paint();

    if (m_saveShaders)
    {
      std::map<dp::fx::Domain, std::string> sources = getSceneRenderer()->getShaderSources(m_geoNode, false);
      for (auto const& it : sources)
      {
        std::string effectFile = dp::fx::EffectLibrary::instance()->getEffectFile(m_effectNames[idx]);
        dp::util::saveStringToFile(dp::util::getFilePath(effectFile) + "\\" + dp::util::getFileStem(effectFile) + "." + domainString(it.first) + ".glsl", it.second);
      }
    }

    if (0 < m_bench)
    {
      dp::util::Timer timer;
      timer.start();
      for (size_t i = 0; i < m_bench; i++)
      {
        dp::sg::ui::glut::SceneRendererWidget::paint();
      }
      timer.stop();
      std::cout << " -> measured time per frame: " << timer.getTime() * 1000 / m_bench << " ms";
    }
    std::cout << std::endl;

    ++idx;
  }
}

int main( int argc, char *argv[] )
{
  boost::program_options::options_description od("Usage: MaterialLoaderTest");
  od.add_options()
    ( "bench",          boost::program_options::value<size_t>()->default_value( 0 ),                  "number of frames to bench per material" )
    ( "ext",            boost::program_options::value<std::string>(),                                 "extension of multiple files to handle" )
    ( "file",           boost::program_options::value<std::string>(),                                 "single file to handle" )
    ( "help",                                                                                         "show help")
    ( "path",           boost::program_options::value<std::string>(),                                 "path to multiple files to handle" )
    ( "renderEngine",   boost::program_options::value<std::string>()->default_value( "Bindless" ),    "choose a render engine from this list: VBO|VAB|BVAB|VBOVAO|Bindless|BindlessVAO|DisplayList" )
    ( "root",           boost::program_options::value<std::string>(),                                 "root path of the material package" )
    ( "saveShader",                                                                                   "save shader sources to files" )
    ( "shaderManager",  boost::program_options::value<std::string>()->default_value( "rix:ubo140" ),  "choose a shader manager from this list: rixfx:uniform|rixfx:ubo140|rixfx:ssbo140|rixfx:shaderbufferload|rix:ubo140|rix:ssbo140" )
    ;

  boost::program_options::variables_map opts;
  boost::program_options::store( boost::program_options::parse_command_line( argc, argv, od ), opts );

  if ( !opts["help"].empty() )
  {
    std::cout << od << std::endl;
    return( 0 );
  }
  if ( opts["ext"].empty() && opts["file"].empty() && opts["path"].empty() )
  {
    std::cout << argv[0] << " : at least argument --file or arguments --path and --ext are needed!" << std::endl;
    return( 0 );
  }

  std::string ext, file, root, path;
  if ( !opts["ext"].empty() )
  {
    if ( !opts["file"].empty() )
    {
      std::cout << argv[0] << " : argument --ext and argument --file exclude each other!" << std::endl;
      return( 0 );
    }
    if ( opts["path"].empty() )
    {
      std::cout << argv[0] << " : argument --ext needs argument --path as well!" << std::endl;
      return( 0 );
    }
    ext = opts["ext"].as<std::string>();
    if ( ext.front() != '.' )
    {
      ext = "." + ext;
    }
  }
  if ( !opts["file"].empty() )
  {
    if ( !opts["path"].empty() )
    {
      std::cout << argv[0] << " : argument --file and argument --path exclude each other!" << std::endl;
      return( 0 );
    }
    file = opts["file"].as<std::string>();
  }
  if ( !opts["root"].empty() )
  {
    root = opts["root"].as<std::string>();
    if ( ! dp::util::directoryExists( root ) )
    {
      std::cout << argv[0] << " : root <" << root << "> not found!" << std::endl;
      return( 0 );
    }
    if ( ( root.back() != '\\' ) && ( root.back() != '/' ) )
    {
      root.push_back( '\\' );
    }
  }
  if ( !opts["path"].empty() )
  {
    path = opts["path"].as<std::string>();
  }

  std::vector<std::string> files;
  if ( !file.empty() )
  {
    files.push_back( root + file );
    if ( ! dp::util::fileExists( files.back() ) )
    {
      std::cout << argv[0] << " : file <" << files.back() << "> not found!" << std::endl;
      return( 0 );
    }
  }
  else
  {
    path = root + path;
    if ( ! dp::util::directoryExists( path ) )
    {
      std::cout << argv[0] << " : path <" << path << "> not found!" << std::endl;
      return( 0 );
    }
    dp::util::findFilesRecursive( ext, path, files );
  }

  if ( files.empty() )
  {
    std::cerr << argv[0] << " : No files found!";
    return( -1 );
  }

  dp::util::FileFinder fileFinder;
  if ( !root.empty() )
  {
    DP_ASSERT( ( root.back() == '\\' ) || ( root.back() == '/' ) );
    root.pop_back();
    fileFinder.addSearchPath( root );
  }
  fileFinder.addSearchPath( dp::home() + "/media/effects/mdl" );
  fileFinder.addSearchPath( dp::home() + "/media/textures/mdl" );

  std::vector<std::string> effectNames;
  for ( size_t i=0 ; i<files.size() ; i++ )
  {
    std::cout << "MaterialLoaderTest: loading effect from " << files[i] << std::endl;
    DP_VERIFY( dp::fx::EffectLibrary::instance()->loadEffects( files[i], fileFinder ) );
    dp::fx::EffectLibrary::instance()->getEffectNames( files[i], dp::fx::EffectSpec::Type::PIPELINE, effectNames );
  }
  std::cout << "MaterialLoaderTest: finished loading " << effectNames.size() <<" effects from " << files.size() << " files." << std::endl;

  glutInit( &argc, argv );
  glutSetOption( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION );

  MaterialLoaderTestWidget w( opts["renderEngine"].as<std::string>(), getShaderManager( opts["shaderManager"].as<std::string>() ), !opts["saveShader"].empty(), effectNames, opts["bench"].as<size_t>() );

  glutMainLoop();

  return( 0 );
}

