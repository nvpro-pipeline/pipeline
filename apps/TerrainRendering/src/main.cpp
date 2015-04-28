// Copyright NVIDIA Corporation 2012-2015
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


#include <GL/glew.h>
#include <GL/freeglut.h>

#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/core/EffectData.h>

#include <dp/sg/io/IO.h>

#include <dp/sg/renderer/rix/gl/SceneRenderer.h>

#include <dp/sg/ui/SceniXWidget.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/sg/ui/glut/SceneRendererWidget.h>
#include <dp/sg/ui/manipulator/TrackballCameraManipulatorHIDSync.h>

#include <dp/sg/generator/Terrain.h>

#include <dp/util/File.h>
#include <dp/util/FrameProfiler.h>

#include <boost/program_options.hpp>

#include <fstream>
#include <iomanip>
#include <dp/sg/algorithm/Search.h>

namespace options = boost::program_options;

/************************************************************************/
/* TerrainRendering                                                          */
/************************************************************************/

class TerrainRenderer : public dp::sg::ui::glut::SceneRendererWidget
{
public:
  TerrainRenderer();
  ~TerrainRenderer();

  /** \brief Exit after the given number of frames + 1. Use getExitCode() to retrieve the framerate.
  **/
  void setNumberOfFrames( dp::Uint32 numberOfFrames );

  /** \brief Exit after the given duration. Use getExitCode() to retrieve the framerate.
  **/
  void setDuration( double duration );

  virtual void onHIDEvent( dp::util::PropertyId propertyId );

  int getExitCode() const;

  virtual void paint();
 
protected:
    virtual void onSceneRendererChanged( const dp::sg::ui::SceneRendererSharedPtr &sceneRenderer );

private:
  std::shared_ptr<dp::sg::ui::manipulator::TrackballCameraManipulatorHIDSync> m_trackballHIDSync;

  void updateSceneRendererEngine();

  enum AttributeType
  {
      ATTRIBUTE_GENERIC // GL 2.x
    , ATTRIBUTE_VAO // GL 3.x
    , ATTRIBUTE_VAB // GL 4.3
  };

  // benchmark
  dp::Uint32      m_renderedFrames;
  dp::Uint32      m_benchmarkFrames;
  dp::util::Timer m_benchmarkTimer;
  dp::util::Timer m_benchmarkProgressTimer;
  int             m_exitCode;
  double          m_duration;

  bool            m_engineBindless;
  AttributeType   m_attributeType;

  std::string     m_renderEngine;
  dp::fx::Manager m_shaderManager;
};

TerrainRenderer::TerrainRenderer()
  : m_trackballHIDSync(new dp::sg::ui::manipulator::TrackballCameraManipulatorHIDSync( ) )
  , m_benchmarkFrames( ~0 )
  , m_renderedFrames( 0 )
  , m_exitCode( 0 )
  , m_duration( 0.0 )
  , m_engineBindless( true )
  , m_attributeType( ATTRIBUTE_GENERIC )
  , m_shaderManager( dp::fx::MANAGER_SHADERBUFFER )
{
  m_trackballHIDSync->setHID( this );
  m_trackballHIDSync->setRenderTarget( getRenderTarget() );
  setManipulator( m_trackballHIDSync.get() );
}

TerrainRenderer::~TerrainRenderer()
{
  setManipulator( 0 );
}

void TerrainRenderer::setNumberOfFrames( dp::Uint32 numberOfFrames )
{
  m_benchmarkFrames = numberOfFrames;
  if( numberOfFrames != ~0 )
  {
    setContinuousUpdate( true );
  }
}

void TerrainRenderer::setDuration( double duration )
{
  m_duration = duration;
  if( duration > 0.0 )
  {
    setContinuousUpdate( true );
  }
}

void TerrainRenderer::paint()
{
  dp::sg::renderer::rix::gl::SceneRendererSharedPtr renderer = getSceneRenderer().staticCast<dp::sg::renderer::rix::gl::SceneRenderer>();
  if ( !m_renderEngine.empty() && renderer->getRenderEngine() != m_renderEngine )
  {
    std::cout << "Setting renderengine: " << m_renderEngine << std::endl;
    renderer->setRenderEngine( m_renderEngine );
  }

  renderer->setShaderManager( m_shaderManager);

  glPatchParameteri( GL_PATCH_VERTICES, 1 ); // TODO temporary, terrain patch has only 1 'virtual' vertex per patch
  //glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

  dp::util::FrameProfiler::instance().beginFrame();
  if ( m_benchmarkFrames != ~0 || m_duration != 0.0 )
  {
    if ( m_renderedFrames == 1 )
    {
      m_benchmarkTimer.start();
      m_benchmarkProgressTimer.start();
    }

    SceneRendererWidget::paint();

    if ( m_benchmarkProgressTimer.getTime() > 1.0 )
    {
      m_benchmarkProgressTimer.restart();
      std::ostringstream os;
      os << "Benchmark Progress: ";
      if ( m_benchmarkFrames != ~0 )
      {
        os << m_renderedFrames << "/" << m_benchmarkFrames;
      }
      else
      {
        os << std::setprecision(2) << m_benchmarkTimer.getTime() << "/" << m_duration;
      }
      setWindowTitle( os.str() );
    }
    if ( (m_benchmarkFrames != ~0 && m_renderedFrames == m_benchmarkFrames) || (m_duration > 0.0 && m_benchmarkTimer.getTime() > m_duration) )
    {
      m_benchmarkTimer.stop();
      m_exitCode = int(double(m_renderedFrames) / m_benchmarkTimer.getTime());
      glutLeaveMainLoop();
    }

    // at the end since the first frame does not count
    ++m_renderedFrames;
  }
  else
  {
    SceneRendererWidget::paint();
  }
  dp::util::FrameProfiler::instance().endFrame();
}

int TerrainRenderer::getExitCode() const
{
  return m_exitCode;
}

void TerrainRenderer::onHIDEvent( dp::util::PropertyId propertyId )
{
  if ( propertyId == PID_Key_P )
  {
    if ( getValue<bool>( propertyId ) )
    {
      dp::util::FrameProfiler::instance().setEnabled( !dp::util::FrameProfiler::instance().isEnabled() );
    }
  }
  if ( propertyId == PID_Mouse_Right && getValue<bool>( propertyId ) )
  {
    dp::math::Vec2i position = getValue<dp::math::Vec2i>( PID_Mouse_Position );
    
    float depth;
    glReadPixels( position[0], glutGet( GLUT_WINDOW_HEIGHT ) - position[1],  1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth );
    std::cout << "depth: " << depth << std::endl;
  }
}

void TerrainRenderer::onSceneRendererChanged( const dp::sg::ui::SceneRendererSharedPtr &sceneRenderer )
{
  if ( sceneRenderer )
  {
    m_shaderManager = sceneRenderer->getShaderManager();
  }
}

/************************************************************************/
/* End of TerrainRendering                                                   */
/************************************************************************/

dp::fx::Manager getShaderManager( std::string const& name )
{
  std::map<std::string, dp::fx::Manager> shaderManager;
  shaderManager["rix:ubo140"] = dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX;
  shaderManager["rixfx:uniform"] = dp::fx::MANAGER_UNIFORM;
  shaderManager["rixfx:shaderbufferload"] = dp::fx::MANAGER_SHADERBUFFER;
  shaderManager["rixfx:ubo140"] = dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX_FX;
  shaderManager["rixfx:ssbo140"] = dp::fx::MANAGER_SHADER_STORAGE_BUFFER_OBJECT;
  if ( shaderManager.find(name) != shaderManager.end() )
  {
    return shaderManager[name];
  }
  else
  {
    return dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX;
  }
}


dp::sg::ui::ViewStateSharedPtr loadTerrain( std::string const& fileHeightMap, std::string const& fileTextureMap, dp::math::Vec3f const& resolution, dp::math::Vec3f const & offset )
{
  dp::sg::ui::ViewStateSharedPtr viewStateHandle = dp::sg::ui::ViewState::create();
  dp::sg::core::SceneSharedPtr scene = dp::sg::core::Scene::create();

  dp::sg::core::GeoNodeSharedPtr geoNode = dp::sg::generator::generateTerrain( fileHeightMap, fileTextureMap, resolution, offset );

  scene->setRootNode(geoNode);

  viewStateHandle->setScene(scene);

  return viewStateHandle;

}

int runApp( options::variables_map const& opts )
{
  // Create renderer

  dp::sg::renderer::rix::gl::SceneRendererSharedPtr renderer = dp::sg::renderer::rix::gl::SceneRenderer::create
  ( 
      opts["renderengine"].as<std::string>().c_str()
    , getShaderManager( opts["shadermanager"].as<std::string>() )
  );
  renderer->setCullingEnabled( false );

  dp::math::Vec3f offset( 0.0f, 0.0f, 0.0f );
  std::vector<float> resolution;
  try
  {
	   resolution = opts["resolution"].as<std::vector<float> >();
  }
  catch (boost::bad_any_cast&)
  {
	  resolution.push_back(640);
	  resolution.push_back(480);
	  resolution.push_back(100);
  }
  if ( !opts["offset"].empty() )
  {
    std::vector<float> ov = opts["resolution"].as<std::vector<float> >();
    if ( ov.size() == 3 )
    {
      offset[0] = ov[0];
      offset[1] = ov[1];
      offset[2] = ov[2];
    }
    else
    {
      std::cerr << "resolution argument count is wrong, skipping." << std::endl;
    }
  }

  dp::sg::ui::ViewStateSharedPtr viewStateHandle = loadTerrain( opts["heightmap"].as<std::string>(), opts["texturemap"].as<std::string>(), dp::math::Vec3f(resolution[0], resolution[1], resolution[2]), offset );

  dp::sg::ui::setupDefaultViewState( viewStateHandle );

  viewStateHandle->setAutoClipPlanes(true);

  if ( !opts["headlight"].empty() )
  {
    if ( viewStateHandle && viewStateHandle->getScene() && !dp::sg::algorithm::containsLight( viewStateHandle->getScene() )
      && viewStateHandle->getCamera() && ( viewStateHandle->getCamera()->getNumberOfHeadLights() == 0 ) )
    {
      // Use the defaults! Note that LightSource ambientColor is black.
      viewStateHandle->getCamera()->addHeadLight( dp::sg::core::createStandardPointLight() );
    }
  }

  // Setup default OpenGL format descriptor
  // We need to create a default format first to be able to check if a stereo pixelformat is available later.
  // (An unfortunate RenderContextFormat.isAvailable() interface due to Linux.)
  dp::gl::RenderContextFormat format;

  // create a widget which shows the scene
  //dp::sg::ui::glut::SceneRendererWidget w( format );
  TerrainRenderer w;

  // TODO format is not yet supported
#if 0
  if (stereo)
  {
    format.setStereo( stereo );
    if ( !w.setFormat( format ) )  // This automatically checks if the format is available.
    {
      std::cout << "Warning: No stereo pixelformat available." << std::endl;
    }
  }
#endif

  viewStateHandle->setAutoClipPlanes( opts["autoclipplanes"].as<bool>() );

  //w.setNumberOfFrames( frames );
  w.setViewState( viewStateHandle );
  w.setSceneRenderer( renderer );
  if ( !opts["continuous"].empty() )
  {
    w.setContinuousUpdate( true );
    w.setShowFrameRate( true );
  }

  if( opts["frames"].as<int>() != -1 )
  {
    w.setNumberOfFrames( opts["frames"].as<int>() );
  }
  w.setDuration( opts["duration"].as<double>() );
  
  w.setWindowSize( 640, 480 );
  //w.show();

  // Keep only once reference to the renderer in the widget. This is necessary since the OpenGL resources
  // used by the renderer must be deleted before the window gets destroyed.
  renderer.reset(); 

  glutMainLoop();

  return w.getExitCode();
}

int main(int argc, char *argv[])
{
#if defined(DP_OS_WINDOWS)
  SetProcessAffinityMask( GetCurrentProcess(), 1 << 4 );
#endif

  // initialize GLUT, set window size and display mode, create the main window
  glutInit( &argc, argv );
  glutSetOption( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION );
  
  options::options_description od("Usage: TerrainRendering");
  od.add_options()
    ( "heightmap", options::value<std::string>(), "height map to use" )
    ( "texturemap", options::value<std::string>(), "texture map to use" )
    ( "resolution", options::value<std::vector<float> >()->multitoken(), "resolution of the height map (--resolution x y height)" )
    ( "offset", options::value<std::vector<float> >()->multitoken(), "offset of the height map in the world space (--offset x y z)" )
    ( "stereo", "enable stereo" )
    ( "continuous", "enable continuous rendering" )
    ( "headlight", "add a headlight to the camera" )
    ( "frames", options::value<int>()->default_value(-1), "benchmark a specific number of frames. The exit code returns the frames per second." )
    ( "duration", options::value<double>()->default_value(0.0), "benchmark for a specific duration. The exit code returns the frames per second." )
    ( "renderengine", options::value<std::string>()->default_value("Bindless"), "choose a renderengine from this list: VBO|VAB|VBOVAO|Bindless|BindlessVAO|DisplayList" )
    ( "shadermanager", options::value<std::string>()->default_value("rixfx:shaderbufferload"), "rixfx:uniform|rixfx:ubo140|rixfx:ssbo140|rixfx:shaderbufferload" )
    ( "autoclipplanes", options::value<bool>()->default_value(true), "enable/disable autoclipplane")
    ( "help", "show help")
    ;

#if 0
  // not yet implemented
  std::cout << "During execution hit 's' for screenshot and 'x' to toggle stereo" << std::endl;
  std::cout << "Stereo screenshots will be saved as side/side png with filename 'stereo.pns'." << std::endl;
  std::cout << "They can be viewed with the 3D Vision Photo Viewer." << std::endl;
#endif

  int result = -1;
  try
  {
    options::variables_map opts;
    options::store( options::parse_command_line( argc, argv, od ), opts );

    if ( dp::util::fileExists( "TerrainRendering.cfg" ) )
    {
      options::store( options::parse_config_file<char>( "TerrainRendering.cfg", od), opts);
    }

    options::notify( opts );

    if ( !opts["help"].empty() )
    {
      std::cout << od << std::endl;
    }

    result = runApp( opts );
  }
  catch ( options::unknown_option e )
  {
    std::cerr << "Unknown option: " << e.get_option_name() << ". ";
    std::cout << od << std::endl;
    std::cerr << "Press enter to continue." << std::endl;
    std::string line;
    getline( std::cin, line );
  }

  return result;
}
