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

#include <dp/sg/algorithm/CombineTraverser.h>
#include <dp/sg/algorithm/DeindexTraverser.h>
#include <dp/sg/algorithm/DestrippingTraverser.h>
#include <dp/sg/algorithm/EliminateTraverser.h>
#include <dp/sg/algorithm/Replace.h>
#include <dp/sg/algorithm/SearchTraverser.h>
#include <dp/sg/algorithm/StatisticsTraverser.h>
#include <dp/sg/algorithm/UnifyTraverser.h>

// scenes
#include <dp/sg/generator/GeoSphereScene.h>
#include <dp/sg/generator/SimpleScene.h>
#include <dp/sg/generator/PreviewScene.h>

#include <dp/fx/EffectLibrary.h>
#include <dp/util/File.h>
#include <dp/util/FrameProfiler.h>

#include <boost/program_options.hpp>

#include <inc/CFRPipeline.h>

#include <fstream>
#include <queue>

namespace options = boost::program_options;


std::ofstream file;

/************************************************************************/
/* GLUTMinimalCFR                                                       */
/************************************************************************/

class GLUTMinimalCFR : public dp::sg::ui::glut::SceneRendererWidget
{
public:
  GLUTMinimalCFR();
  ~GLUTMinimalCFR();

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
  std::unique_ptr<dp::sg::ui::manipulator::TrackballCameraManipulatorHIDSync> m_trackballHIDSync;

  void updateSceneRendererEngine();

  enum AttributeType
  {
      ATTRIBUTE_GENERIC // GL 2.x
    , ATTRIBUTE_VAO // GL 3.x
    , ATTRIBUTE_VAB // GL 4.3
  };

  // benchmark
  dp::Uint32          m_renderedFrames;
  dp::Uint32          m_benchmarkFrames;
  dp::util::Timer     m_benchmarkTimer;
  dp::util::Timer     m_benchmarkProgressTimer;
  int                 m_exitCode;
  double              m_duration;

  bool                m_engineBindless;
  AttributeType       m_attributeType;

  std::string         m_renderEngine;
  dp::fx::Manager     m_shaderManager;

  dp::util::Timer     m_globalTimer;
  std::deque<double>  m_paintTimes;
};

GLUTMinimalCFR::GLUTMinimalCFR()
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

  m_globalTimer.start();
  double firstFrame = 2.0;
  m_paintTimes.push_back( firstFrame + 0.0 );
  m_paintTimes.push_back( firstFrame + 1.0 );
  m_paintTimes.push_back( firstFrame + 1.2 );
  m_paintTimes.push_back( firstFrame + 1.4 );
  m_paintTimes.push_back( firstFrame + 2.0 );
  m_paintTimes.push_back( firstFrame + 3.0 );

  file.open("c:\\temp\\output.txt");
  file << "start " << m_globalTimer.getTime() << "\n";
}

GLUTMinimalCFR::~GLUTMinimalCFR()
{
  setManipulator( nullptr );
  file << "stop " << m_globalTimer.getTime() << "\n";
  file.close();
}

void GLUTMinimalCFR::setNumberOfFrames( dp::Uint32 numberOfFrames )
{
  m_benchmarkFrames = numberOfFrames;
  if( numberOfFrames != ~0 )
  {
    setContinuousUpdate( true );
  }
}

void GLUTMinimalCFR::setDuration( double duration )
{
  m_duration = duration;
  if( duration > 0.0 )
  {
    setContinuousUpdate( true );
  }
}

void GLUTMinimalCFR::paint()
{
  //std::cout << "paint\n";
#if 1
  //std::cout << "NOT using m_paintTimes\n";
#else
  double time = m_globalTimer.getTime();
  if( m_paintTimes.empty() )
  {
    glutLeaveMainLoop();
    return;
  }
  if( time < m_paintTimes.front() )
  {
    if( getContinuousUpdate() )
    {
      glutPostRedisplay();
    }
    return;
  }
  m_paintTimes.pop_front();
  file << "paint() " << time << "\n";
#endif

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

int GLUTMinimalCFR::getExitCode() const
{
  return m_exitCode;
}

void GLUTMinimalCFR::onHIDEvent( dp::util::PropertyId propertyId )
{
  if ( propertyId == PID_Key_F1 )
  {
    if ( getValue<bool>( propertyId ) )
    {
      getSceneRenderer()->setCullingMode(dp::culling::MODE_CPU);
      std::cout << "culling: CPU" << std::endl;
    }
  }
  else if ( propertyId == PID_Key_F2 )
  {
    if ( getValue<bool>( propertyId ) )
    {
      getSceneRenderer()->setCullingMode(dp::culling::MODE_OPENGL_COMPUTE);
      std::cout << "culling: GL_COMPUTE" << std::endl;
    }
  }
  else if ( propertyId == PID_Key_F3 )
  {
    if ( getValue<bool>( propertyId ) )
    {
      getSceneRenderer()->setCullingMode(dp::culling::MODE_CUDA);
      std::cout << "culling: CUDA" << std::endl;
    }
  }
  else if ( propertyId == PID_Key_F4 )
  {
    if ( getValue<bool>( propertyId ) )
    {
      getSceneRenderer()->setCullingMode(dp::culling::MODE_AUTO);
      std::cout << "culling: AUTO" << std::endl;
    }
  }
  else if ( propertyId == PID_Key_C )
  {
    if ( getValue<bool>( propertyId) )
    {
      bool enabled = !getSceneRenderer()->isCullingEnabled();
      getSceneRenderer()->setCullingEnabled(enabled);
      std::cout << "culling " << (enabled ? "enabled" : "disabled") << std::endl;
    }
  }
  else if ( propertyId == PID_Key_B ) // Toggle Bindless
  {
    if ( getValue<bool>( propertyId ) )
    {
      m_engineBindless = !m_engineBindless;
      updateSceneRendererEngine();
    }
  }
  else if ( propertyId == PID_Key_G ) // attrib generic
  {
    if ( getValue<bool>( propertyId ) )
    {
      m_attributeType = ATTRIBUTE_GENERIC;
      updateSceneRendererEngine();
    }
  }

  else if ( propertyId == PID_Key_V ) // attrib VAO
  {
    if ( getValue<bool>( propertyId ) )
    {
      m_attributeType = ATTRIBUTE_VAO;
      updateSceneRendererEngine();
    }
  }

  else if ( propertyId == PID_Key_A ) // Toggle attrib VAB
  {
    if ( getValue<bool>( propertyId ) )
    {
      m_attributeType = ATTRIBUTE_VAB;
      updateSceneRendererEngine();
    }
  }
  else if ( propertyId == PID_Key_Escape )
  {
    if ( getValue<bool>( propertyId ) )
    {
      glutLeaveMainLoop();
    }
  }
  else if ( propertyId == PID_Key_F9 )
  {
    if ( getValue<bool>( propertyId ) )
    {
      std::cout << "Setting shadermanager: " << "uniform" << std::endl;
      m_shaderManager = dp::fx::MANAGER_UNIFORM;
    }
  }
  else if ( propertyId == PID_Key_F10 )
  {
    if ( getValue<bool>( propertyId ) )
    {
      std::cout << "Setting shadermanager: " << "uniform buffer object" << std::endl;
      m_shaderManager = dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX;
    }
  }
  else if ( propertyId == PID_Key_F11 )
  {
    if ( getValue<bool>( propertyId ) )
    {
      std::cout << "Setting shadermanager: " << "shaderbufferload" << std::endl;
      m_shaderManager = dp::fx::MANAGER_SHADERBUFFER;
    }
  }
  else if ( propertyId == PID_Key_F12 )
  {
    if ( getValue<bool>( propertyId ) )
    {
      std::cout << "Setting shadermanager: " << "shader storage buffer object" << std::endl;
      m_shaderManager = dp::fx::MANAGER_SHADER_STORAGE_BUFFER_OBJECT;
    }
  }
  else if ( propertyId == PID_Key_P )
  {
    if ( getValue<bool>( propertyId ) )
    {
      dp::util::FrameProfiler::instance().setEnabled( !dp::util::FrameProfiler::instance().isEnabled() );
    }
  }
  else if ( propertyId == PID_Key_Space )
  {
    if ( getValue<bool>( propertyId ) )
    {
      paint();
    }
  }
}

void GLUTMinimalCFR::updateSceneRendererEngine()
{
  std::string engine;
  if ( m_engineBindless )
  {
    switch ( m_attributeType )
    {
    case ATTRIBUTE_GENERIC:
      engine = "Bindless";
    break;
    case ATTRIBUTE_VAO:
      engine = "BindlessVAO";
      break;
    case ATTRIBUTE_VAB:
      engine = "BVAB";
      break;
    }
  }
  else
  {
    switch ( m_attributeType )
    {
    case ATTRIBUTE_GENERIC:
      engine = "VBO";
      break;
    case ATTRIBUTE_VAO:
      engine = "VBOVAO";
      break;
    case ATTRIBUTE_VAB:
      engine = "VAB";
      break;
    }
  }

  DP_ASSERT( !engine.empty() );

  m_renderEngine = engine;
}

void GLUTMinimalCFR::onSceneRendererChanged( const dp::sg::ui::SceneRendererSharedPtr &sceneRenderer )
{
  if ( sceneRenderer )
  {
    m_shaderManager = sceneRenderer->getShaderManager();
  }
}

/************************************************************************/
/* End of GLUTMinimal                                                   */
/************************************************************************/

//
// global variables
// 

// the scene's viewstate
dp::sg::ui::ViewStateSharedPtr g_viewState;
// number of lights in the scene
const size_t g_numLights = 49;
// lights in the scene
dp::sg::core::LightSourceSharedPtr g_lightSources[g_numLights];


void combineVertexAttributes( dp::sg::ui::ViewStateSharedPtr const& viewState )
{
  dp::sg::algorithm::SearchTraverser searchTraverser;
  searchTraverser.setClassName("class dp::sg::core::VertexAttributeSet");
  searchTraverser.apply( viewState );
  std::vector<dp::sg::core::ObjectWeakPtr> results = searchTraverser.getResults();
  for ( std::vector<dp::sg::core::ObjectWeakPtr>::iterator it = results.begin(); it != results.end(); ++it )
  {
    dp::util::weakPtr_cast<dp::sg::core::VertexAttributeSet>(*it)->combineBuffers();
  }
}

void showStatistics( dp::sg::ui::ViewStateSharedPtr const& viewState )
{
  dp::sg::algorithm::StatisticsTraverser statisticsTraverser;
  statisticsTraverser.apply( viewState );
  dp::sg::algorithm::Statistics const* statistics = statisticsTraverser.getStatistics();
  std::cout << "vertices : " << statistics->m_statVertexAttributeSet.m_numberOfVertices << std::endl;
  std::cout << "faces: " << statistics->m_statPrimitives.m_faces << std::endl;
  std::cout << "faces instances: " << statistics->m_statPrimitiveInstances.m_faces << std::endl;
  std::cout << "primitives unique: " << statistics->m_statPrimitives.m_count << std::endl;
  std::cout << "primitives instanced: " << statistics->m_statPrimitives.m_instanced<< std::endl;
  std::cout << "groups unique: " << statistics->m_statGroup.m_count << std::endl;
  std::cout << "group referenced: " << statistics->m_statGroup.m_referenced << std::endl;
  std::cout << "transforms unique: " << statistics->m_statTransform.m_count << std::endl;
  std::cout << "transform referenced: " << statistics->m_statTransform.m_referenced << std::endl;
}

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

dp::sg::ui::ViewStateSharedPtr loadScene( std::string const& filename )
{
  dp::sg::ui::ViewStateSharedPtr viewStateHandle;
  dp::sg::core::SceneSharedPtr scene;
  if ( filename == "cubes" )
  {
    dp::sg::generator::SimpleScene simpleScene;
    scene = simpleScene.m_sceneHandle;
  }
  else if ( filename == "preview")
  {
    PreviewScene previewScene;  // Vertex_3f, TexCoord0_3f, Tangent_3f, Binormal_3f, Normal_3f on all five objects. Only front faces visible, no thin geometry.
    scene = previewScene.m_sceneHandle;
  }
  else if ( filename == "geosphere")
  {
    GeoSphereScene geoSphereScene; // Vertex_3f, Normal_3f. Only front faces visible, no thin geometry.
    scene = geoSphereScene.m_sceneHandle;
  }
  else
  {
    viewStateHandle = dp::sg::io::loadScene( filename );
  }
  if ( !viewStateHandle )
  {
    if ( !scene )
    {
      std::cerr << "no valid scene found, using SimpleScene" << std::endl;
      dp::sg::generator::SimpleScene simpleScene;
      scene = simpleScene.m_sceneHandle;
    }
    viewStateHandle = dp::sg::ui::ViewState::create();
    viewStateHandle->setSceneTree( dp::sg::xbar::SceneTree::create( scene ) );
  }
  return viewStateHandle;
}

void setLights( size_t counter = ~0 )
{
  if( !g_lightSources[0] )
  {
    dp::sg::core::GroupSharedPtr const& rootPtr = g_viewState->getScene()->getRootNode().staticCast<dp::sg::core::Group>();
    DP_ASSERT( rootPtr );

    // add own lights to the root node
    dp::math::Vec3f firstLightPos = g_viewState->getCamera()->getPosition();
    for( size_t i = 0; i < g_numLights; ++i )
    {
      g_lightSources[i] = dp::sg::core::createStandardPointLight( firstLightPos + dp::math::Vec3f( (float)2*i, 0.0f, 0.0f ) );
      rootPtr->addChild( g_lightSources[i] );
    }
  }

  for( size_t i = 0; i < g_numLights; ++i )
  {
    g_lightSources[i]->setEnabled( !!( counter & (1i64 << i) ) );
  }
}

int runApp( options::variables_map const& opts )
{
  // Create renderer
  std::string cullingEngine = opts["cullingengine"].as<std::string>();
  dp::culling::Mode cullingMode = dp::culling::MODE_AUTO;
  if ( cullingEngine == "cpu" )
  {
    cullingMode = dp::culling::MODE_CPU;
  }
  else if ( cullingEngine == "gl_compute" )
  {
    cullingMode = dp::culling::MODE_OPENGL_COMPUTE;
  }
  else if ( cullingEngine == "cuda" )
  {
    cullingMode = dp::culling::MODE_CUDA;
  }
  else if ( cullingEngine != "auto" )
  {
    std::cerr << "unknown culling engine, abort" << std::endl;
    return -1;
  }

  CFRPipelineSharedPtr renderer = CFRPipeline::create
  ( 
      opts["renderengine"].as<std::string>().c_str()
    , getShaderManager( opts["shadermanager"].as<std::string>() )
    , cullingMode
  );
  //renderer->setCullingEnabled( opts["culling"].as<bool>() );

  dp::sg::ui::ViewStateSharedPtr viewStateHandle = loadScene( opts["filename"].as<std::string>() );

  g_viewState = viewStateHandle;

  if ( opts.count("replace") )
  {
    // process replacements
    std::vector< std::string> replacementStrings = opts["replace"].as< std::vector<std::string > >();
    dp::sg::algorithm::ReplacementMapNames replacements;
    for ( std::vector<std::string>::iterator it = replacementStrings.begin(); it != replacementStrings.end(); ++it )
    {
      size_t equalChar = it->find_first_of(':');
      if ( equalChar != std::string::npos && equalChar < it->size() - 1)
      {
        std::string str1 = it->substr( 0, equalChar );
        std::string str2 = it->substr( equalChar + 1, it->size() - equalChar - 1);
        replacements[str1] = str2;
      }
      else
      {
        std::cerr << "invalid replacement token: " << *it << std::endl;
      }
    }
    dp::sg::algorithm::replaceEffectDatas( viewStateHandle->getScene(), replacements );
  }

  if ( !opts["statistics"].empty() )
  {
    showStatistics( viewStateHandle );
  }

  dp::sg::ui::setupDefaultViewState( viewStateHandle );

  if ( !opts["combineVertexAttributes"].empty() )
  {
    combineVertexAttributes( viewStateHandle );
  }

  {
    // Replace MatrixCamera by PerspectiveCamera to get all manipulator features
    if ( viewStateHandle->getCamera()->getObjectCode() == dp::sg::core::OC_MATRIXCAMERA )
    {
      dp::sg::core::PerspectiveCameraSharedPtr perspectiveCamera = dp::sg::core::PerspectiveCamera::create();
      perspectiveCamera->setOrientation(viewStateHandle->getCamera()->getOrientation());
      perspectiveCamera->setDirection((viewStateHandle->getCamera()->getDirection()));
      perspectiveCamera->setPosition(viewStateHandle->getCamera()->getPosition());

      viewStateHandle->setAutoClipPlanes(true);
      viewStateHandle->setCamera(perspectiveCamera);
    }
  }

  if ( !opts["headlight"].empty() )
  {
    // TODO is this still a bug?
    // Bug 914976 containsLight() doesn't find lights in the scene. Force adding the headlight anyway when the user specified it.
    if ( viewStateHandle /* && viewStateHandle->getScene() && !SceneLock( viewStateHandle->getScene() )->containsLight() */
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
  GLUTMinimalCFR w;

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

  w.setViewState( viewStateHandle );
  w.setSceneRenderer( renderer );
//always on  if ( !opts["continuous"].empty() )
  {
    w.setContinuousUpdate( true );
    w.setShowFrameRate( true );
  }

  if( opts["frames"].as<int>() != -1 )
  {
    w.setNumberOfFrames( opts["frames"].as<int>() );
  }
  w.setDuration( opts["duration"].as<double>() );
  
  w.setWindowSize( 1280, 720 );
  //w.show();

  // Keep only once reference to the renderer in the widget. This is necessary since the OpenGL resources
  // used by the renderer must be deleted before the window gets destroyed.
  renderer.reset(); 

  g_viewState->getCamera()->setPosition(dp::math::Vec3f(0.0f, 0.0f, 5.0f));
  setLights();

  glutMainLoop();

  return w.getExitCode();
}

int main(int argc, char *argv[])
{
  // initialize GLUT, set window size and display mode, create the main window
  glutInit( &argc, argv );
  glutSetOption( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION );
  
  options::options_description od("Usage: GLUTMinimal");
  od.add_options()
    ( "filename", options::value<std::string>()->default_value("cubes"), "file to load" )
    ( "effectlibrary", options::value<std::string>(), "effectlibrary to load for replacements" )
    ( "replace", options::value< std::vector<std::string> >()->composing()->multitoken(), "file to load" )
    ( "stereo", "enable stereo" )
//always on     ( "continuous", "enable continuous rendering" )
    ( "headlight", "add a headlight to the camera" )
    ( "statistics", "show statistics of scene" )
    ( "combineVertexAttributes", "combine all vertexattribute into a single buffer" )
    ( "frames", options::value<int>()->default_value(-1), "benchmark a specific number of frames. The exit code returns the frames per second." )
    ( "duration", options::value<double>()->default_value(0.0), "benchmark for a specific duration. The exit code returns the frames per second." )
    ( "renderengine", options::value<std::string>()->default_value("Bindless"), "choose a renderengine from this list: VBO|VAB|VBOVAO|Bindless|BindlessVAO|DisplayList" )
    ( "shadermanager", options::value<std::string>()->default_value("rixfx:shaderbufferload"), "rixfx:uniform|rixfx:ubo140|rixfx:ssbo140|rixfx:shaderbufferload" )
    ( "cullingengine", options::value<std::string>()->default_value("cpu"), "auto|cpu|cuda|gl_compute")
    ( "culling", options::value<bool>()->default_value(true), "enable/disable culling")
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

    if ( dp::util::fileExists( "GLUTMinimal.cfg" ) )
    {
      options::store( options::parse_config_file<char>( "GLUTMinimal.cfg", od), opts);
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
