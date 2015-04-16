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
#include <dp/sg/core/TextureFile.h>

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
#include <dp/sg/algorithm/Search.h>
#include <dp/sg/algorithm/SearchTraverser.h>
#include <dp/sg/algorithm/StatisticsTraverser.h>
#include <dp/sg/algorithm/UnifyTraverser.h>

// scenes
#include <dp/sg/generator/GeoSphereScene.h>
#include <dp/sg/generator/SimpleScene.h>
#include <dp/sg/generator/PreviewScene.h>
#include <dp/sg/generator/MeshGenerator.h>

#include <dp/fx/EffectLibrary.h>
#include <dp/util/File.h>
#include <dp/util/FrameProfiler.h>

#include <boost/program_options.hpp>

#include <fstream>

namespace options = boost::program_options;

/************************************************************************/
/* GLUTMinimal                                                          */
/************************************************************************/

class GLUTMinimal : public dp::sg::ui::glut::SceneRendererWidget
{
public:
  GLUTMinimal();
  ~GLUTMinimal();

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

GLUTMinimal::GLUTMinimal()
  : m_trackballHIDSync(new dp::sg::ui::manipulator::TrackballCameraManipulatorHIDSync( ) )
  , m_benchmarkFrames( ~0 )
  , m_renderedFrames( 0 )
  , m_exitCode( 0 )
  , m_duration( 0.0 )
  , m_engineBindless( true )
  , m_attributeType( ATTRIBUTE_GENERIC )
  , m_shaderManager( dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX )
{
  m_trackballHIDSync->setHID( this );
  m_trackballHIDSync->setRenderTarget( getRenderTarget() );
  setManipulator( m_trackballHIDSync.get() );
}

GLUTMinimal::~GLUTMinimal()
{
  setManipulator( 0 );
}

void GLUTMinimal::setNumberOfFrames( dp::Uint32 numberOfFrames )
{
  m_benchmarkFrames = numberOfFrames;
  if( numberOfFrames != ~0 )
  {
    setContinuousUpdate( true );
  }
}

void GLUTMinimal::setDuration( double duration )
{
  m_duration = duration;
  if( duration > 0.0 )
  {
    setContinuousUpdate( true );
  }
}

void GLUTMinimal::paint()
{
  try
  {
    dp::sg::renderer::rix::gl::SceneRendererSharedPtr renderer = getSceneRenderer().staticCast<dp::sg::renderer::rix::gl::SceneRenderer>();
    if ( !m_renderEngine.empty() && renderer->getRenderEngine() != m_renderEngine )
    {
      std::cout << "Setting renderengine: " << m_renderEngine << std::endl;
      renderer->setRenderEngine( m_renderEngine );
    }

    renderer->setShaderManager( m_shaderManager);

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
  catch (std::exception &e)
  {
    std::cout << "caught exception: " << std::endl << e.what() << std::endl;
    glutLeaveMainLoop();
  }
  catch (...)
  {
    std::cout << "caught exception" << std::endl;
    glutLeaveMainLoop();
  }
}

int GLUTMinimal::getExitCode() const
{
  return m_exitCode;
}

dp::sg::core::SamplerSharedPtr sampler;

void GLUTMinimal::onHIDEvent( dp::util::PropertyId propertyId )
{
  SceneRendererWidget::onHIDEvent(propertyId);

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
      std::cout << "Setting shadermanager: " << "uniform buffer object rix" << std::endl;
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
}

void GLUTMinimal::updateSceneRendererEngine()
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

void GLUTMinimal::onSceneRendererChanged( const dp::sg::ui::SceneRendererSharedPtr &sceneRenderer )
{
  if ( sceneRenderer )
  {
    m_shaderManager = sceneRenderer->getShaderManager();
  }
}

/************************************************************************/
/* End of GLUTMinimal                                                   */
/************************************************************************/


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
  shaderManager["rix:ssbo140"] = dp::fx::MANAGER_SHADER_STORAGE_BUFFER_OBJECT_RIX;
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
  dp::sg::ui::ViewStateSharedPtr viewState;
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
    viewState = dp::sg::io::loadScene( filename );
  }
  if ( !viewState )
  {
    if ( !scene )
    {
      std::cerr << "no valid scene found, using SimpleScene" << std::endl;
      dp::sg::generator::SimpleScene simpleScene;
      scene = simpleScene.m_sceneHandle;
    }
    viewState = dp::sg::ui::ViewState::create();
    viewState->setSceneTree( dp::sg::xbar::SceneTree::create( scene ) );
  }
  return viewState;
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

  dp::sg::renderer::rix::gl::SceneRendererSharedPtr renderer = dp::sg::renderer::rix::gl::SceneRenderer::create
  ( 
      opts["renderengine"].as<std::string>().c_str()
    , getShaderManager( opts["shadermanager"].as<std::string>() )
    , cullingMode
  );
  renderer->setCullingEnabled( opts["culling"].as<bool>() );

  if ( !opts["effectlibrary"].empty() )
  {
    dp::fx::EffectLibrary::instance()->loadEffects( opts["effectlibrary"].as<std::string>() );
  }

  dp::sg::ui::ViewStateSharedPtr viewState = loadScene( opts["filename"].as<std::string>() );

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
    dp::sg::algorithm::replaceEffectDatas( viewState->getScene(), replacements );
  }
  else if ( !opts["replaceAll"].empty() )
  {
    dp::sg::core::EffectDataSharedPtr replacement = dp::sg::core::EffectData::create( dp::fx::EffectLibrary::instance()->getEffectData( opts["replaceAll"].as<std::string>() ) );
    DP_ASSERT( replacement );

    dp::sg::algorithm::SearchTraverser searchTraverser;
    searchTraverser.setClassName( "class dp::sg::core::GeoNode" );
    searchTraverser.setBaseClassSearch( true );
    searchTraverser.apply( viewState->getScene() );
    const std::vector<dp::sg::core::ObjectWeakPtr> &vp = searchTraverser.getResults();
    for ( size_t i=0 ; i<vp.size() ; i++ )
    {
      DP_ASSERT( dynamic_cast<dp::sg::core::GeoNodeWeakPtr>(vp[i]) );
      static_cast<dp::sg::core::GeoNodeWeakPtr>(vp[i])->setMaterialEffect( replacement );
    }
  }

  if ( opts.count("gridSize") )
  {
    DP_ASSERT( viewState && viewState->getScene() && viewState->getScene()->getRootNode() );

    std::vector<unsigned int> gridSizes = opts["gridSize"].as<std::vector<unsigned int> >();
    dp::math::Vec3ui gridSize;
    int i;
    for ( i=0 ; i<3 && i<gridSizes.size() ; i++ )
    {
      gridSize[i] = gridSizes[i];
    }
    for ( ; i<3 ; i++ )
    {
      gridSize[i] = 1;
    }

    dp::math::Vec3f gridSpacing = dp::math::Vec3f( 1.0f, 1.0f, 1.0f );
    if ( opts.count("gridSpacing") )
    {
      std::vector<float> gridSpacings = opts["gridSpacing"].as<std::vector<float> >();
      for ( int i=0 ; i<3 && i<gridSpacings.size() ; i++ )
      {
        gridSpacing[i] = gridSpacings[i];
      }
    }

    viewState->getScene()->setRootNode( dp::sg::generator::replicate( viewState->getScene()->getRootNode(), gridSize, gridSpacing, opts["gridClone"].as<bool>() ) );
  }

  if ( !opts["statistics"].empty() )
  {
    showStatistics( viewState );
  }

  dp::sg::ui::setupDefaultViewState( viewState );

  if ( !opts["combineVertexAttributes"].empty() )
  {
    combineVertexAttributes( viewState );
  }

  {
    // Replace MatrixCamera by PerspectiveCamera to get all manipulator features
    if ( viewState->getCamera()->getObjectCode() == dp::sg::core::OC_MATRIXCAMERA )
    {
      dp::sg::core::PerspectiveCameraSharedPtr perspectiveCamera = dp::sg::core::PerspectiveCamera::create();
      perspectiveCamera->setOrientation(viewState->getCamera()->getOrientation());
      perspectiveCamera->setDirection((viewState->getCamera()->getDirection()));
      perspectiveCamera->setPosition(viewState->getCamera()->getPosition());

      viewState->setAutoClipPlanes(true);
      viewState->setCamera(perspectiveCamera);
    }
  }

  if ( !opts["headlight"].empty() )
  {
    if ( viewState && viewState->getScene() && !dp::sg::algorithm::containsLight( viewState->getScene() )
      && viewState->getCamera() && ( viewState->getCamera()->getNumberOfHeadLights() == 0 ) )
    {
      // Use the defaults! Note that LightSource ambientColor is black.
      viewState->getCamera()->addHeadLight( dp::sg::core::createStandardPointLight() );
    }
  }

  if ( !opts["environment"].empty() )
  {
    dp::sg::core::TextureSharedPtr texture = dp::sg::core::TextureFile::create( opts["environment"].as<std::string>() );
    dp::sg::core::SamplerSharedPtr sampler = dp::sg::core::Sampler::create( texture );
    renderer->setEnvironmentSampler( sampler );
  }

  // Setup default OpenGL format descriptor
  // We need to create a default format first to be able to check if a stereo pixelformat is available later.
  // (An unfortunate RenderContextFormat.isAvailable() interface due to Linux.)
  dp::gl::RenderContextFormat format;

  // create a widget which shows the scene
  //dp::sg::ui::glut::SceneRendererWidget w( format );
  GLUTMinimal w;

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

  viewState->setAutoClipPlanes( opts["autoclipplanes"].as<bool>() );

  //w.setNumberOfFrames( frames );
  w.setViewState( viewState );
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

  if ( opts["fullscreen"].empty() )
  {
    size_t width = 640;
    size_t height = 480;

    if ( opts.count("windowSize") )
    {
      std::vector<size_t> sizes = opts["windowSize"].as<std::vector<size_t> >();
      if ( 1 <= sizes.size() )
      {
        width = sizes[0];
        if ( 2 <= sizes.size() )
        {
          height = sizes[1];
        }
      }
    }

    w.setWindowSize( width, height );
  }
  else
  {
    w.setWindowFullScreen();
  }
  //w.show();

  // Keep only once reference to the renderer in the widget. This is necessary since the OpenGL resources
  // used by the renderer must be deleted before the window gets destroyed.
  renderer.reset(); 

  glutMainLoop();

  return w.getExitCode();
}

int main(int argc, char *argv[])
{
  int result = -1;
  try
  {
    // initialize GLUT, set window size and display mode, create the main window
    glutInit( &argc, argv );
    glutSetOption( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION );
  
    options::options_description od("Usage: GLUTMinimal");
    od.add_options()
      ( "autoclipplanes", options::value<bool>()->default_value(true), "enable/disable autoclipplane")
      ( "combineVertexAttributes", "combine all vertexattribute into a single buffer" )
      ( "continuous", "enable continuous rendering" )
      ( "culling", options::value<bool>()->default_value("true"), "enable/disable culling")
      ( "cullingengine", options::value<std::string>()->default_value("auto"), "auto|cpu|cuda|gl_compute")
      ( "duration", options::value<double>()->default_value(0.0), "benchmark for a specific duration. The exit code returns the frames per second." )
      ( "effectlibrary", options::value<std::string>(), "effectlibrary to load for replacements" )
      ( "environment", options::value<std::string>(), "environment texture" )
      ( "filename", options::value<std::string>()->default_value("cubes"), "file to load" )
      ( "frames", options::value<int>()->default_value(-1), "benchmark a specific number of frames. The exit code returns the frames per second." )
      ( "fullscreen", "start in full screen mode" )
      ( "gridClone", options::value<bool>()->default_value(true), "enable/disable cloning of the node to grid" )
      ( "gridSize", options::value< std::vector<unsigned int> >()->composing()->multitoken(), "three-dimensional replication of the scene: x y z" )
      ( "gridSpacing", options::value< std::vector<float> >()->composing()->multitoken(), "three-dimensional spacing of the scene: x y z" )
      ( "headlight", "add a headlight to the camera" )
      ( "help", "show help")
      ( "renderengine", options::value<std::string>()->default_value("Bindless"), "choose a renderengine from this list: VBO|VAB|BVAB|VBOVAO|Bindless|BindlessVAO|DisplayList" )
      ( "replace", options::value< std::vector<std::string> >()->composing()->multitoken(), "file to load" )
      ( "replaceAll", options::value<std::string>(), "EffectData to replace all EffectData in the scene" )
      ( "shadermanager", options::value<std::string>()->default_value("rix:ubo140"), "rixfx:uniform|rixfx:ubo140|rixfx:ssbo140|rixfx:shaderbufferload|rix:ubo140|rix:ssbo140" )
      ( "statistics", "show statistics of scene" )
      ( "stereo", "enable stereo" )
      ( "windowSize", options::value< std::vector<size_t> >()->composing()->multitoken(), "Window size: x y" )
      ;

  #if 0
    // not yet implemented
    std::cout << "During execution hit 's' for screenshot and 'x' to toggle stereo" << std::endl;
    std::cout << "Stereo screenshots will be saved as side/side png with filename 'stereo.pns'." << std::endl;
    std::cout << "They can be viewed with the 3D Vision Photo Viewer." << std::endl;
  #endif

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
  }
  catch (std::runtime_error & e)
  {
    std::cout << "caught exception: " << std::endl << e.what() << std::endl;
  }
  catch (...)
  {
    std::cout << "caught unknown exception: " << std::endl;
  }

  if (dp::gl::RenderContext::getCurrentRenderContext()) {
    dp::gl::RenderContext::getCurrentRenderContext()->makeNoncurrent();
  }

  return result;
}
