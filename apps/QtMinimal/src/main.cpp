// Copyright NVIDIA Corporation 2009-2011
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


#include <QApplication>
#include <QElapsedTimer>
#include <dp/sg/ui/qt5/SceniXQGLSceneRendererWidget.h>

#include <fstream>

#include <boost/assign/list_of.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>

#include <dp/fx/EffectLibrary.h>
#include <dp/sg/algorithm/Replace.h>
#include <dp/sg/core/nvsg.h>
#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/core/TextureFile.h>
#include <dp/sg/algorithm/Optimize.h>
#include <dp/sg/algorithm/Search.h>
#include <dp/sg/generator/GeoSphereScene.h>
#include <dp/sg/generator/MeshGenerator.h>
#include <dp/sg/generator/PreviewScene.h>
#include <dp/sg/generator/SimpleScene.h>
#include <dp/sg/io/IO.h>
#include <dp/sg/renderer/rix/gl/SceneRenderer.h>
#include <dp/sg/ui/manipulator/TrackballCameraManipulatorHIDSync.h>
#include <dp/util/File.h>

namespace options = boost::program_options;


class QtMinimalWidget : public dp::sg::ui::qt5::SceniXQGLSceneRendererWidget
{
  public:
    QtMinimalWidget( const dp::gl::RenderContextFormat &format );
    virtual ~QtMinimalWidget();

  public:
    void setDuration( double duration );
    void setNumberOfFrames( unsigned int numberOfFrames );
    void setOrbit( float orbitDegree );
    void setSceneName( std::string const& name );
    void setShowFrameRate( bool showFPS );

  private:
    void keyPressEvent( QKeyEvent *event);
    void paintGL();
    void screenshot();

    virtual void resizeEvent( QResizeEvent *event );

  protected:
    double                                                        m_benchmarkDuration;
    unsigned int                                                  m_benchmarkFrameCount;
    unsigned int                                                  m_benchmarkFrames;
    double                                                        m_benchmarkTime;
    unsigned int                                                  m_frameCount;
    unsigned int                                                  m_framesInSecond;
    dp::util::Timer                                               m_frameRateTimer;
    float                                                         m_orbitRadians;
    std::string                                                   m_sceneName;
    bool                                                          m_showFPS;
    dp::sg::ui::manipulator::TrackballCameraManipulatorHIDSync  * m_trackballHIDSync;
    std::string                                                   m_windowTitle;
};

QtMinimalWidget::QtMinimalWidget( const dp::gl::RenderContextFormat &format )
  : SceniXQGLSceneRendererWidget(0, format )
  , m_benchmarkDuration( 0.0 )
  , m_benchmarkFrameCount( 0 )
  , m_benchmarkFrames( ~0 )
  , m_benchmarkTime( 0.0 )
  , m_frameCount( 0 )
  , m_framesInSecond( ~0 )
  , m_orbitRadians( 0.0f )
  , m_windowTitle( "QtMinimal" )
{
  m_trackballHIDSync = new dp::sg::ui::manipulator::TrackballCameraManipulatorHIDSync( );
  m_trackballHIDSync->setHID( this );
  m_trackballHIDSync->setRenderTarget( getRenderTarget() );
  setManipulator( m_trackballHIDSync );
}

QtMinimalWidget::~QtMinimalWidget()
{
  // Delete SceneRenderer here to cleanup resources before the OpenGL context dies
  setSceneRenderer( 0 );

  // Reset Manipulator
  setManipulator( 0 );
  delete m_trackballHIDSync;
}

void QtMinimalWidget::keyPressEvent( QKeyEvent *event )
{
  SceniXQGLWidget::keyPressEvent( event );

  if ( ( event->text().compare( " " ) == 0 ) && ! getContinuousUpdate() )
  {
    repaint();
  }
  else if ( event->text().compare( "d" ) == 0 )
  {
    dp::sg::renderer::rix::gl::SmartSceneRenderer renderer = dp::util::smart_cast<dp::sg::renderer::rix::gl::SceneRenderer>( getSceneRenderer() );
    renderer->setDepthPass( ! renderer->getDepthPass() );
  }
  else if ( event->text().compare("s") == 0 )
  {
    screenshot();
  }
  else if ( event->text().compare( "o" ) == 0 )
  {
    dp::sg::algorithm::optimizeScene( getViewState()->getScene(), true, true
                                    , dp::sg::algorithm::CombineTraverser::CT_ALL_TARGETS_MASK
                                    , dp::sg::algorithm::EliminateTraverser::ET_ALL_TARGETS_MASK
                                    , dp::sg::algorithm::UnifyTraverser::UT_ALL_TARGETS_MASK
                                    , FLT_EPSILON );
  }
  else if ( event->text().compare("x") == 0 )
  {
    dp::gl::RenderContextFormat format = getFormat();
    format.setStereo( !format.isStereo() );
    setFormat( format );
  }
}

void QtMinimalWidget::paintGL()
{
  ++m_benchmarkFrameCount;

  SceniXQGLSceneRendererWidget::paintGL();

  if ( getContinuousUpdate() && ( m_orbitRadians != 0.0f ) )
  {
    DP_ASSERT( getViewState() && getViewState()->getCamera() );
    getViewState()->getCamera()->orbitY( getViewState()->getCamera()->getFocusDistance(), m_orbitRadians );
  }

  // fps counter
  if ( m_showFPS )
  {
    if ( m_benchmarkFrameCount == 1 )
    {
      m_frameRateTimer.start();
    }
    else
    {
      DP_ASSERT( 1 < m_benchmarkFrameCount );
      double elapsedSeconds = m_frameRateTimer.getTime();
      ++m_frameCount;

      if ( elapsedSeconds > 1.0 )
      {
        double fps = double(m_frameCount) / elapsedSeconds;
        m_benchmarkTime += elapsedSeconds;

        std::ostringstream windowTitle;
        windowTitle.precision(2);
        windowTitle.setf( std::ios::fixed, std::ios::floatfield );
        windowTitle << m_windowTitle << ", " << fps << " FPS";

        if ( m_benchmarkFrames != ~0 )
        {
          windowTitle << " Benchmark " << m_benchmarkFrameCount-1 << "/" << m_benchmarkFrames;
        }
        else if ( 0.0 < m_benchmarkDuration )
        {
          windowTitle << " Benchmark " << m_benchmarkTime << "/" << m_benchmarkDuration;
        }

        setWindowTitle( windowTitle.str().c_str() );

        m_frameCount = 0;
        m_frameRateTimer.restart();
      }
    }
  }
  if (  ( ( m_benchmarkFrames != ~0 ) && ( m_benchmarkFrames == m_benchmarkFrameCount-1 ) )
     || ( ( 0.0 < m_benchmarkDuration ) && ( m_benchmarkDuration <= m_benchmarkTime ) ) )
  {
    m_frameRateTimer.stop();
    QCoreApplication::instance()->exit( int(100 * double(m_benchmarkFrameCount-1) / m_benchmarkTime) );
  }
}

void QtMinimalWidget::screenshot()
{
  std::string filename = getRenderTarget()->isStereoEnabled() ? "stereo.pns" : "mono.png";
  dp::util::imageToFile( getRenderTarget()->getImage(), filename );
}

void QtMinimalWidget::setDuration( double duration )
{
  m_benchmarkDuration = duration;
  m_benchmarkTime = 0.0;
  if( duration > 0.0 )
  {
    setContinuousUpdate( true );
  }
}

void QtMinimalWidget::setNumberOfFrames( unsigned int numberOfFrames )
{
  DP_ASSERT( numberOfFrames != ~0 );
  m_benchmarkFrames = numberOfFrames;
  m_benchmarkTime = 0.0;
  setContinuousUpdate( true );
}

void QtMinimalWidget::setOrbit( float orbitDegree )
{
  m_orbitRadians = dp::math::degToRad( orbitDegree );
}

void QtMinimalWidget::setSceneName( std::string const& name )
{
  m_sceneName = name;
}

void QtMinimalWidget::setShowFrameRate( bool showFPS )
{
  m_showFPS = showFPS;
}

void QtMinimalWidget::resizeEvent( QResizeEvent *re )
{
  dp::sg::ui::qt5::SceniXQGLSceneRendererWidget::resizeEvent( re );

  std::ostringstream windowTitle;
  windowTitle << "QtMinimal: " << m_sceneName << " (" << re->size().width() << "," << re->size().height() << ")";
  m_windowTitle = windowTitle.str();
}

void combineVertexAttributes( dp::sg::ui::ViewStateSharedPtr const& viewState )
{
  DP_ASSERT( viewState && viewState->getScene() && viewState->getScene()->getRootNode() );
  std::vector<dp::sg::core::ObjectWeakPtr> results = dp::sg::algorithm::searchClass( viewState->getScene()->getRootNode(), "class::dp::sg::core::VertexAttributeSet" );
  for ( std::vector<dp::sg::core::ObjectWeakPtr>::iterator it = results.begin(); it != results.end(); ++it )
  {
    dp::sg::core::weakPtr_cast<dp::sg::core::VertexAttributeSet>(*it)->combineBuffers();
  }
}

dp::culling::Mode getCullingMode( std::string const& name )
{
  static const std::map<std::string,dp::culling::Mode> cullingModes = boost::assign::map_list_of
    ( "cpu", dp::culling::MODE_CPU )
    ( "gl_compute", dp::culling::MODE_OPENGL_COMPUTE )
    ( "cuda", dp::culling::MODE_CUDA )
    ( "auto", dp::culling::MODE_AUTO );

  dp::culling::Mode mode = dp::culling::MODE_AUTO;
  std::map<std::string,dp::culling::Mode>::const_iterator it = cullingModes.find( name );
  if ( it != cullingModes.end() )
  {
    mode = it->second;
  }
  else
  {
    std::cerr << "Unknown culling mode <" << name << ">. Using MODE_AUTO instead.\n";
    DP_ASSERT( !"Unknown culling mode" );
  }
  return( mode );
}

dp::fx::Manager getShaderManager( std::string const& name )
{
  static const std::map<std::string,dp::fx::Manager> shaderManagers = boost::assign::map_list_of
    ( "rix:ubo140",             dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX )
    ( "rixfx:uniform",          dp::fx::MANAGER_UNIFORM )
    ( "rixfx:shaderbufferload", dp::fx::MANAGER_SHADERBUFFER )
    ( "rixfx:ubo140",           dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX_FX )
    ( "rixfx:ssbo140",          dp::fx::MANAGER_SHADER_STORAGE_BUFFER_OBJECT );

  dp::fx::Manager manager = dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX;
  std::map<std::string,dp::fx::Manager>::const_iterator it = shaderManagers.find( name );
  if ( it != shaderManagers.end() )
  {
    manager = it->second;
  }
  else
  {
    std::cerr << "Unknown shader manager <" << name << ">. Using MANAGER_UNIFORM_BUFFER_OBJECT_RIX instead.\n";
    DP_ASSERT( !"Unknown shader manager" );
  }
  return( manager );
}

dp::sg::core::TextureCoordType getTextureCoordType( std::string const& name )
{
  static const std::map<std::string,dp::sg::core::TextureCoordType> textureCoordTypes = boost::assign::map_list_of
    ( "cylindrical",  dp::sg::core::TCT_CYLINDRICAL )
    ( "planar",       dp::sg::core::TCT_PLANAR )
    ( "spherical",    dp::sg::core::TCT_SPHERICAL );

  dp::sg::core::TextureCoordType tct = dp::sg::core::TCT_PLANAR;
  std::map<std::string,dp::sg::core::TextureCoordType>::const_iterator it = textureCoordTypes.find( name );
  if ( it != textureCoordTypes.end() )
  {
    tct = it->second;
  }
  else
  {
    std::cerr << "Unknown texture coord type <" << name << ">. Using TCT_PLANAR instead.\n";
    DP_ASSERT( !"Unknown texture coord type" );
  }
  return( tct );
}

dp::sg::renderer::rix::gl::TransparencyMode getTransparencyMode( std::string const& name )
{
  static const std::map<std::string,dp::sg::renderer::rix::gl::TransparencyMode> transparencyModes = boost::assign::map_list_of
    ( "none", dp::sg::renderer::rix::gl::TM_NONE )
    ( "OITAll", dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_ALL )
    ( "OITClosestArray", dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_CLOSEST_ARRAY )
    ( "OITClosestList", dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_CLOSEST_LIST )
    ( "SB", dp::sg::renderer::rix::gl::TM_SORTED_BLENDED );

  dp::sg::renderer::rix::gl::TransparencyMode mode = dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_CLOSEST_LIST;
  std::map<std::string,dp::sg::renderer::rix::gl::TransparencyMode>::const_iterator it = transparencyModes.find( name );
  if ( it != transparencyModes.end() )
  {
    mode = it->second;
  }
  else
  {
    std::cerr << "Unknown transparency mode <" << name << ">. Using TM_ORDER_INDEPENDENT_CLOSEST_LIST instead.\n";
    DP_ASSERT( !"Unknown transparency mode" );
  }
  return( mode );
}

dp::sg::ui::ViewStateSharedPtr loadScene( std::string const& filename )
{
  dp::sg::ui::ViewStateSharedPtr viewState;
  dp::sg::core::SceneSharedPtr scene;
  if ( filename == "cubes" )
  {
    SimpleScene simpleScene;
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
      SimpleScene simpleScene;
      scene = simpleScene.m_sceneHandle;
    }
    viewState = dp::sg::ui::ViewState::create();
    viewState->setScene( scene );
  }
  return viewState;
}

void showStatistics( dp::sg::ui::ViewStateSharedPtr const& viewState )
{
  dp::util::SmartPtr<dp::sg::algorithm::StatisticsTraverser> statisticsTraverser = new dp::sg::algorithm::StatisticsTraverser;
  statisticsTraverser->apply( viewState );
  dp::sg::algorithm::Statistics const* statistics = statisticsTraverser->getStatistics();
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


int runApp( int argc, char *argv[], options::variables_map const& opts )
{
  QApplication app( argc, argv );

  // Create rendering engine
  dp::sg::renderer::rix::gl::SmartSceneRenderer renderer = dp::sg::renderer::rix::gl::SceneRenderer::create
  (
      opts["renderengine"].as<std::string>().c_str()
    , getShaderManager( opts["shadermanager"].as<std::string>() )
    , getCullingMode( opts["cullingengine"].as<std::string>() )
    , getTransparencyMode( opts["transparency"].as<std::string>() )
  );
  renderer->setCullingEnabled( opts["culling"].as<bool>() );
  renderer->setDepthPass( opts["depthPass"].as<bool>() );

  if ( !opts["effectlibrary"].empty() )
  {
    dp::fx::EffectLibrary::instance()->loadEffects( opts["effectlibrary"].as<std::string>() );
  }

  if ( !opts["environment"].empty() )
  {
    dp::sg::core::TextureSharedPtr texture = dp::sg::core::TextureFile::create( opts["environment"].as<std::string>() );
    dp::sg::core::SamplerSharedPtr sampler = dp::sg::core::Sampler::create( texture );
    renderer->setEnvironmentSampler( sampler );
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

    DP_ASSERT( viewState && viewState->getScene() && viewState->getScene()->getRootNode() );
    const std::vector<dp::sg::core::ObjectWeakPtr> vp = dp::sg::algorithm::searchClass( viewState->getScene()->getRootNode(), "class dp::sg::core::GeoNode", true );
    for ( size_t i=0 ; i<vp.size() ; i++ )
    {
      DP_ASSERT( dynamic_cast<dp::sg::core::GeoNodeWeakPtr>(vp[i]) );
      static_cast<dp::sg::core::GeoNodeWeakPtr>(vp[i])->setMaterialEffect( replacement );
    }
  }

  if ( !opts["generateTexCoords"].empty() )
  {
    dp::sg::core::TextureCoordType tct = getTextureCoordType( opts["generateTexCoords"].as<std::string>() );

    DP_ASSERT( viewState && viewState->getScene() && viewState->getScene()->getRootNode() );
    const std::vector<dp::sg::core::ObjectWeakPtr> vp = dp::sg::algorithm::searchClass( viewState->getScene()->getRootNode(), "class dp::sg::core::Primitive", true );
    for ( size_t i=0 ; i<vp.size() ; i++ )
    {
      DP_ASSERT( dynamic_cast<dp::sg::core::PrimitiveWeakPtr>(vp[i]) );
      static_cast<dp::sg::core::PrimitiveWeakPtr>(vp[i])->generateTexCoords( tct, dp::sg::core::VertexAttributeSet::NVSG_TEXCOORD0, false );
      // don't overwrite if there already are some texture coordinate                                                               ^^^^^
    }
  }

  if ( !opts["generateTangentSpace"].empty() )
  {
    DP_ASSERT( viewState && viewState->getScene() && viewState->getScene()->getRootNode() );
    const std::vector<dp::sg::core::ObjectWeakPtr> vp = dp::sg::algorithm::searchClass( viewState->getScene()->getRootNode(), "class dp::sg::core::Primitive", true );
    for ( size_t i=0 ; i<vp.size() ; i++ )
    {
      DP_ASSERT( dynamic_cast<dp::sg::core::PrimitiveWeakPtr>(vp[i]) );
      static_cast<dp::sg::core::PrimitiveWeakPtr>(vp[i])->generateTangentSpace( dp::sg::core::VertexAttributeSet::NVSG_TEXCOORD0
                                                                              , dp::sg::core::VertexAttributeSet::NVSG_TANGENT
                                                                              , dp::sg::core::VertexAttributeSet::NVSG_BINORMAL
                                                                              , false );
      // don't overwrite if there already are some texture coordinate           ^^^^^
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

  viewState->setAutoClipPlanes( opts["autoclipplanes"].as<bool>() );

  if ( !opts["zoom"].empty() )
  {
    DP_ASSERT( viewState && viewState->getCamera() );
    viewState->getCamera()->zoom( opts["zoom"].as<float>() );
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

  // Setup default OpenGL format descriptor
  // We need to create a default format first to be able to check if a stereo pixelformat is available later.
  // (An unfortunate RenderContextFormat.isAvailable() interface due to Linux.)
  dp::gl::RenderContextFormat format;

  // create a widget which shows the scene
  QtMinimalWidget w( format );
  w.setSceneName( opts["filename"].as<std::string>() );
  w.setViewState( viewState );
  w.setSceneRenderer( renderer );

  if ( opts["multiSample"].as<unsigned int>() != 0 )
  {
    unsigned int samples = opts["multiSample"].as<unsigned int>();
    unsigned int coverage = std::max( samples, opts["multiSampleCoverage"].as<unsigned int>() );
    format.setMultisampleCoverage( samples, coverage );
    if ( !w.setFormat( format ) )
    {
      std::cout << "Warning: No Coverage Sampling Antialiasing ( " << samples << "/" << coverage << " ) pixelformat available." << std::endl;
      if ( samples != coverage )
      {
        format.setMultisample( samples );
        if ( !w.setFormat( format ) )
        {
          std::cout << "Warning: No Multisample Antialiasing ( " << samples << " ) pixelformat available." << std::endl;
        }
      }
    }
  }

  if ( !opts["stereo"].empty() )
  {
    format.setStereo( true );
    if ( !w.setFormat( format ) )  // This automatically checks if the format is available.
    {
      std::cout << "Warning: No stereo pixelformat available." << std::endl;
    }
  }

  if( opts["frames"].as<unsigned int>() != ~0 )
  {
    w.setNumberOfFrames( opts["frames"].as<unsigned int>() );
  }
  w.setDuration( opts["duration"].as<double>() );

  if ( !opts["continuous"].empty() )
  {
    w.setContinuousUpdate( true );
    w.setShowFrameRate( true );
  }

  if ( !opts["orbit"].empty() )
  {
    w.setOrbit( opts["orbit"].as<float>() );
  }

  if ( !opts["fullscreen"].empty() )
  {
    w.showFullScreen();
  }
  else if ( !opts["maximized"].empty() )
  {
    w.showMaximized();
  }
  else
  {
    int width = 640;
    int height = 480;

    if ( opts.count("windowSize") )
    {
      std::vector<unsigned int> sizes = opts["windowSize"].as<std::vector<unsigned int> >();
      if ( 1 <= sizes.size() )
      {
        width = sizes[0];
        if ( 2 <= sizes.size() )
        {
          height = sizes[1];
        }
      }
    }

    w.resize( width, height );
  }

  w.show();

  // Keep only once reference to the renderer in the widget. This is necessary since the OpenGL resources
  // used by the renderer must be deleted before the window gets destroyed.
  renderer.reset(); 

  int result = app.exec();

  return result;
}

int main(int argc, char *argv[])
{
  dp::sg::core::nvsgInitialize( );
#if !defined(NDEBUG)
  dp::sg::core::nvsgSetDebugFlags( dp::sg::core::NVSG_DBG_ASSERT /*| dp::sg::core::NVSG_DBG_LEAK_DETECTION*/ );
#endif

  options::options_description od("Usage: QtMinimal");

  od.add_options()
    ( "autoclipplanes", options::value<bool>()->default_value(true), "enable/disable autoclipplane")
    ( "combineVertexAttributes", "combine all vertexattribute into a single buffer" )
    ( "continuous", "enable continuous rendering" )
    ( "culling", options::value<bool>()->default_value(true), "enable/disable culling")
    ( "cullingengine", options::value<std::string>()->default_value("auto"), "auto|cpu|cuda|gl_compute")
    ( "depthPass", options::value<bool>()->default_value(false), "enable depth pass rendering" )
    ( "duration", options::value<double>()->default_value(0.0), "benchmark for a specific duration. The exit code returns the frames per second." )
    ( "effectlibrary", options::value<std::string>(), "effectlibrary to load for replacements" )
    ( "environment", options::value<std::string>(), "environment texture" )
    ( "filename", options::value<std::string>()->default_value("cubes"), "file to load" )
    ( "frames", options::value<unsigned int>()->default_value(~0), "benchmark a specific number of frames. The exit code returns the frames per second." )
    ( "fullscreen", "start in full screen mode" )
    ( "generateTangentSpace", "generate tangents and binormals" )
    ( "generateTexCoords", options::value<std::string>(), "choose texture coordinate from this list: cylindrical|planar|spherical" )
    ( "gridClone", options::value<bool>()->default_value(true), "enable/disable cloning of the node to grid" )
    ( "gridSize", options::value< std::vector<unsigned int> >()->composing()->multitoken(), "three-dimensional replication of the scene: x y z" )
    ( "gridSpacing", options::value< std::vector<float> >()->composing()->multitoken(), "three-dimensional spacing of the scene: x y z" )
    ( "headlight", "add a headlight to the camera" )
    ( "help", "show help")
    ( "maximized", "show window maximized" )
    ( "multiSample", options::value<unsigned int>()->default_value(0), "AntiAliasing with that number of color/z/stencil samples" )
    ( "multiSampleCoverage", options::value<unsigned int>()->default_value(0), "AntiAliasing with that number of coverage samples" )
    ( "optionsFile", options::value<std::string>(), "file to load (additional) options from" )
    ( "orbit", options::value<float>(), "orbit around the scene by that many degrees per frame" )
    ( "renderengine", options::value<std::string>()->default_value("Bindless"), "choose a renderengine from this list: VBO|VAB|BVAB|VBOVAO|Bindless|BindlessVAO" )
    ( "replace", options::value< std::vector<std::string> >()->composing()->multitoken(), "file to load" )
    ( "replaceAll", options::value<std::string>(), "EffectData to replace all EffectData in the scene" )
    ( "shadermanager", options::value<std::string>()->default_value("rixfx:shaderbufferload"), "rixfx:uniform|rixfx:ubo140|rixfx:ssbo140|rixfx:shaderbufferload" )
    ( "statistics", "show statistics of scene" )
    ( "stereo", "enable stereo" )
    ( "transparency", options::value<std::string>()->default_value("OITClosestList"), "choose transparency mode from list: none|OITAll|OITClosestArray|OITClosestList|SB" )
    ( "windowSize", options::value< std::vector<unsigned int> >()->composing()->multitoken(), "Window size: x y" )
    ( "zoom", options::value<float>(), "zoom in with values less than one, zoom out with values greater one" )
    ;

  int result = -1;
  try
  {
    options::variables_map opts;
    options::store( options::parse_command_line( argc, argv, od ), opts );

    if ( !opts["optionsFile"].empty() )
    {
      std::string optionsFile = opts["optionsFile"].as<std::string>();
      if ( dp::util::fileExists( optionsFile ) )
      {
         // Load the file and tokenize it
        std::ifstream ifs( optionsFile.c_str() );
        if ( ifs )
        {
          // Read the whole file into a string
          std::stringstream ss;
          ss << ifs.rdbuf();

          // Split the file content
          boost::char_separator<char> sep( " \n\r" );
          std::string ResponsefileContents( ss.str() );
          boost::tokenizer<boost::char_separator<char>> tok( ResponsefileContents, sep );
          std::vector<std::string> args;
          copy( tok.begin(), tok.end(), back_inserter( args ) );

          // Parse the file and store the options
          options::store( options::command_line_parser( args ).options( od ).run(), opts );
        }
        else
        {
          std::cout << "Could not open options file <" << optionsFile << ">!" << std::endl;
          return 1;
        }
      }
      else
      {
        std::cout << "Could not find options file <" << optionsFile << ">!" << std::endl;
      }
    }

    options::notify( opts );

    if ( !opts["help"].empty() )
    {
      std::cout << od << std::endl;
    }

    result = runApp( argc, argv, opts );
  }
  catch ( options::unknown_option e )
  {
    std::cerr << "Unknown option: " << e.get_option_name() << ". ";
    std::cout << od << std::endl;
    std::cerr << "Press enter to continue." << std::endl;
    std::string line;
    getline( std::cin, line );
  }

  dp::sg::core::nvsgTerminate();

  return result;
}
