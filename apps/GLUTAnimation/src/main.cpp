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


#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <dp/sg/core/nvsg.h>

#include <dp/sg/generator/SimpleScene.h>
#include "AnimatedScene.h"

#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/core/BufferHost.h>
#include <dp/sg/ui/ViewState.h>

#include <dp/sg/renderer/rix/gl/SceneRenderer.h>

#include <dp/sg/ui/glut/SceneRendererWidget.h>

#include <dp/sg/ui/manipulator/TrackballCameraManipulatorHIDSync.h>

#include <dp/sg/algorithm/CombineTraverser.h>
#include <dp/sg/algorithm/EliminateTraverser.h>
#include <dp/sg/algorithm/UnifyTraverser.h>
#include <dp/sg/algorithm/SearchTraverser.h>
#include <dp/sg/algorithm/DestrippingTraverser.h>
#include <dp/sg/algorithm/DeindexTraverser.h>
#include <dp/sg/algorithm/StatisticsTraverser.h>
#include <dp/sg/io/IO.h>

#include <dp/fx/EffectLibrary.h>

#include <dp/util/Memory.h>
#include <dp/util/FrameProfiler.h>

using namespace dp::math;
using namespace dp::sg::ui;

class GLUTAnimationWidget : public dp::sg::ui::glut::SceneRendererWidget
{
public:
  GLUTAnimationWidget( const dp::gl::RenderContextFormat &format, int gridSize );
  virtual ~GLUTAnimationWidget();

  virtual void onHIDEvent( dp::util::PropertyId propertyId );
  void screenshot();

  void paint();
  void setNumberOfFrames( int numberOfFrames ) { m_frames = numberOfFrames; }
protected:
  AnimatedSceneSharedPtr m_animatedScene;

  dp::util::Timer m_animationTimer;
  dp::util::Timer m_elapsedTimer;
  unsigned int m_framesInSecond; // frames rendered within one second
  int m_frames; // frames to render
  std::unique_ptr<dp::sg::ui::manipulator::TrackballCameraManipulatorHIDSync> m_trackballHIDSync;

  bool m_animateTransforms;
  bool m_animateColors;
};

GLUTAnimationWidget::GLUTAnimationWidget( const dp::gl::RenderContextFormat &format, int gridSize )
  : m_animatedScene( AnimatedScene::create( Vec2f( gridSize * 4.0f, gridSize * 4.0f), Vec2i( gridSize, gridSize) ) )
  , m_frames(-1)
  , m_framesInSecond( ~0 )
  , m_animateTransforms( true )
  , m_animateColors( true )
  , m_trackballHIDSync(new dp::sg::ui::manipulator::TrackballCameraManipulatorHIDSync( ) )
{
  ViewStateSharedPtr viewState = ViewState::create();
  viewState->setScene( m_animatedScene->getScene() );
  setupDefaultViewState(viewState);
  viewState->getCamera()->addHeadLight( dp::sg::core::createStandardPointLight() );
  setViewState(viewState);

  m_trackballHIDSync->setHID( this );
  m_trackballHIDSync->setRenderTarget( getRenderTarget() );
  setManipulator( m_trackballHIDSync.get() );
  m_animationTimer.start();
}

GLUTAnimationWidget::~GLUTAnimationWidget()
{
  // Delete SceneRenderer here to cleanup resources before the OpenGL context dies
  setSceneRenderer( SceneRendererSharedPtr::null );

  // Reset Manipulator
  setManipulator( 0 );
}

void GLUTAnimationWidget::onHIDEvent( dp::util::PropertyId propertyId )
{
  SceneRendererWidget::onHIDEvent( propertyId );

  if ( propertyId == PID_Key_S && getValue<bool>(PID_Key_S) )
  {
    screenshot();
  }

  if ( propertyId == PID_Key_D && getValue<bool>(PID_Key_D) )
  {
    dp::sg::io::saveScene( "e:\\tmp\\qtminimalxbar.nbf", getViewState() );
  }

  if ( propertyId == PID_Key_T && getValue<bool>(PID_Key_T) )
  {
    m_animateTransforms = !m_animateTransforms;
  }

  if ( propertyId == PID_Key_C && getValue<bool>(PID_Key_C) )
  {
    m_animateColors = !m_animateColors;
  }
}

void GLUTAnimationWidget::screenshot()
{
  std::string filename = getRenderTarget()->isStereoEnabled() ? "stereo.pns" : "mono.png";
  dp::util::imageToFile( getRenderTarget()->getImage(), filename );
}

void GLUTAnimationWidget::paint()
{
  dp::util::FrameProfiler::instance().beginFrame();
  if ( m_animateColors )
  {
    m_animatedScene->update( dp::util::checked_cast<float>(m_animationTimer.getTime()) );
  }
  if ( m_animateTransforms )
  {
    m_animatedScene->updateTransforms( dp::util::checked_cast<float>(m_animationTimer.getTime()) );
  }
  dp::sg::ui::glut::SceneRendererWidget::paint();

  // fps counter
  if ( m_framesInSecond == ~0 )
  {
    m_elapsedTimer.start();
    m_framesInSecond = 0;
  }
  if ( m_elapsedTimer.getTime() > 1.0)
  {
    double fps = double(m_framesInSecond) / m_elapsedTimer.getTime();
    std::ostringstream windowTitle;
    windowTitle.precision(2);
    windowTitle.setf( std::ios::fixed, std::ios::floatfield );
    windowTitle<< "AnimationTest: " << fps << " FPS";
    setWindowTitle(windowTitle.str().c_str());
    m_framesInSecond = 0;
    m_elapsedTimer.restart();
  }
  ++m_framesInSecond;

  if( m_frames > 0 )
  {
    --m_frames;
  }

  if ( m_frames == 0 )
  {
    exit(0);
  }

  dp::util::FrameProfiler::instance().endFrame();
}

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

int runApp( int argc, char *argv[], bool stereo, bool continuous, int frames, const char *renderEngine, dp::fx::Manager smt )
{
  // Create rendering engine
  SceneRendererSharedPtr renderer = dp::sg::renderer::rix::gl::SceneRenderer::create( renderEngine, smt, dp::culling::MODE_CPU );

  // Setup default OpenGL format descriptor
  // We need to create a default format first to be able to check if a stereo pixelformat is available later.
  // (An unfortunate RenderContextFormat.isAvailable() interface due to Linux.)
  dp::gl::RenderContextFormat format;

#if !defined(NDEBUG)
  int gridSize = 10;
#else
  int gridSize = 80;
#endif
  // create a widget which shows the scene
  GLUTAnimationWidget w( format, gridSize );

  w.setNumberOfFrames( frames );
  w.setSceneRenderer( renderer );
  w.setContinuousUpdate( continuous );
  w.setWindowSize( 640, 480 );

  // Keep only once reference to the renderer in the widget. This is necessary since the OpenGL resources
  // used by the renderer must be deleted before the window gets destroyed.
  renderer.reset(); 

  glutMainLoop();

  return 0;
}

int main(int argc, char *argv[])
{
  dp::sg::core::nvsgInitialize( );
#if !defined(NDEBUG)
  dp::sg::core::nvsgSetDebugFlags( dp::sg::core::NVSG_DBG_ASSERT | dp::sg::core::NVSG_DBG_LEAK_DETECTION );
#endif

  glutInit( &argc, argv );
  glutSetOption( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION );

  std::cout << "Usage: QtMinimalXBAR [--stereo] [--frames #numFrames] [--renderengine (VBO|VAB|VBOVAO|Bindless|BindlessVAO|DisplayList|Indirect) --shadermanager (rixfx:uniform|rixfx:ubo140|rixfx:ssbo140|rixfx:shaderbufferload)" << std::endl;
  std::cout << "During execution hit 's' for screenshot, 'x' to toggle stereo, 'c' to toggle color animation and 't' to toggle transform animation" << std::endl;
  std::cout << "Stereo screenshots will be saved as side/side png with filename 'stereo.pns'." << std::endl;
  std::cout << "They can be viewed with the 3D Vision Photo Viewer." << std::endl;

  dp::util::FrameProfiler::instance().setEnabled(true);

  bool stereo = false;
  bool continuous = true;
  dp::fx::Manager smt = dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX;
  int frames = -1;
  const char *renderEngine = "Bindless";

  for (int arg = 0;arg < argc;++arg)
  {
    if ( strcmp( "--stereo", argv[arg] ) == 0 )
    {
      stereo = true;
    }

    if ( strcmp( "--frames", argv[arg] ) == 0)
    {
      ++arg;
      if ( arg < argc )
      {
        frames = atoi(argv[arg]);
      }
    }
    if ( strcmp( "--renderengine", argv[arg]) == 0 )
    {
      ++arg;
      if ( arg < argc )
      {
        renderEngine = argv[arg];
      }
    }
    if ( strcmp("--shadermanager", argv[arg]) == 0)
    {
      ++arg;
      std::map<std::string, dp::fx::Manager> shaderManager;
      shaderManager["rix:ubo140"] = dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX;
      shaderManager["rixfx:uniform"] = dp::fx::MANAGER_UNIFORM;
      shaderManager["rixfx:shaderbufferload"] = dp::fx::MANAGER_SHADERBUFFER;
      shaderManager["rixfx:ubo140"] = dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX_FX;
      shaderManager["rixfx:ssbo140"] = dp::fx::MANAGER_SHADER_STORAGE_BUFFER_OBJECT;
      if ( shaderManager.find(argv[arg]) != shaderManager.end() )
      {
        smt = shaderManager[argv[arg]];
      }
    }
  }

  int result = runApp( argc, argv, stereo, continuous, frames, renderEngine, smt );

  //dp::sg::core::nvsgRuntimeDebugControl( dp::sg::core::NVSG_RUNTIME_DEBUG_DUMP_ALLOCATION_DIFFERENCES );

  dp::sg::core::nvsgTerminate();

  return result;
}
