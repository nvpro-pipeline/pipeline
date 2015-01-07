// Copyright NVIDIA Corporation 2014
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

#include <dp/sg/core/nvsg.h>
#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/TextureFile.h>

#include <dp/sg/io/IO.h>

#include <dp/sg/renderer/rix/gl/SceneRenderer.h>

#include <dp/sg/ui/SceniXWidget.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/sg/ui/glut/SceneRendererWidget.h>
#include <dp/sg/ui/manipulator/TrackballCameraManipulatorHIDSync.h>

// scenes
#include <dp/sg/generator/SimpleScene.h>

#include <dp/fx/EffectLibrary.h>
#include <dp/util/File.h>

#include <fstream>

// Derive a glut SceneRendererWidget.
class Tutorial : public dp::sg::ui::glut::SceneRendererWidget
{
public:
  Tutorial();
  ~Tutorial();

  // HID events are keyboard and mouse events.
  virtual void onHIDEvent( dp::util::PropertyId propertyId );
};

Tutorial::Tutorial()
{
}

Tutorial::~Tutorial()
{
}

void Tutorial::onHIDEvent( dp::util::PropertyId propertyId )
{
  SceneRendererWidget::onHIDEvent(propertyId);

  // Escape key status has been changed
  if (propertyId == PID_Key_Escape)
  {
    // Escape key is being pressed, exit application
    if (getValue<bool>( propertyId ))
    {
      glutLeaveMainLoop();
    }
  }
}

int runApp()
{
  // create a glut-based widget for rendering
  Tutorial w;

  // create a rix::gl based SceneRenderer using the VBO vertex technique and uniform buffer ojects for the parameters
  dp::sg::renderer::rix::gl::SceneRendererSharedPtr renderer = dp::sg::renderer::rix::gl::SceneRenderer::create("VBO", dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX);

  // Use the SceneRenderer as rendering algorithm for the widget
  w.setSceneRenderer( renderer );

  // Generate a simple scene with 4 cubes
  dp::sg::generator::SimpleScene simpleScene;

  // create a ViewState objects which combines the scene and the camera.
  dp::sg::ui::ViewStateSharedPtr viewState = dp::sg::ui::ViewState::create();

  // attach the simple scene to the ViewState
  viewState->setScene(simpleScene.m_sceneHandle);
  
  // determine the near/far plane of the camera automatically based on the scenes bounding box.
  viewState->setAutoClipPlanes(true);

  // create a default camera which looks at the scene.
  dp::sg::ui::setupDefaultViewState( viewState );

  // The simple scene does not have a light by default. Create a PointLight attached to the camera.
  viewState->getCamera()->addHeadLight( dp::sg::core::createStandardPointLight() );

  // Attach the Viewstate to the Widget
  w.setViewState( viewState );

  // 1024x576 seems to be a nice window size
  w.setWindowSize(1024, 576);

  // Keep only single reference to the renderer in the widget. The SceneRenderer keeps the OpenGL resources
  // required to render the Scene alive. If the the window closes the OpenGL context used by the SceneRenderer
  // will be destroyed which results in a failure when deleting the SceneRenderer and thus its resources
  // afterwards.
  renderer.reset(); 

  glutMainLoop();

  return 0;
}

int main(int argc, char *argv[])
{
  int result = -1;
  try
  {
    // initialize the pipeline
    dp::sg::core::nvsgInitialize( );

  #if !defined(NDEBUG)
    // enable assertions in debug case.
    dp::sg::core::nvsgSetDebugFlags( dp::sg::core::NVSG_DBG_ASSERT);
  #endif

    // initialize GLUT
    glutInit( &argc, argv );
    glutSetOption( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION );

    //  It is recommended to put the 'application' logic into a function to ensure that all objects are being
    // destroyed before shutting down.
    result = runApp();

    // dp::gl keeps a reference to the dp::gl::RenderContext object and glut bypasses the destruction
    // Tell dp::gl that there's no context alive anymore here.
    if (dp::gl::RenderContext::getCurrentRenderContext()) {
      dp::gl::RenderContext::getCurrentRenderContext()->makeNoncurrent();
    }

    // shut down the pipeline
    dp::sg::core::nvsgTerminate();
  }
  catch (...)
  {
    std::cerr << "caught exception" << std::endl;
  }

  return result;
}
