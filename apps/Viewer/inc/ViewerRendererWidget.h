// Copyright NVIDIA Corporation 2009-2010
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


#pragma once

#include <QMenu>
#include <dp/sg/algorithm/RayIntersectTraverser.h>
#include <dp/util/Timer.h>
#include <dp/math/Vecnt.h>
#include <dp/sg/ui/RendererOptions.h>
#include <dp/sg/ui/qt5/SceniXQGLSceneRendererWidget.h>
#include "SceneRendererPipeline.h"


// Qt Uses Bool which is defined by X11. Undef it here.
#if defined(Bool)
#undef Bool
#endif

class SceniXSceneItemObject;
class QAction;
class CylindricalCameraManipulatorHIDSync;


class ViewerRendererWidget : public dp::sg::ui::qt5::SceniXQGLSceneRendererWidget
{
  Q_OBJECT

  Q_PROPERTY( RendererType    rendererType    READ getRendererType     WRITE setRendererType );
  Q_PROPERTY( ManipulatorType manipulatorType READ getManipulatorType  WRITE setManipulatorType );

public:

  // Renderer type.
  enum RendererType
  {
    RENDERER_NONE,                    // default
    RENDERER_RASTERIZE_GL2,           // OpenGL renderer version 2
    RENDERER_RASTERIZE_XBAR,          // rasterize with new xbar gl pipeline
  };

  enum ManipulatorType
  { 
    MANIPULATOR_NONE        = ~0,
    MANIPULATOR_TRACKBALL   = 0,
    MANIPULATOR_CYLINDRICAL = 1,
    MANIPULATOR_FLY         = 2,
    MANIPULATOR_WALK        = 3,
    MANIPULATOR_COUNT       = 4
  };

  ViewerRendererWidget( QWidget *parent = 0, SceniXQGLWidget *shareWidget = 0 );
  virtual ~ViewerRendererWidget();

  virtual void setScene( dp::sg::core::SceneSharedPtr const & scene );
  virtual void setCamera( const dp::sg::core::CameraSharedPtr & iter );
  virtual void setRendererType( RendererType type );
  virtual RendererType getRendererType() const;
  virtual void resizeGL( int width, int height );  // Required to restart accumulations when resizing.
  virtual void paintGL();

  void checkViewportFormatSRGB( QAction * action );
  void checkViewportFormat30Bit( QAction * action );
  void checkViewportFormatAntialiasing( QMenu * menu );
  void checkViewportFormatStencil( QAction * action );
  void checkViewportFormatStereo( QAction * action );

  dp::gl::RenderContextFormat getFormat() const;    // override of SceniXQGLWidget::getFormat() !!

  dp::sg::ui::SceneRendererSharedPtr getSceneRenderer() const;
  void setSceneRenderer( const dp::sg::ui::SceneRendererSharedPtr & );

  void setIsViewport( bool isViewport );

  static void clear();

  bool screenShot( const QString & file );

  bool isOptionSupported( const std::string & option );
  template <typename T> T getOptionValue( const std::string & option );
  template <typename T> void setOptionValue( const std::string & option, T value );

  unsigned int getFrameCount() const;

  bool isCullingEnabled() const;
  void setCullingEnabled( bool enabled );

  dp::sg::renderer::rix::gl::TransparencyMode getTransparencyMode() const;
  void setTransparencyMode( dp::sg::renderer::rix::gl::TransparencyMode mode );

  unsigned int getOITDepth() const;
  void setOITDepth( unsigned int depth );

  void setTonemapperEnabled( bool enabled );
  bool isTonemapperEnabled() const;
  TonemapperValues getTonemapperValues() const;
  void setTonemapperValues( const TonemapperValues& tonemapperValues );

signals:
  void objectSelected( dp::sg::core::PathSharedPtr );
  void present( ViewerRendererWidget * );

protected:
  virtual void initializeGL();
  void displayFPS();

  virtual void contextMenuEvent( QContextMenuEvent * event );
  bool menuConstraintsSatisfied( QObject * object );
  QAction * findAction( const QString & name );
  bool selectObject( QMouseEvent * mouseEvent );
  bool intersectObject( const dp::sg::core::NodeSharedPtr & baseSearch, unsigned int screenX, unsigned int screenY,
                        dp::sg::algorithm::Intersection & result );

  void highlightObject( const dp::sg::core::NodeSharedPtr & which );
  void highlightGeoNode( const dp::sg::core::GeoNodeSharedPtr& );
  void dehighlightAll();

  /** HID input **/
  virtual void keyPressEvent ( QKeyEvent * keyEvent );
  virtual void keyReleaseEvent ( QKeyEvent * keyEvent );
  virtual void mouseDoubleClickEvent ( QMouseEvent * mouseEvent );
  virtual void mouseMoveEvent ( QMouseEvent * mouseEvent );
  virtual void mousePressEvent ( QMouseEvent * mouseEvent );
  virtual void mouseReleaseEvent ( QMouseEvent * mouseEvent );
  virtual void wheelEvent ( QWheelEvent * wheelEvent );

  // D&D interface
  virtual void dropEvent( QDropEvent * event );
  virtual void dragEnterEvent( QDragEnterEvent * event );
  virtual void dragLeaveEvent( QDragLeaveEvent * event );
  virtual void dragMoveEvent( QDragMoveEvent * event );

  void addDefaultActions();
  void addRasterizeActions();
  void addRaytraceActions();
  void clearContextMenu();

  double determineDurationFactor();

  void enableHighlighting( bool onOff );
  void onRenderTargetChanged( const dp::gl::RenderTargetSharedPtr &oldTarget, const dp::gl::RenderTargetSharedPtr &newTarget );

public slots:
  void addCamera();
  void currentItemChanged( dp::sg::core::ObjectSharedPtr current, dp::sg::core::ObjectSharedPtr previous );
  void restartUpdate();
  void present();
  // must fully qualify ManipulatorType here, or moc gets confused...
  void setManipulatorType( ViewerRendererWidget::ManipulatorType type );
  ViewerRendererWidget::ManipulatorType getManipulatorType() const;
  void triggeredAddHeadlightMenu( QAction * action );
  void triggeredAddLightSourceMenu( QAction * action );
  void triggeredAntialiasingMenu( QAction * action );
  void triggeredCullingMenu( QAction * action );
  void triggeredViewportFormatSRGB( bool checked );
  void triggeredViewportFormat30Bit( bool checked );
  void triggeredRenderEngineMenu( QAction * action );
  void triggeredViewportFormatStencil( bool checked );
  void triggeredViewportFormatStereo( bool checked );

  void updateEnvironment(); // gets called on an environmentChanged signal and propagates 

protected slots:
  void moveSelectedObject();
  void aboutToShowCullingMenu();
  void aboutToShowRenderEngineMenu();
  void aboutToShowViewportFormatMenu();
  void aboutToShowAntialiasingMenu();

private:
  bool actionCheck( QAction * action );
  void addDirectedLight();
  void addPointLight();
  void addSpotLight();

protected:
  dp::sg::ui::ViewStateSharedPtr m_defaultViewState;
  RendererType m_rendererType;
  bool m_canManipulateObject;
  dp::sg::ui::Manipulator * m_currentManipulator;
  ManipulatorType m_manipulatorType;

  SceneRendererPipelineSharedPtr m_sceneRendererPipeline;
  
  std::set< dp::sg::core::GeoNodeSharedPtr > m_selectedGeoNodes;

  bool m_restartAccumulation;

  dp::sg::core::ObjectSharedPtr m_highlightedObject;

  std::vector< QObject * > m_contextMenuEntries;
  bool m_isViewport;  // Used to distinguish main viewports from material preview.

  dp::sg::core::SceneSharedPtr m_scene;

  bool m_sRGBDesired;   // needed to work around a driver problem, always reporting support of sRGB

  std::vector<QAction *>  m_renderEngineActions;

  static dp::gl::RenderContextFormat s_glFormat;
  static unsigned int s_cameraCount;
  static unsigned int s_dlightCount;
  static unsigned int s_slightCount;
  static unsigned int s_plightCount;

private:
  unsigned int  m_frameCount;
  unsigned int  m_oitDepth;
};


template <typename T>
inline T ViewerRendererWidget::getOptionValue( const std::string & option )
{
  DP_ASSERT( getViewState() && getViewState()->getRendererOptions() && getViewState()->getRendererOptions()->hasProperty( option ) );
  return( getViewState()->getRendererOptions()->getProperty( option ) );
}

template <typename T>
inline void ViewerRendererWidget::setOptionValue( const std::string & option, T value )
{
  DP_ASSERT( getViewState() && getViewState()->getRendererOptions() );
  dp::sg::ui::RendererOptionsSharedPtr const& options = getViewState()->getRendererOptions();
  DP_ASSERT( options->hasProperty( option ) );
  options->setValue<T>( options->getProperty( option ), value );
}

inline unsigned int ViewerRendererWidget::getFrameCount() const
{
  return( m_frameCount );
}

inline bool ViewerRendererWidget::isCullingEnabled() const
{
  DP_ASSERT( m_sceneRendererPipeline );
  return( m_sceneRendererPipeline->isCullingEnabled() );
}

inline void ViewerRendererWidget::setCullingEnabled( bool enabled )
{
  DP_ASSERT( m_sceneRendererPipeline );
  m_sceneRendererPipeline->setCullingEnabled( enabled );
}
