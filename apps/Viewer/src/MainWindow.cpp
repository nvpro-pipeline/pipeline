// Copyright NVIDIA Corporation 2009-2015
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


#include <QComboBox>
#include <QStandardPaths>
#include <QFileDialog>
#include <QMessageBox>
#include <QMenuBar>
#include <QSettings>
#include <QStatusBar>
#include <QToolBar>

#include "PlainTextDialog.h"
#include "MainWindow.h"
#include "NormalizeDialog.h"
#include "NormalsDialog.h"
#include "OptimizerDialog.h"
#include "PreferencesDialog.h"
#include "SmoothDialog.h"
#include "StereoDialog.h"
#include "TonemapperDialog.h"
#include "TransparencyDialog.h"
#include "Viewer.h"

#include <dp/sg/algorithm/AnalyzeTraverser.h>
#include <dp/sg/algorithm/DestrippingTraverser.h>
#include <dp/sg/algorithm/Search.h>
#include <dp/sg/algorithm/SearchTraverser.h>
#include <dp/sg/algorithm/StatisticsTraverser.h>
#include <dp/sg/algorithm/StrippingTraverser.h>
#include <dp/sg/algorithm/TriangulateTraverser.h>
#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/io/IO.h>
#include <dp/sg/io/PlugInterfaceID.h>
#include <dp/util/PlugIn.h>

#include <iomanip>

#undef MEMDEBUG

#if defined(DP_OS_LINUX)
#undef KeyPress
#endif

QString analyze( dp::sg::core::SceneSharedPtr const & scene );
bool containsQuadPrimitives( std::vector<dp::sg::core::ObjectWeakPtr> const & vp );
bool containsStripablePrimitives( std::vector<dp::sg::core::ObjectWeakPtr> const & vp );
bool containsStrippedPrimitives( std::vector<dp::sg::core::ObjectWeakPtr> const & vp );
bool findOrCreateCameras( dp::sg::ui::ViewStateSharedPtr const& viewState, std::vector<dp::sg::core::CameraSharedPtr> & cameras );
void nameCamera( const dp::sg::core::CameraSharedPtr & camera, const std::string & baseName, unsigned int & index );
std::string samplerTypeToString( unsigned int samplerType );
void setTraversalMasks( dp::sg::core::SceneSharedPtr const & scene, unsigned int mask );
bool textureTypeIsCompatible( unsigned int id, dp::sg::core::TextureTarget tt );
std::string textureTargetToString( dp::sg::core::TextureTarget tt );

MainWindow::MainWindow()
: QMainWindow()
, m_continuousRedraw(false)
, m_continuousRedrawTimerID(0)
, m_currentViewport(~0)
, m_fpsTimerID(0)
, m_lastTime(0.0)
, m_lastFrame(0)
, m_log(nullptr)
, m_materialBrowser(nullptr)
, m_normalsColor(Qt::white)
, m_normalsDisplayed(false)
, m_sceneProperties(nullptr)
, m_sceneTreeBrowser(nullptr)
, m_scriptSandbox(nullptr)
, m_undo(nullptr)
{
  setWindowTitle( VIEWER_APPLICATION_NAME );
  resize( 768, 512 );

  setupActions();
  setupToolbar();
  setupDockWidgets();
  setupMenus();
  setupStatusBar();

  restoreSettings();

  // connect scene changed signal
  connect( GetApp(), SIGNAL(sceneChanged()), this, SLOT(sceneChanged()) );

  // gather all scene loaders
  std::stringstream sceneStream;
  sceneStream << "All Scenes ( ";
  const dp::util::UPITID PITID_SCENE_LOADER(UPITID_SCENE_LOADER, UPITID_VERSION);
  std::vector<dp::util::UPIID> piids;
  if ( queryInterfaceType(std::vector<std::string>(), PITID_SCENE_LOADER, piids) )
  {
    for ( std::vector<dp::util::UPIID>::iterator it = piids.begin() ; it != piids.end() ; ++it )
    {
      sceneStream << "*" << it->getPlugSpecificIDString() << " ";
    }
  }
  sceneStream << " )";
  m_sceneLoaderFilter = sceneStream.str();

  // gather all texture loaders
  std::stringstream textureStream;
  textureStream << "All Textures ( ";
  const dp::util::UPITID PITID_TEXTURE_LOADER(UPITID_TEXTURE_LOADER, UPITID_VERSION);
  piids.clear();
  if ( queryInterfaceType(std::vector<std::string>(), PITID_TEXTURE_LOADER, piids) )
  {
    for ( std::vector<dp::util::UPIID>::iterator it = piids.begin() ; it != piids.end() ; ++it )
    {
      textureStream << "*" << it->getPlugSpecificIDString() << " ";
    }
  }
  textureStream << " )";
  m_textureLoaderFilter = textureStream.str();
}

MainWindow::~MainWindow()
{
  killTimer( m_fpsTimerID );
  m_timer.stop();
}

void MainWindow::aboutToShowConvertSceneMenu()
{
  dp::sg::core::SceneSharedPtr scene = GetApp()->getScene();
  DP_ASSERT( scene );

  dp::sg::algorithm::SearchTraverser searchTraverser;
  searchTraverser.setClassName( "class dp::sg::core::Primitive" );
  searchTraverser.setBaseClassSearch( true );
  searchTraverser.apply( scene );

  const std::vector<dp::sg::core::ObjectWeakPtr> &vp = searchTraverser.getResults();

  m_destripSceneAction->setEnabled( containsStrippedPrimitives( vp ) );
  m_stripSceneAction->setEnabled( containsStripablePrimitives( vp ) );
  m_triangulateSceneAction->setEnabled( containsQuadPrimitives( vp ) );
}

void MainWindow::aboutToShowEditMenu()
{
  bool enabled = !!GetApp()->getScene();
  m_optimizeSceneAction->setEnabled( enabled );
  m_menus[MID_CONVERT_SCENE]->setEnabled( enabled );
  m_menus[MID_MODIFY_SCENE]->setEnabled( enabled );

  m_menus[MID_ADD_HEADLIGHT]->setEnabled( m_currentViewport != ~0 );
  m_menus[MID_ADD_LIGHT_SOURCE]->setEnabled( m_currentViewport != ~0 );
}

void MainWindow::aboutToShowFileMenu()
{
  // if there is a scene, enable the close and save, otherwise disable
  bool enabled = !!GetApp()->getScene();
  m_saveAction->setEnabled( enabled );
  m_closeAction->setEnabled( enabled );
  m_analyzeSceneAction->setEnabled( enabled );
  m_sceneStatisticsAction->setEnabled( enabled );
  m_clearUndoStackAction->setEnabled( !m_undo->empty() );
}

void MainWindow::aboutToShowViewMenu()
{
  bool viewing = ( m_currentViewport != ~0 );

  m_menus[MID_RENDER_ENGINE]->setEnabled( viewing );
  m_menus[MID_VIEWPORT_FORMAT]->setEnabled( viewing );
  m_menus[MID_CULLING]->setEnabled( viewing );

  if ( viewing )
  {
    // Enable/disable Stereo ... menu. This will also add a checkmark if stereo is enabled.
    m_renderWidgets[m_currentViewport]->checkViewportFormatStereo( m_stereoDialogAction );

    dp::gl::RenderContextFormat format = m_renderWidgets[m_currentViewport]->getFormat();
    m_tonemapperDialogAction->setEnabled( format.getMultisample() < 1 );
  }
  else
  {
    m_stereoDialogAction->setEnabled( false );
    m_tonemapperDialogAction->setEnabled( false );
  }

  bool enable = viewing && m_renderWidgets[m_currentViewport]->getSceneRenderer().isPtrTo<dp::sg::renderer::rix::gl::SceneRenderer>();
  m_depthPassAction->setEnabled( enable );
  m_transparencyDialogAction->setEnabled( enable );
  if ( enable )
  {
    m_depthPassAction->setChecked( m_renderWidgets[m_currentViewport]->getSceneRenderer().staticCast<dp::sg::renderer::rix::gl::SceneRenderer>()->getDepthPass() );
  }
}

void MainWindow::activeViewportChanged( int index, QWidget * widget )
{
  if ( index != m_currentViewport )
  {
    if ( m_currentViewport != ~0 )
    {
      ViewerRendererWidget * cvp = m_renderWidgets[m_currentViewport];
      disconnect( m_menus[MID_ADD_HEADLIGHT], SIGNAL(triggered(QAction*)), cvp, SLOT(triggeredAddHeadlightMenu(QAction*) ) );
      disconnect( m_menus[MID_ADD_LIGHT_SOURCE], SIGNAL(triggered(QAction*)), cvp, SLOT(triggeredAddLightSourceMenu(QAction*) ) );
      disconnect( m_menus[MID_ANTIALIASING], SIGNAL(aboutToShow()), cvp, SLOT(aboutToShowAntialiasingMenu()) );
      disconnect( m_menus[MID_ANTIALIASING], SIGNAL(triggered(QAction*)), cvp, SLOT(triggeredAntialiasingMenu(QAction*)) );
      disconnect( m_menus[MID_CULLING], SIGNAL(aboutToShow()), cvp, SLOT(aboutToShowCullingMenu()) );
      disconnect( m_menus[MID_CULLING], SIGNAL(triggered(QAction*)), cvp, SLOT(triggeredCullingMenu(QAction*) ) );
      disconnect( this, SIGNAL(manipulatorChanged(ViewerRendererWidget::ManipulatorType)), 
                  cvp,  SLOT(setManipulatorType(ViewerRendererWidget::ManipulatorType)) );
      disconnect( m_menus[MID_RENDER_ENGINE], SIGNAL(aboutToShow()), cvp, SLOT(aboutToShowRenderEngineMenu()) );
      disconnect( m_menus[MID_RENDER_ENGINE], SIGNAL(triggered(QAction*)), cvp, SLOT(triggeredRenderEngineMenu(QAction*)) );
      disconnect( m_menus[MID_VIEWPORT_FORMAT], SIGNAL(aboutToShow()), cvp, SLOT(aboutToShowViewportFormatMenu()) );
      disconnect( m_viewportFormat30BitAction, SIGNAL(triggered(bool)), cvp, SLOT(triggeredViewportFormat30Bit(bool)) );
      disconnect( m_viewportFormatSRGBAction, SIGNAL(triggered(bool)), cvp, SLOT(triggeredViewportFormatSRGB(bool)) );
      disconnect( m_viewportFormatStencilAction, SIGNAL(triggered(bool)), cvp, SLOT(triggeredViewportFormatStencil(bool)) );
      disconnect( m_viewportFormatStereoAction, SIGNAL(triggered(bool)), cvp, SLOT(triggeredViewportFormatStereo(bool)) );
    }

    // reset active viewport
    m_currentViewport = index;

    // set these before we connect the signals..
    CameraAnimator * ca = static_cast< CameraAnimator * >( getCurrentCameraAnimator() );
    m_cameraIterationAction->setChecked( ca->isCameraCycle() );
    m_orbitXAction->setChecked( ca->isCameraOrbitX() );
    m_orbitYAction->setChecked( ca->isCameraOrbitY() );
    m_orbitZAction->setChecked( ca->isCameraOrbitZ() );

    // clear these, and then set the appropriate one
    for( unsigned int i = 0; i < ViewerRendererWidget::MANIPULATOR_COUNT; i ++ )
    {
      m_manipulatorAction[i]->setChecked( false );
    }

    ViewerRendererWidget * cvp = m_renderWidgets[m_currentViewport];
    DP_ASSERT( (int)(cvp->getManipulatorType()) < (int)(ViewerRendererWidget::MANIPULATOR_COUNT) );
    // our buttons are layed out in the same order as the enum in ViewerRendererWidget
    m_manipulatorAction[cvp->getManipulatorType()]->setChecked( true );

    initMultisampleMenu( m_menus[MID_ANTIALIASING] );
    connect( m_menus[MID_ADD_HEADLIGHT], SIGNAL(triggered(QAction*)), cvp, SLOT(triggeredAddHeadlightMenu(QAction*) ) );
    connect( m_menus[MID_ADD_LIGHT_SOURCE], SIGNAL(triggered(QAction*)), cvp, SLOT(triggeredAddLightSourceMenu(QAction*) ) );
    connect( m_menus[MID_ANTIALIASING], SIGNAL(aboutToShow()), cvp, SLOT(aboutToShowAntialiasingMenu()) );
    connect( m_menus[MID_ANTIALIASING], SIGNAL(triggered(QAction*)), cvp, SLOT(triggeredAntialiasingMenu(QAction*)) );
    connect( m_menus[MID_CULLING], SIGNAL(aboutToShow()), cvp, SLOT(aboutToShowCullingMenu()) );
    connect( m_menus[MID_CULLING], SIGNAL(triggered(QAction*)), cvp, SLOT(triggeredCullingMenu(QAction*) ) );
    connect( this, SIGNAL(manipulatorChanged(ViewerRendererWidget::ManipulatorType)), 
             cvp,  SLOT(setManipulatorType(ViewerRendererWidget::ManipulatorType)) );
    connect( m_menus[MID_RENDER_ENGINE], SIGNAL(aboutToShow()), cvp, SLOT(aboutToShowRenderEngineMenu()) );
    connect( m_menus[MID_RENDER_ENGINE], SIGNAL(triggered(QAction*)), cvp, SLOT(triggeredRenderEngineMenu(QAction*)) );
    connect( m_menus[MID_VIEWPORT_FORMAT], SIGNAL(aboutToShow()), cvp, SLOT(aboutToShowViewportFormatMenu()) );
    connect( m_viewportFormat30BitAction, SIGNAL(triggered(bool)), cvp, SLOT(triggeredViewportFormat30Bit(bool)) );
    connect( m_viewportFormatSRGBAction, SIGNAL(triggered(bool)), cvp, SLOT(triggeredViewportFormatSRGB(bool)) );
    connect( m_viewportFormatStencilAction, SIGNAL(triggered(bool)), cvp, SLOT(triggeredViewportFormatStencil(bool)) );
    connect( m_viewportFormatStereoAction, SIGNAL(triggered(bool)), cvp, SLOT(triggeredViewportFormatStereo(bool)) );
  }
}

void MainWindow::closeEvent( QCloseEvent * event )
{
  saveSettings();
  QMainWindow::closeEvent( event );
}

CameraAnimator * MainWindow::createAnimator( ViewerRendererWidget * vrw )
{
  CameraAnimator * animator = new CameraAnimator( vrw );
  animator->setViewState( vrw->getViewState() );

  // the update signal from the camera animator
  connect( animator, SIGNAL(update()), vrw, SLOT(restartUpdate()) );

  return animator;
}

ViewerRendererWidget * MainWindow::createRenderer( QWidget * parent, dp::sg::core::SceneSharedPtr const & scene )
{
  // add the renderer - use sharewidget, add menu, not standalone
  ViewerRendererWidget *vrw = new ViewerRendererWidget( parent, GetApp()->getGlobalShareGLWidget() );
   
  vrw->setRendererType( ViewerRendererWidget::RENDERER_RASTERIZE_XBAR );

  vrw->setScene( scene );

  //
  // configure signals
  //
  
  // sent from the app
  connect(GetApp(), SIGNAL(continuousUpdate()), vrw, SLOT(update()));         // Continuous redraw just needs to rerender. 
  connect(GetApp(), SIGNAL(viewChanged()),      vrw, SLOT(restartUpdate()));  // These need to additionally restart accumulation.
  connect(GetApp(), SIGNAL(materialChanged()),  vrw, SLOT(restartUpdate()));
  connect(GetApp(), SIGNAL(sceneTreeChanged()), vrw, SLOT(restartUpdate()));

  // sent from other widgets
  connect( m_sceneTreeBrowser, SIGNAL(currentItemChanged(dp::sg::core::ObjectSharedPtr,dp::sg::core::ObjectSharedPtr))
         , vrw, SLOT(currentItemChanged(dp::sg::core::ObjectSharedPtr,dp::sg::core::ObjectSharedPtr)) );

  // sent from the vrw
  connect( vrw, SIGNAL(objectSelected(dp::sg::core::PathSharedPtr)), this, SLOT(selectObject(dp::sg::core::PathSharedPtr)) );
  connect( vrw, SIGNAL(objectSelected(dp::sg::core::PathSharedPtr)), m_sceneTreeBrowser, SLOT(selectObject(dp::sg::core::PathSharedPtr)) );

  // event filter will filter the keys for camera cycling
  vrw->installEventFilter( this );

  return vrw;
}

//
// For now, this method is used to notify the MainWindow 1) that a keypress has occurred and 2) to filter the 0-9 keys
// to move to appropriate cameras.  This may not be the best way, but we do it for now.
//
bool MainWindow::eventFilter( QObject * obj, QEvent * event )
{
  bool found = false;
  for ( size_t i = 0; i < m_renderWidgets.size(); i ++ )
  {
    if ( obj == m_renderWidgets[i] )
    {
      found = true;
      if ( event->type() == QEvent::KeyPress )
      {
        // stop orbiting and such on a keypress event for active view
        m_cameraAnimators[i]->cancel();

        // cancel all of these on a keypress, for the active viewport
        m_cameraIterationAction->setChecked( false );
        m_zoomAllAction->setChecked( false );
        m_orbitXAction->setChecked( false );
        m_orbitYAction->setChecked( false );
        m_orbitZAction->setChecked( false );

        // now, check to see if it was one of the number keys
        QKeyEvent * keyEvent = static_cast< QKeyEvent * >( event );

        if( keyEvent->key() >= '0' && keyEvent->key() <= '9' )
        {
          // could connect as a slot/signal as well
          m_cameraAnimators[i]->moveToCameraIndex( (keyEvent->key() - '0') );
          // if it was a key event, then we return true here, so the key is not processed by
          // the window
          return true;
        }
      }

      break;
    }
  }

  return QMainWindow::eventFilter( obj, event );
}

QAction * MainWindow::getContinuousRedrawAction() const
{
  return( m_continuousRedrawAction );
}

QObject * MainWindow::getCurrentCameraAnimator() const
{
  return( ( m_currentViewport != ~0 ) ? m_cameraAnimators[m_currentViewport] : nullptr );
}

ViewerRendererWidget * MainWindow::getCurrentViewport() const
{
  return( ( m_currentViewport != ~0 ) ? m_renderWidgets[m_currentViewport] : nullptr );
}

QWidget * MainWindow::getLog() const
{
  return( m_log );
}

QMenu * MainWindow::getMenu( MenuID id ) const
{
  return( m_menus[id] );
}

QLabel * MainWindow::getStatisticsLabel() const
{
  return( m_statisticsLabel );
}

QString MainWindow::getTextureFile( unsigned int samplerType )
{
  static QFileDialog * fileDialog = nullptr;
  QString textureFile;
  std::string samplerTypeString = samplerTypeToString( samplerType );
  bool done;
  do
  {
    done = true;
    textureFile = QFileDialog::getOpenFileName( this, QString( "Select a " ) + QString( samplerTypeString.c_str() ) + QString( " File" )
                                              , GetApp()->getPreferences()->getTextureSelectionPath(), QString( m_textureLoaderFilter.c_str() ) );
    if ( ! textureFile.isEmpty() )
    {
      dp::sg::core::TextureHostSharedPtr textureHost = dp::sg::io::loadTextureHost( textureFile.toStdString() );
      DP_ASSERT( textureHost );
      dp::sg::core::TextureTarget tt = textureHost->getTextureTarget();
      if ( ! textureTypeIsCompatible( samplerType, tt ) )
      {
        std::string text = "The selected file \n\t" + textureFile.toStdString()
                          + "\nis a " + textureTargetToString( tt )
                          + "\nbut needs to be a " + samplerTypeString + "!";
        QMessageBox::warning( this, "Incompatible Texture File selected", QString( text.c_str() ) );
        done = false;
      }
    }
  } while ( !done );
  if ( ! textureFile.isEmpty() )
  {
    GetApp()->getPreferences()->setTextureSelectionPath( QFileInfo( textureFile ).absolutePath() );
  }
  return( textureFile );
}

void MainWindow::initMultisampleMenu( QMenu * menu )
{
  menu->clear();

  dp::gl::RenderContextFormat format;
  dp::gl::RenderContextFormat::FormatInfo info;
  format.getFormatInfo( info );
  for ( std::set<std::pair<unsigned int,unsigned int> >::const_iterator it = info.multisampleModesSupported.begin() ; it != info.multisampleModesSupported.end() ; ++it )
  {
    QString text;
    if ( it->first == 0 )
    {
      DP_ASSERT( it->second == 0 );
      text = "&Off";
    }
    else if ( it->first == it->second )
    {
      text.sprintf( "&%d", it->first );
    }
    else
    {
      text.sprintf( "&%d/%d", it->first, it->second );
    }
    QAction * action = menu->addAction( text );
    action->setCheckable( true );
    DP_ASSERT( ( ( it->first << 8 ) & it->second ) == 0 );
    action->setData( ( it->first << 8 ) | it->second );
  }
}

void MainWindow::loadFile( const QString & fileName, bool replaceShaders )
{
  if ( !fileName.isEmpty() )
  {
    GetApp()->setOverrideCursor( Qt::WaitCursor );

    if( !GetApp()->loadScene( fileName ) )
    {
      QMessageBox::critical( 0, "Scene Load Error", "Unable to load file: " + fileName );
    }
    else
    {
      // don't add to list if we imported it, only opened
      setCurrentFile( fileName, !replaceShaders );
    }

    GetApp()->restoreOverrideCursor();
  }
}

void MainWindow::removeNormals()
{
  m_normalsDisplayed = false;

  // The dialog's apply method has the smarts for the add/remove normals.
  // We don't actually show it, just use it to apply.
  dp::sg::core::SceneSharedPtr scene = GetApp()->getScene();
  if ( scene )
  {
    NormalsDialog dlg( this, m_normalsDisplayed, 0.f, Qt::black );
    dlg.apply( scene );
    GetApp()->emitSceneTreeChanged();
  }
}

void MainWindow::restoreSettings( const QString & window )
{
  QSettings settings( VIEWER_APPLICATION_VENDOR, VIEWER_APPLICATION_NAME );
  QString windowState;
  QString geometry;

  if ( !window.isEmpty() )
  {
    windowState = window + QString("/");
    geometry    = windowState;
  }

  windowState += "windowState";
  geometry    += "geometry";

  restoreGeometry( settings.value( geometry ).toByteArray() );
  restoreState( settings.value( windowState ).toByteArray() );
}

void MainWindow::saveSettings( const QString & window )
{
  QSettings settings( VIEWER_APPLICATION_VENDOR, VIEWER_APPLICATION_NAME );
  QString windowState;
  QString geometry;

  if ( !window.isEmpty() )
  {
    windowState = window + QString("/");
    geometry    = windowState;
  }

  windowState += "windowState";
  geometry    += "geometry";

  settings.setValue( geometry, saveGeometry() );
  settings.setValue( windowState, saveState() );
}

void MainWindow::sceneChanged()
{
  m_sceneProperties->clear();

  // reset this on file load
  m_normalsDisplayed = false;

  dp::sg::core::SceneSharedPtr scene = GetApp()->getScene();
  if ( scene )
  {
    m_sceneTreeBrowser->setScene( scene );

    // enable the buttons since we have a scene loaded
    m_continuousRedrawAction->setEnabled( true );
    m_cameraIterationAction->setEnabled( true );
    m_zoomAllAction->setEnabled( true );
    m_orbitXAction->setEnabled( true );
    m_orbitYAction->setEnabled( true );
    m_orbitZAction->setEnabled( true );
    m_normalsDialogAction->setEnabled( true );
    m_viewportCombo->setEnabled( true );
    m_saveAction->setEnabled( true );
    m_closeAction->setEnabled( true );

    for ( unsigned int i = 0; i < ViewerRendererWidget::MANIPULATOR_COUNT; i ++ )
    {
      m_manipulatorAction[i]->setEnabled( true );
    }
  }
  else
  {
    m_sceneTreeBrowser->setScene( dp::sg::core::SceneSharedPtr() );

    // disable the buttons since we have no scene loaded
    m_continuousRedrawAction->setEnabled( false );
    m_cameraIterationAction->setEnabled( false );
    m_zoomAllAction->setEnabled( false );
    m_orbitXAction->setEnabled( false );
    m_orbitYAction->setEnabled( false );
    m_orbitZAction->setEnabled( false );
    m_normalsDialogAction->setEnabled( false );
    m_viewportCombo->setEnabled( false );
    m_saveAction->setEnabled( false );
    m_closeAction->setEnabled( false );

    for ( unsigned int i = 0; i < ViewerRendererWidget::MANIPULATOR_COUNT; i ++ )
    {
      m_manipulatorAction[i]->setEnabled( false );
    }
  }

  // reset these on file load
  m_continuousRedrawAction->setChecked( false );
  m_cameraIterationAction->setChecked( false );
  m_orbitXAction->setChecked( false );
  m_orbitYAction->setChecked( false );
  m_orbitZAction->setChecked( false );

  // clear static state
  ViewerRendererWidget::clear();

  // update renderers
  updateRenderers( GetApp()->getViewState() );
}

void MainWindow::selectObject( dp::sg::core::PathSharedPtr const& path )
{
  dp::sg::core::ObjectSharedPtr object = path->getTail();
  if ( object.isPtrTo<dp::sg::core::FrustumCamera>() )
  {
    static_cast<CameraAnimator*>(getCurrentCameraAnimator())->moveToCamera( object.staticCast<dp::sg::core::FrustumCamera>().getWeakPtr() );
  }
  else if ( object.isPtrTo<dp::sg::core::LightSource>() )
  {
    static_cast<CameraAnimator*>(getCurrentCameraAnimator())->moveToLight( object.staticCast<dp::sg::core::LightSource>().getWeakPtr() );
  }
}

void MainWindow::setCurrentFile( const QString & fileName, bool addToRecent )
{
  if ( addToRecent )
  {
    QSettings settings( VIEWER_APPLICATION_VENDOR, VIEWER_APPLICATION_NAME );
    QStringList files = settings.value( "recentFileList" ).toStringList();
    files.removeAll( fileName );
    files.prepend( fileName );
    while ( MaxRecentFiles < files.size() )
    {
      files.removeLast();
    }

    settings.setValue( "recentFileList", files );

    updateRecentFileActions();
  }

  // set Title Bar name
  // NOTE: we don't attempt to translate this because it wouldn't make any sense..
  QString title = QString( VIEWER_APPLICATION_NAME ) + QString(" - ") + fileName;
  setWindowTitle( title );
}

void MainWindow::setupActions()
{
  m_openAction = new QAction( this );
  m_openAction->setIcon( QIcon( ":/images/LoadFile_off.png" ) );
  m_openAction->setText( "&Open..." );
  m_openAction->setShortcut( QKeySequence( "Ctrl+O" ) );
  connect( m_openAction, SIGNAL(triggered()), this, SLOT(triggeredOpen()) );

  m_saveAction = new QAction( this );
  m_saveAction->setIcon( QIcon( ":/images/SaveFile_off.png" ) );
  m_saveAction->setText( "&Save as..." );
  m_saveAction->setShortcut( QKeySequence( "Ctrl+S" ) );
  connect( m_saveAction, SIGNAL(triggered()), this, SLOT(triggeredSave()) );

  m_quitAction = new QAction( this );
  m_quitAction->setIcon( QIcon( ":/images/Power_off.png" ) );
  m_quitAction->setText( "&Quit" );
  m_quitAction->setShortcut( QKeySequence( "Ctrl+Q" ) );
  connect( m_quitAction, SIGNAL(triggered()), this, SLOT(triggeredQuit()) );

  m_closeAction = new QAction( this );
  m_closeAction->setText( "&Close" );
  m_closeAction->setShortcut( QKeySequence( "Ctrl+C" ) );
  connect( m_closeAction, SIGNAL(triggered()), this, SLOT(triggeredClose()) );

  // should start disabled
  m_saveAction->setEnabled( false );
  m_closeAction->setEnabled( false );

  for ( int i=0 ; i<MaxRecentFiles ; ++i )
  {
    m_recentFileAction[i] = new QAction( this );
    m_recentFileAction[i]->setVisible( false );
    connect( m_recentFileAction[i], SIGNAL(triggered()), this, SLOT(triggeredRecentFile()) );
  }

  // add the view layout combo
  m_viewportCombo = new QComboBox(this);
  m_viewportCombo->setEnabled( false );
  m_viewportCombo->addItem( QIcon( ":/images/LayoutOne_off.png" ), "" );
  m_viewportCombo->addItem( QIcon( ":/images/LayoutTwoLeft_off.png" ), "" );
  m_viewportCombo->addItem( QIcon( ":/images/LayoutTwoTop_off.png" ), "" );
  m_viewportCombo->addItem( QIcon( ":/images/LayoutThreeLeft_off.png" ), "" );
  m_viewportCombo->addItem( QIcon( ":/images/LayoutThreeTop_off.png" ), "" );
  m_viewportCombo->addItem( QIcon( ":/images/LayoutFour_off.png" ), "" );
  m_viewportCombo->setIconSize( QSize( 48, 48 ) );
  m_viewportCombo->setStyleSheet("QComboBox { border:1px outset #3c3c3c; border-radius:25px; padding:0px; }\
                                  QComboBox::drop-down { subcontrol-origin:padding; subcontrol-position:right; width:7px; }");
  // m_viewportCombo signals are connected to m_viewportLayout slots after that has been created.

  QIcon ContinuousRenderingIcon( ":/images/ContinuousRendering_off.png" );
  ContinuousRenderingIcon.addFile( ":/images/ContinuousRendering_on.png", QSize(), QIcon::Normal, QIcon::On );

  m_continuousRedrawAction = new QAction( this );
  m_continuousRedrawAction->setToolTip( "Switch Continuous Redraw on/off" );
  m_continuousRedrawAction->setIcon( ContinuousRenderingIcon );
  m_continuousRedrawAction->setEnabled( false );
  m_continuousRedrawAction->setCheckable( true );
  connect( m_continuousRedrawAction, SIGNAL(toggled(bool)), this, SLOT(toggledContinuousRedraw(bool)) );

  m_zoomAllAction = new QAction( this );
  m_zoomAllAction->setToolTip( "Zoom All" );
  m_zoomAllAction->setIcon( QIcon( ":/images/ZoomAll_off.png" ) );
  m_zoomAllAction->setEnabled( false );
  connect( m_zoomAllAction, SIGNAL(triggered()), this, SLOT(triggeredZoomAll()) );

  QIcon CameraCyclingIcon( ":/images/CameraCycling_off.png" );
  CameraCyclingIcon.addFile( ":/images/CameraCycling_on.png", QSize(), QIcon::Normal, QIcon::On );

  m_cameraIterationAction = new QAction( this );
  m_cameraIterationAction->setToolTip( "Switch Camera Cycling on/off" );
  m_cameraIterationAction->setIcon( CameraCyclingIcon );
  m_cameraIterationAction->setEnabled( false );
  m_cameraIterationAction->setCheckable( true );
  connect( m_cameraIterationAction, SIGNAL(toggled(bool)), this, SLOT(toggledCameraCycle(bool)) );

  QIcon CameraOrbitXIcon( ":/images/CameraOrbitX_off.png" );
  CameraOrbitXIcon.addFile( ":/images/CameraOrbitX_on.png", QSize(), QIcon::Normal, QIcon::On );

  m_orbitXAction = new QAction( this );
  m_orbitXAction->setToolTip( "Orbit Around X Axis" );
  m_orbitXAction->setIcon( CameraOrbitXIcon );
  m_orbitXAction->setCheckable( true );
  m_orbitXAction->setChecked( false );
  m_orbitXAction->setEnabled( false );
  connect( m_orbitXAction, SIGNAL(toggled(bool)), this, SLOT(toggledCameraOrbitX(bool)) );

  QIcon iconCameraOrbitY( ":/images/CameraOrbitY_off.png" );
  iconCameraOrbitY.addFile( ":/images/CameraOrbitY_on.png", QSize(), QIcon::Normal, QIcon::On );

  m_orbitYAction = new QAction( this );
  m_orbitYAction->setToolTip( "Orbit Around Y Axis" );
  m_orbitYAction->setIcon( iconCameraOrbitY );
  m_orbitYAction->setCheckable( true );
  m_orbitYAction->setChecked( false );
  m_orbitYAction->setEnabled( false );
  connect( m_orbitYAction, SIGNAL(toggled(bool)), this, SLOT(toggledCameraOrbitY(bool)) );

  QIcon CameraOrbitZIcon( ":/images/CameraOrbitZ_off.png" );
  CameraOrbitZIcon.addFile( ":/images/CameraOrbitZ_on.png", QSize(), QIcon::Normal, QIcon::On );

  m_orbitZAction = new QAction( this );
  m_orbitZAction->setToolTip( "Orbit Around Z Axis" );
  m_orbitZAction->setIcon( CameraOrbitZIcon );
  m_orbitZAction->setCheckable( true );
  m_orbitZAction->setChecked( false );
  m_orbitZAction->setEnabled( false );
  connect( m_orbitZAction, SIGNAL(toggled(bool)), this, SLOT(toggledCameraOrbitZ(bool)) );

  m_normalsDialogAction = new QAction( this );
  m_normalsDialogAction->setToolTip( "Display Normals" );
  m_normalsDialogAction->setEnabled( false );
  m_normalsDialogAction->setIcon( QIcon( ":/images/Normals_off.png" ) );
  connect( m_normalsDialogAction, SIGNAL(triggered()), this, SLOT(triggeredNormalsDialog()) );

  // Add undo and redo actions. 
  // Icons only. The operation will be shown inside the tooltip and the undo widget stack.
  QIcon undoIcon( ":/images/Undo_off.png" );
  undoIcon.addFile( ":/images/Undo_on.png", QSize(), QIcon::Normal, QIcon::On );

  m_undoAction = GetSceneStateUndoStack().createUndoAction( this );
  m_undoAction->setIcon( undoIcon );

  QIcon redoIcon( ":/images/Redo_off.png" );
  redoIcon.addFile( ":/images/Redo_on.png", QSize(), QIcon::Normal, QIcon::On );

  m_redoAction = GetSceneStateUndoStack().createRedoAction( this );
  m_redoAction->setIcon(redoIcon);

  // make the manipulators mutually exclusive
  QActionGroup * manipulatorsActionGroup = new QActionGroup( this );
  manipulatorsActionGroup->setExclusive( true );

  for ( unsigned int i = 0; i < ViewerRendererWidget::MANIPULATOR_COUNT; i ++ )
  {
    m_manipulatorAction[i] = new QAction( this );
    QString name;
    QString toolTip;
    QVariant data;
    bool checked = false;

    switch( i )
    {
      case ViewerRendererWidget::MANIPULATOR_TRACKBALL:
        name = "TrackballCameraManipulator";
        toolTip = "Manipulate camera as if scene was encased in a sphere";
        data = ViewerRendererWidget::MANIPULATOR_TRACKBALL;
        checked = true;
        break;

      case ViewerRendererWidget::MANIPULATOR_CYLINDRICAL:
        name = "CylindricalCameraManipulator";
        toolTip = "Manipulate camera as if scene was encased in cylinder";
        data = ViewerRendererWidget::MANIPULATOR_CYLINDRICAL;
        break;

      case ViewerRendererWidget::MANIPULATOR_FLY:
        name = "FlightCameraManipulator";
        toolTip = "Manipulate camera as if flying through the scene";
        data = ViewerRendererWidget::MANIPULATOR_FLY;
        break;

      case ViewerRendererWidget::MANIPULATOR_WALK:
        name = "WalkCameraManipulator";
        toolTip = "Manipulate camera as if walking through the scene";
        data = ViewerRendererWidget::MANIPULATOR_WALK;
        break;

      default :
        DP_ASSERT( false );
    }

    m_manipulatorAction[i]->setObjectName( name );
    m_manipulatorAction[i]->setToolTip( toolTip );
    m_manipulatorAction[i]->setData( data );

    QIcon manipulatorIcon;
    name = QString(":/images/") + name;
    manipulatorIcon.addFile( name + QString("_off.png"), QSize(), QIcon::Normal, QIcon::Off );
    manipulatorIcon.addFile( name + QString("_on.png"), QSize(), QIcon::Normal, QIcon::On );
    m_manipulatorAction[i]->setIcon( manipulatorIcon );

    m_manipulatorAction[i]->setCheckable( true );
    m_manipulatorAction[i]->setChecked( checked );
    m_manipulatorAction[i]->setEnabled( false );
    manipulatorsActionGroup->addAction( m_manipulatorAction[i] );

    connect( m_manipulatorAction[i], SIGNAL(toggled(bool)), this, SLOT(toggledManipulator(bool)) );
  }
}

void MainWindow::setupDockWidgets()
{
  m_log = new LogWidget(this);
  this->addDockWidget( Qt::RightDockWidgetArea, m_log );
  m_log->hide();

  m_scriptSandbox = new ScriptWidget(this);
  this->addDockWidget( Qt::RightDockWidgetArea, m_scriptSandbox );
  m_scriptSandbox->hide();

  m_undo = new UndoWidget(this);
  this->addDockWidget( Qt::RightDockWidgetArea, m_undo );
  m_undo->hide();

  m_sceneTreeBrowser = new SceneTreeBrowser( this );
  this->addDockWidget( Qt::LeftDockWidgetArea, m_sceneTreeBrowser );

  m_sceneProperties = new ScenePropertiesWidget(this);
  this->addDockWidget(Qt::LeftDockWidgetArea, m_sceneProperties);

  m_materialBrowser = new MaterialBrowser( "Material Browser", this );
  this->addDockWidget( Qt::RightDockWidgetArea, m_materialBrowser );

  m_viewportLayout = new ViewportLayout( this );
  connect( m_viewportLayout, SIGNAL(activeViewportChanged(int,QWidget*)), this, SLOT(activeViewportChanged(int,QWidget*)));
  connect( m_viewportCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(viewportLayoutChanged(int)));

  setCentralWidget( m_viewportLayout );

  connect( m_sceneTreeBrowser, SIGNAL(currentItemChanged(dp::sg::core::ObjectSharedPtr,dp::sg::core::ObjectSharedPtr))
         , m_sceneProperties, SLOT(currentItemChanged(dp::sg::core::ObjectSharedPtr,dp::sg::core::ObjectSharedPtr)) );
}

void MainWindow::setupMenus()
{
  // File menu
  {
    QMenu * fileMenu = menuBar()->addMenu( "&File" );
    connect( fileMenu, SIGNAL(aboutToShow()), this, SLOT(aboutToShowFileMenu()) );
    fileMenu->addAction( m_openAction );
    fileMenu->addAction( m_closeAction );
    fileMenu->addAction( m_saveAction );
    fileMenu->addSeparator();

    m_analyzeSceneAction = fileMenu->addAction( "&Analyze Scene" );
    connect( m_analyzeSceneAction, SIGNAL(triggered(bool)), this, SLOT(triggeredAnalyzeScene(bool)) );

    m_sceneStatisticsAction = fileMenu->addAction( "S&tatistics" );
    connect( m_sceneStatisticsAction, SIGNAL(triggered(bool)), this, SLOT(triggeredSceneStatistics(bool)) );

    fileMenu->addSeparator();

    m_clearUndoStackAction = fileMenu->addAction( "Clear &Undo Stack" );
    connect( m_clearUndoStackAction, SIGNAL(triggered()), m_undo, SLOT(clear()) );

    QAction * actionPreferencesDialog = fileMenu->addAction( "&Preferences ..." );
    connect( actionPreferencesDialog, SIGNAL(triggered(bool)), this, SLOT(triggeredPreferencesDialog(bool)) );

    fileMenu->addSeparator();

    for ( int i=0 ; i<MaxRecentFiles ; ++i )
    {
      fileMenu->addAction( m_recentFileAction[i] );
    }
    m_separatorAction = fileMenu->addSeparator();
    fileMenu->addAction( m_quitAction );
    updateRecentFileActions();
  }

  // Edit menu
  {
    QMenu * editMenu = menuBar()->addMenu( "&Edit" );
    connect( editMenu, SIGNAL(aboutToShow()), this, SLOT(aboutToShowEditMenu()) );

    // Convert menu
    {
      m_menus[MID_CONVERT_SCENE] = editMenu->addMenu( "&Convert Scene" );
      connect( m_menus[MID_CONVERT_SCENE], SIGNAL(aboutToShow()), this, SLOT(aboutToShowConvertSceneMenu()) );

      m_destripSceneAction = m_menus[MID_CONVERT_SCENE]->addAction( "&Destrip Scene" );
      connect( m_destripSceneAction, SIGNAL(triggered(bool)), this, SLOT(triggeredDestripScene(bool)) );

      m_stripSceneAction = m_menus[MID_CONVERT_SCENE]->addAction( "&Strip Scene" );
      connect( m_stripSceneAction, SIGNAL(triggered(bool)), this, SLOT(triggeredStripScene(bool)) );

      m_menus[MID_CONVERT_SCENE]->addSeparator();

      m_triangulateSceneAction = m_menus[MID_CONVERT_SCENE]->addAction( "&Triangulate Scene" );
      connect( m_triangulateSceneAction, SIGNAL(triggered(bool)), this, SLOT(triggeredTriangulateScene(bool)) );
    }

    // Modify menu
    {
      m_menus[MID_MODIFY_SCENE] = editMenu->addMenu( "&Modify Scene" );

      QAction * action = m_menus[MID_MODIFY_SCENE]->addAction( "&Normalize Scene ..." );
      connect( action, SIGNAL(triggered(bool)), this, SLOT(triggeredNormalizeScene(bool)) );

      action = m_menus[MID_MODIFY_SCENE]->addAction( "&Smooth Scene ..." );
      connect( action, SIGNAL(triggered(bool)), this, SLOT(triggeredSmoothScene(bool)) );
    }

    // Optimize action
    m_optimizeSceneAction = editMenu->addAction( "&Optimize Scene ..." );
    connect( m_optimizeSceneAction, SIGNAL(triggered(bool)), this, SLOT(triggeredOptimizeScene(bool)) );

    editMenu->addSeparator();

    // Edit -> Add Headlight menu
    {
      m_menus[MID_ADD_HEADLIGHT] = new QMenu( "Add &Headlight" );
      m_menus[MID_ADD_HEADLIGHT]->addAction( "&Directed Light" );
      m_menus[MID_ADD_HEADLIGHT]->addAction( "&Point Light" );
      m_menus[MID_ADD_HEADLIGHT]->addAction( "&Spot Light" );
      for ( int i=0 ; i<m_menus[MID_ADD_HEADLIGHT]->actions().size() ; i++ )
      {
        m_menus[MID_ADD_HEADLIGHT]->actions()[i]->setData( i );
      }
      editMenu->addMenu( m_menus[MID_ADD_HEADLIGHT] );
    }

    // Edit -> Add Light Source menu
    {
      m_menus[MID_ADD_LIGHT_SOURCE] = new QMenu( "Add &Light Source" );
      m_menus[MID_ADD_LIGHT_SOURCE]->addAction( "&Directed Light" );
      m_menus[MID_ADD_LIGHT_SOURCE]->addAction( "&Point Light" );
      m_menus[MID_ADD_LIGHT_SOURCE]->addAction( "&Spot Light" );
      for ( int i=0 ; i<m_menus[MID_ADD_LIGHT_SOURCE]->actions().size() ; i++ )
      {
        m_menus[MID_ADD_LIGHT_SOURCE]->actions()[i]->setData( i );
      }
      editMenu->addMenu( m_menus[MID_ADD_LIGHT_SOURCE] );
    }
  }

  // View menu
  {
    QMenu * viewMenu = menuBar()->addMenu( "&View" );
    connect( viewMenu, SIGNAL(aboutToShow()), this, SLOT(aboutToShowViewMenu()) );

    // View -> Render Engine menu
    {
      m_menus[MID_RENDER_ENGINE] = new QMenu( "Render &Engine" );

      QAction * action = m_menus[MID_RENDER_ENGINE]->addAction( "&OpenGL" );
      action->setCheckable( true );
      action->setData( ViewerRendererWidget::RENDERER_RASTERIZE_XBAR );

      // DAR FIXME Switching the renderer dynamically doesn't work, yet.
      // The OpenGL driver (310.90 QK5000) throws an error about invalid clear bits and 
      // a performance warning that the pixel pipeline is synchronized with the 3D engine.
      // Rendering is corrupted.
      //action = m_menus[MID_RENDER_ENGINE]->addAction( "Opti&X" );
      //action->setCheckable( true );
      //action->setData( RENDERER_RAYTRACE_XBAR );

      viewMenu->addMenu( m_menus[MID_RENDER_ENGINE] );
    }

    // View -> Viewport Format menu
    {
      m_menus[MID_VIEWPORT_FORMAT] = new QMenu( "&Viewport Formats" );

      m_viewportFormat30BitAction = m_menus[MID_VIEWPORT_FORMAT]->addAction( "&30 Bit" );
      m_viewportFormat30BitAction->setCheckable( true );

      // View -> Viewport Format -> Antialiasing menu
      {
        m_menus[MID_ANTIALIASING] = new QMenu( "&Antialiasing" );
        m_menus[MID_VIEWPORT_FORMAT]->addMenu( m_menus[MID_ANTIALIASING] );
        // the actions of this submenu can be created in when there is a ViewerRenderer created, earliest,
        // as a Renderer is needed to determine the multisample support
      }

      m_viewportFormatSRGBAction = m_menus[MID_VIEWPORT_FORMAT]->addAction( "s&RGB" );
      m_viewportFormatSRGBAction->setCheckable( true );

      m_viewportFormatStencilAction = m_menus[MID_VIEWPORT_FORMAT]->addAction( "&Stencil" );
      m_viewportFormatStencilAction->setCheckable( true );

      m_viewportFormatStereoAction = m_menus[MID_VIEWPORT_FORMAT]->addAction( "S&tereo" );
      m_viewportFormatStereoAction->setCheckable( true );

      viewMenu->addMenu( m_menus[MID_VIEWPORT_FORMAT] );
    }

    // add separator between viewport specific and renderer specific menu entries
    viewMenu->addSeparator();

    // View -> Culling menu
    {
      m_menus[MID_CULLING] = new QMenu( "&Culling" );
      m_menus[MID_CULLING]->addAction( "CPU" );
      m_menus[MID_CULLING]->addAction( "OpenGL Compute" );
      m_menus[MID_CULLING]->addAction( "CUDA" );
      m_menus[MID_CULLING]->addAction( "Auto" );
      m_menus[MID_CULLING]->addAction( "Off" );
      for ( int i=0 ; i<m_menus[MID_CULLING]->actions().size() ; i++ )
      {
        m_menus[MID_CULLING]->actions()[i]->setCheckable( true );
      }
      viewMenu->addMenu( m_menus[MID_CULLING] );
    }

    // View -> Depth Pass
    {
      m_depthPassAction = viewMenu->addAction( "&Depth Pass" );
      m_depthPassAction->setCheckable( true );
      connect( m_depthPassAction, SIGNAL(triggered(bool)), this, SLOT(triggeredDepthPass(bool)) );
    }

    // View -> Stereo ... action
    {
      m_stereoDialogAction = viewMenu->addAction( "&Stereo ..." );
      m_stereoDialogAction->setCheckable( true );
      connect( m_stereoDialogAction, SIGNAL(triggered()), this, SLOT(triggeredStereoDialog()) );
    }

    // View -> Tonemapper ... menu
    {
      m_tonemapperDialogAction = viewMenu->addAction( "Tone&mapper ..." );
      connect( m_tonemapperDialogAction, SIGNAL(triggered()), this, SLOT(triggeredTonemapperDialog()) );
    }

    // View -> Transparency ... menu
    {
      m_transparencyDialogAction = viewMenu->addAction( "&Transparency ..." );
      connect( m_transparencyDialogAction, SIGNAL(triggered()), this, SLOT(triggeredTransparencyDialog()) );
    }

  } // View Menu scope end

  // separator to Help menu
  menuBar()->addSeparator();
 
  // Help menu
  QAction * actionAbout = new QAction( "&About", this );
  connect( actionAbout, SIGNAL(triggered()), this, SLOT(triggeredAbout()) );

  QMenu * helpMenu = menuBar()->addMenu( "&Help" );
  helpMenu->addAction( actionAbout );
}

void MainWindow::setupStatusBar()
{
  m_fpsLabel = new QLabel( this );
  m_fpsLabel->setMargin( 2 );
  m_fpsLabel->setStyleSheet( "background-color:transparent;" );
  m_fpsTimerID = startTimer(500);     // check every half second for fps display
  m_timer.start();                    // timer used to get fps information

  m_statisticsLabel = new QLabel( this );
  m_statisticsLabel->setMargin( 2 );
  m_statisticsLabel->setStyleSheet( "background-color:transparent;" );

  QStatusBar * sb = statusBar();
  sb->setSizeGripEnabled(false);
  sb->addPermanentWidget( m_fpsLabel );
  sb->addPermanentWidget( m_statisticsLabel );
}

void MainWindow::setupToolbar()
{
  QToolBar * toolBar = new QToolBar( "ToolBar" );
  toolBar->setObjectName( "toolBar" );    // needed, to keep saveState quiet
  toolBar->setStyleSheet( "QToolButton\n"
                            "{\n"
                            " background-color:transparent;\n"
                            " border:1px solid transparent;\n"
                            " border-radius:25px;\n"
                            "}\n"
                            "QToolButton:hover\n"
                            "{\n"
                            " color:#FFFFFF;\n"
                            " background-color:#b4b4b4;\n"
                            " border-color:#969696;\n"
                            "}\n"
                            "QToolButton:pressed\n"
                            "{\n"
                            " color:#b4b4b4;\n"
                            " background-color:#5a5a5a;\n"
                            " border-style:inset;\n"
                            " border-color:#3c3c3c;\n"
                            "}\n"
                            "");
  toolBar->setIconSize( QSize( 48, 48 ) );
  toolBar->setToolButtonStyle( Qt::ToolButtonIconOnly );
  toolBar->addAction( m_openAction );
  toolBar->addAction( m_saveAction );
  toolBar->addSeparator();
  toolBar->addWidget( m_viewportCombo );
  toolBar->addAction( m_continuousRedrawAction );
  toolBar->addSeparator();
  toolBar->addAction( m_zoomAllAction );
  toolBar->addAction( m_cameraIterationAction );
  toolBar->addAction( m_orbitXAction );
  toolBar->addAction( m_orbitYAction );
  toolBar->addAction( m_orbitZAction );
  toolBar->addAction( m_normalsDialogAction );
  toolBar->addSeparator();
  toolBar->addAction( m_undoAction );
  toolBar->addAction( m_redoAction );
  toolBar->addSeparator();
  for( unsigned int i = 0; i < ViewerRendererWidget::MANIPULATOR_COUNT; i ++ )
  {
    toolBar->addAction( m_manipulatorAction[i] );
  }

  addToolBar( Qt::TopToolBarArea, toolBar );
}

// start the timer if needed, kill the timer if allowed
void MainWindow::startStopTimer( bool & currentFlag, bool newFlag, bool killIt )
{
  if ( currentFlag != newFlag )
  {
    if ( newFlag != ( m_continuousRedrawTimerID != 0 ) )   // timer needs to be started or killed
    {
      if ( newFlag )
      {
        if ( ! m_continuousRedrawTimerID )
        {
          m_continuousRedrawTimerID = startTimer(0);
        }
      }
      else if ( killIt )
      {
        killTimer( m_continuousRedrawTimerID );
        m_continuousRedrawTimerID = 0;
      }
    }
    currentFlag = newFlag;
  }
}

void MainWindow::timerEvent( QTimerEvent * te )
{
  if ( te->timerId() == m_continuousRedrawTimerID )
  {
    DP_ASSERT( m_currentViewport != ~0 );
    // Special action to update without restarting accumulation.
    m_renderWidgets[m_currentViewport]->update();
    //GetApp()->emitContinuousUpdate();   // use this to trigger redraw in all viewports
  }
  else if ( ( te->timerId() == m_fpsTimerID ) && ( m_currentViewport != ~0 ) )
  {
    double currentTime = m_timer.getTime();
    DP_ASSERT( m_lastTime < currentTime );

    unsigned int currentFrame = m_renderWidgets[m_currentViewport]->getFrameCount();
    unsigned int fps = ( currentFrame - m_lastFrame ) / ( currentTime - m_lastTime );

    m_fpsLabel->setText( QString( "FPS: %1" ).arg( fps ) );

    m_lastTime = currentTime;
    m_lastFrame = currentFrame;
  }
}

void MainWindow::toggledCameraCycle( bool onoff )
{
  CameraAnimator * ca = static_cast<CameraAnimator*>(getCurrentCameraAnimator());
  if ( ca )
  {
    if ( onoff )
    {
      if ( ca->isCameraOrbitX() )
      {
        ca->cameraOrbitX( false );
        m_orbitXAction->setChecked( false );
      }
      if ( ca->isCameraOrbitY() )
      {
        ca->cameraOrbitY( false );
        m_orbitYAction->setChecked( false );
      }
      if ( ca->isCameraOrbitZ() )
      {
        ca->cameraOrbitZ( false );
        m_orbitZAction->setChecked( false );
      }
    }
    ca->cameraCycle( onoff );
    if ( onoff && !ca->isCameraCycle() )
    {
      // special case: just one camera doesn't iterate -> uncheck the button
      m_cameraIterationAction->setChecked( !onoff );
    }
  }
}

void MainWindow::toggledCameraOrbitX( bool onoff )
{
  CameraAnimator * ca = static_cast<CameraAnimator*>(getCurrentCameraAnimator());
  if ( ca )
  {
    if ( onoff && ca->isCameraCycle() )
    {
      ca->cameraCycle( false );
      m_cameraIterationAction->setChecked( false );
    }
    ca->cameraOrbitX( onoff );
  }
}

void MainWindow::toggledCameraOrbitY( bool onoff )
{
  CameraAnimator * ca = static_cast<CameraAnimator*>(getCurrentCameraAnimator());
  if ( ca )
  {
    if ( onoff && ca->isCameraCycle() )
    {
      ca->cameraCycle( false );
      m_cameraIterationAction->setChecked( false );
    }
    ca->cameraOrbitY( onoff );
  }
}

void MainWindow::toggledCameraOrbitZ( bool onoff )
{
  CameraAnimator * ca = static_cast<CameraAnimator*>(getCurrentCameraAnimator());
  if ( ca )
  {
    if ( onoff && ca->isCameraCycle() )
    {
      ca->cameraCycle( false );
      m_cameraIterationAction->setChecked( false );
    }
    ca->cameraOrbitZ( onoff );
  }
}

void MainWindow::toggledContinuousRedraw( bool onoff )
{
  startStopTimer( m_continuousRedraw, onoff, true );
}

void MainWindow::toggledManipulator( bool onoff )
{
  // we only care about the activate signal
  if ( onoff )
  {
    QAction * who = static_cast<QAction*>(sender());
    emit manipulatorChanged( static_cast<ViewerRendererWidget::ManipulatorType>( who->data().toInt() ) );
  }
}

void MainWindow::triggeredAbout()
{
  // load help file from resource
  QFile qfile( ":/ui/HelpAbout.html" );
  qfile.open( QIODevice::ReadOnly );
  QString text( qfile.readAll() );

  QMessageBox::about( this, "About SceniX Viewer", text.toStdString().c_str() );
}

void MainWindow::triggeredAnalyzeScene( bool checked )
{
  DP_ASSERT( GetApp()->getScene() );

  PlainTextDialog dlg( "Scene Analysis", "Analyze Results for file " + GetApp()->getDisplayedSceneName(), analyze( GetApp()->getScene() ), this );
  dlg.exec();
}

void MainWindow::triggeredClose()
{
  DP_ASSERT( GetApp()->getViewState() );

  // clear filename
  setCurrentFile( "", false );

  // this kicks everything off
  GetApp()->unloadScene();
}

void MainWindow::triggeredDepthPass( bool checked )
{
  DP_ASSERT( m_renderWidgets[m_currentViewport]->getSceneRenderer().isPtrTo<dp::sg::renderer::rix::gl::SceneRenderer>() );
  m_renderWidgets[m_currentViewport]->getSceneRenderer().staticCast<dp::sg::renderer::rix::gl::SceneRenderer>()->setDepthPass( checked );
  GetApp()->getPreferences()->setDepthPass( checked );
}

void MainWindow::triggeredDestripScene( bool checked )
{
  GetApp()->setOverrideCursor( Qt::WaitCursor );
  dp::sg::algorithm::DestrippingTraverser destrippingTraverser;
  DP_ASSERT( GetApp()->getScene() );
  destrippingTraverser.apply( GetApp()->getScene() );
  if ( destrippingTraverser.getTreeModified() )
  {
    GetApp()->emitSceneTreeChanged();
    GetApp()->outputStatistics();
  }
  GetApp()->restoreOverrideCursor();
}

void MainWindow::triggeredNormalizeScene( bool checked )
{
  DP_ASSERT( GetApp()->getScene() );
  NormalizeDialog dlg( GetApp()->getScene(), this );
  if( dlg.exec() == QDialog::Accepted )
  {
    GetApp()->emitViewChanged();
  }
}

void MainWindow::triggeredNormalsDialog() 
{
  float length = GetPreferences()->getNormalsLineLength();

  NormalsDialog dlg( this, m_normalsDisplayed, length, m_normalsColor );
  if ( dlg.exec() == QDialog::Accepted )
  {
    dp::sg::core::SceneSharedPtr scene = GetApp()->getScene();
    if ( scene )
    {
      dlg.apply( scene );
    }

    dlg.getOptions( m_normalsDisplayed, length, m_normalsColor );

    GetPreferences()->setNormalsLineLength( length );

    GetApp()->emitSceneTreeChanged();
  }
}

void MainWindow::triggeredOpen()
{
  QString sceneFile = QFileDialog::getOpenFileName( this, "Open File", GetApp()->getPreferences()->getSceneSelectionPath() , m_sceneLoaderFilter.c_str() );
  if ( ! sceneFile.isEmpty() )
  {
    loadFile( sceneFile, false );
    GetApp()->getPreferences()->setSceneSelectionPath( QFileInfo( sceneFile ).absolutePath() );
  }
}

void MainWindow::triggeredOptimizeScene( bool checked )
{
  DP_ASSERT( GetApp()->getScene() );
  OptimizerDialog dlg( GetApp()->getScene(), this );
  if( dlg.exec() == QDialog::Accepted )
  {
    GetApp()->emitSceneTreeChanged();
    GetApp()->outputStatistics();
  }
}

void MainWindow::triggeredPreferencesDialog( bool checked )
{
  PreferencesDialog dlg( this );
  if ( dlg.exec() == QDialog::Rejected )
  {
    dlg.restore();
  }
}

void MainWindow::triggeredQuit()
{
  saveSettings();
  GetApp()->quit();
}

void MainWindow::triggeredRecentFile()
{
  QAction *action = qobject_cast<QAction *>(sender());
  DP_ASSERT( action );
  loadFile( action->data().toString(), false );
}

void MainWindow::triggeredSave()
{
  DP_ASSERT( GetApp()->getViewState() );

  if ( m_normalsDisplayed )
  {
    QMessageBox msgbox;
    msgbox.setText( "The scene contains visualized Normals." );
    msgbox.setInformativeText( "SAVE will save the scene including lines representing the Normals.\n"
                               "REMOVE will remove the lines before saving." );
    msgbox.addButton( "Save", QMessageBox::AcceptRole );
    msgbox.addButton( "Remove", QMessageBox::RejectRole );

    if ( msgbox.exec() == QMessageBox::RejectRole )
    {
      removeNormals();
    }
  }

  QString data = QDir::fromNativeSeparators( QStandardPaths::standardLocations( QStandardPaths::DocumentsLocation ).front().append( "/NVIDIA Corporation/SceniX Viewer/" ) );
  QString fileName = QFileDialog::getSaveFileName( this, "Save File", data, "Scenes (*.dpbf *.dpaf *.obj *.csf)" );
  if ( !fileName.isEmpty() )
  {
    GetApp()->setOverrideCursor( Qt::WaitCursor );
    if ( !GetApp()->saveScene( fileName ) )
    {
      QMessageBox::critical( 0, "dp::sg::core::Scene Save Error", "Unable to save file: " + fileName );
    }
    else
    {
      // ok to add to current file list if we are saving, since it should contain the possibly replaced shaders
      setCurrentFile( fileName, true );
    }
    GetApp()->restoreOverrideCursor();
  }
}

void MainWindow::triggeredSceneStatistics( bool checked )
{
  DP_ASSERT( GetApp()->getScene() );

  dp::sg::algorithm::StatisticsTraverser statisticsTraverser;
  statisticsTraverser.apply( GetApp()->getScene() );
  std::stringstream ss;
  ss << statisticsTraverser;

  PlainTextDialog dlg( "Scene Statistics", "Statistics Results for file " + GetApp()->getDisplayedSceneName(), ss.str().c_str(), this );
  dlg.exec();
}

void MainWindow::triggeredSmoothScene( bool checked )
{
  DP_ASSERT( GetApp()->getScene() );
  SmoothDialog dlg( GetApp()->getScene(), this );
  if( dlg.exec() == QDialog::Accepted )
  {
    GetApp()->emitViewChanged();
  }
}

void MainWindow::triggeredStereoDialog() 
{
  StereoDialog dlg( this, getCurrentViewport() );
  if( dlg.exec() == QDialog::Rejected )
  {
    dlg.restore();
  }
}

void MainWindow::triggeredStripScene( bool checked )
{
  GetApp()->setOverrideCursor( Qt::WaitCursor );
  dp::sg::algorithm::StrippingTraverser strippingTraverser;
  DP_ASSERT( GetApp()->getScene() );
  strippingTraverser.apply( GetApp()->getScene() );
  if ( strippingTraverser.getTreeModified() )
  {
    GetApp()->emitSceneTreeChanged();
    GetApp()->outputStatistics();
  }
  GetApp()->restoreOverrideCursor();
}

void MainWindow::triggeredTonemapperDialog()
{
  TonemapperDialog dlg( this, getCurrentViewport() );
  dlg.exec();
}

void MainWindow::triggeredTransparencyDialog()
{
  TransparencyDialog dlg( this, getCurrentViewport() );
  dlg.exec();
}

void MainWindow::triggeredTriangulateScene( bool checked )
{
  GetApp()->setOverrideCursor( Qt::WaitCursor );
  dp::sg::algorithm::TriangulateTraverser triangulateTraverser;
  DP_ASSERT( GetApp()->getScene() );
  triangulateTraverser.apply( GetApp()->getScene() );
  if ( triangulateTraverser.getTreeModified() )
  {
    GetApp()->emitSceneTreeChanged();
    GetApp()->outputStatistics();
  }
  GetApp()->restoreOverrideCursor();
}

void MainWindow::triggeredZoomAll()
{
  CameraAnimator * ca = static_cast<CameraAnimator*>(getCurrentCameraAnimator());
  if ( ca )
  {
    if ( ca->isCameraCycle() )
    {
      ca->cameraCycle( false );
      m_cameraIterationAction->setChecked( false );
    }
    if ( ca->isCameraOrbitX() )
    {
      ca->cameraOrbitX( false );
      m_orbitXAction->setChecked( false );
    }
    if ( ca->isCameraOrbitY() )
    {
      ca->cameraOrbitY( false );
      m_orbitYAction->setChecked( false );
    }
    if ( ca->isCameraOrbitZ() )
    {
      ca->cameraOrbitZ( false );
      m_orbitZAction->setChecked( false );
    }
    ca->zoomAll();
  }
}

void MainWindow::updateRecentFileActions()
{
  QSettings settings( VIEWER_APPLICATION_VENDOR, VIEWER_APPLICATION_NAME );
  QStringList files = settings.value( "recentFileList" ).toStringList();

  int numRecentFiles = std::min( files.size(), (int)MaxRecentFiles );
  for ( int i=0 ; i<numRecentFiles ; ++i )
  {
    QString text = tr("&%1 %2").arg(i+1).arg( QFileInfo(files[i]).fileName() );
    m_recentFileAction[i]->setText( text );
    m_recentFileAction[i]->setData( files[i] );
    m_recentFileAction[i]->setVisible( true );
  }
  for ( int i=numRecentFiles ; i<MaxRecentFiles ; ++i )
  {
    m_recentFileAction[i]->setVisible( false );
  }

  m_separatorAction->setVisible( 0 < numRecentFiles );
}

void MainWindow::updateRenderers( dp::sg::ui::ViewStateSharedPtr const& viewState )
{
  if ( viewState && viewState->getScene() )
  {
    if ( m_renderWidgets.empty() )
    {
      setTraversalMasks( viewState->getScene(), 0x01 );

      // create one RenderWidget, as we always start with VIEWPORT_LAYOUT_ONE
      m_renderWidgets.push_back( createRenderer( m_viewportLayout, viewState->getScene() ) );
      m_cameraAnimators.push_back( createAnimator( m_renderWidgets.back() ) );

      // add it to the viewport layout
      m_viewportLayout->setViewport( 0, m_renderWidgets.back() );
      // Trigger this at least once (because the default m_currentLayout is ~0) or the ViewportLayout doesn't refresh properly.
      m_viewportLayout->setViewportLayout( VIEWPORT_LAYOUT_ONE );

      // Reset the toolbar to the current state, possibly triggered setViewportLayout() finds that state active.
      m_viewportCombo->setCurrentIndex( VIEWPORT_LAYOUT_ONE );
    }
    else
    {
      // reset all renderWidgets and animators
      DP_ASSERT( m_renderWidgets.size() == m_cameraAnimators.size() );
      for ( size_t i=0 ; i<m_renderWidgets.size() ; i++ )
      {
        // make sure any animation is canceled before pulling out the carpet
        m_cameraAnimators[i]->cancel();

        m_renderWidgets[i]->setScene( viewState->getScene() );
        // have to do this second in case renderer creates a new ViewState
        m_cameraAnimators[i]->setViewState( m_renderWidgets[i]->getViewState() );
      }
      // they are already attached to the viewportLayout
    }

    // now, set the cameras
    std::vector<dp::sg::core::CameraSharedPtr> cameras;
    bool sceneChanged = findOrCreateCameras( viewState, cameras ); // Returns true if cameras have been added.
    DP_ASSERT( m_renderWidgets.size() <= cameras.size() );

    for( size_t i=0 ; i<m_renderWidgets.size() ; i++ )
    {
      m_renderWidgets[i]->setCamera( cameras[i] );
    }

#if 0
    // This dialog box disables all input on the viewer window for some unknown reason. Disable it for now.
    if ( !dp::sg::algorithm::containsLight( viewState->getScene() )  && ( dp::sg::core::CameraLock( cameras[0] )->getNumberOfHeadLights() == 0 ) )
    {
      QMessageBox msgbox( this ); // MainWindow as parent to center the dialog on the application.
      
      msgbox.setIcon( QMessageBox::Question );
      msgbox.setText( "Neither scene nor first camera contain lights." );
      msgbox.setInformativeText( "Add Headlights - will add a PointLight as headlight to each camera in the scene when needed.\n"
                                 "Cancel - will leave the scene as is. Lights can be added via right-click context menus later." );
      msgbox.addButton( "Add Headlights", QMessageBox::AcceptRole );
      msgbox.addButton( "Cancel" , QMessageBox::RejectRole );

      // Currently in WaitCursor mode. Change to indicate that user input is required.
      GetApp()->setOverrideCursor( Qt::ArrowCursor );
      switch ( msgbox.exec() )
      {
        case QMessageBox::AcceptRole:
          for ( size_t i = 0; i < cameras.size(); ++i )
          {
            dp::sg::core::CameraLock camera( cameras[i] );
            if ( camera->getNumberOfHeadLights() == 0 )
            {
              // Using plain defaults!
              // Note that the LightSource default ambientColor is black, matching the OpenGL default.
              dp::sg::core::LightSourceSharedPtr pointLight = dp::sg::core::createStandardPointLight();
              dp::sg::core::LightSourceLock( pointLight )->setName( "SVPointLight" );
              camera->addHeadLight( pointLight );
              sceneChanged = true;
            }
          }
          break;

        case QMessageBox::RejectRole:
          // Leave scene as is.
          break;
      }
      GetApp()->restoreOverrideCursor();
    }
#endif
  }
  else
  {
    if ( ! m_renderWidgets.empty() )
    {
      // clear everything
      m_viewportLayout->clear();
      m_cameraAnimators.clear();
      m_renderWidgets.clear();
      m_currentViewport = ~0;
    }
  }
}

void MainWindow::viewportLayoutChanged( int index )
{
  unsigned int numViewports = viewportCount( index );
  if ( numViewports < m_renderWidgets.size() )
  {
    for ( unsigned int i=numViewports ; i<m_renderWidgets.size() ; i++ )
    {
      m_viewportLayout->setViewport( i, nullptr );
    }
    m_renderWidgets.resize( numViewports );   // calls destructors for all ViewerRendererWidgets above numViewports
    m_cameraAnimators.resize( numViewports ); // calls destructors for all CameraAnimators above numViewports
  }
  else if ( m_renderWidgets.size() < numViewports )
  {
    std::vector<dp::sg::core::CameraSharedPtr> cameras;
    bool sceneChanged = findOrCreateCameras( GetApp()->getViewState(), cameras ); // Returns true if cameras have been added.

    for ( unsigned int i = (unsigned int)m_renderWidgets.size() ; i<numViewports ; i++ )
    {
      m_renderWidgets.push_back( createRenderer( m_viewportLayout, GetApp()->getScene() ) );
      m_cameraAnimators.push_back( createAnimator( m_renderWidgets.back() ) );

      // add it to the viewport layout
      m_viewportLayout->setViewport( i, m_renderWidgets.back() );

      m_renderWidgets[i]->setCamera( cameras[i] );
    }
  }
  m_viewportLayout->setViewportLayout( index );
}

QString analyze( const dp::sg::core::SceneSharedPtr & scene )
{
  GetApp()->setOverrideCursor( Qt::WaitCursor );

  dp::sg::algorithm::AnalyzeTraverser analyzeTraverser;
  analyzeTraverser.apply( scene );
  std::vector<dp::sg::algorithm::AnalyzeResult *> results;
  analyzeTraverser.getAnalysis( results );
  QString plainText;
  if ( results.empty() )
  {
    plainText = "No results found!";
  }
  else
  {
    for ( size_t i=0 ; i<results.size() ; i++ )
    {
      plainText += QString::number( i ) + ": ";
      if ( dynamic_cast<dp::sg::algorithm::CombinableResult *>(results[i]) )
      {
        dp::sg::algorithm::CombinableResult * cr = dynamic_cast<dp::sg::algorithm::CombinableResult *>(results[i]);
        plainText += "\tCombinableResult:\n";
        plainText += "\t\tobject code: " + QString( dp::sg::core::objectCodeToName( cr->objectCode ).c_str() ) + "\n";
        plainText += "\t\tnumber of objects: " + QString::number( cr->objectCount ) + "\n";
        plainText += "\t\tnumber of combinable objects: " + QString::number( cr->combinableCount ) + "\n";
        plainText += "\t\tnumber of objects if combined: " + QString::number( cr->reducedCount ) + "\n";
      }
      else if ( dynamic_cast<dp::sg::algorithm::DenormalizedNormalsResult *>(results[i]) )
      {
        dp::sg::algorithm::DenormalizedNormalsResult * dnr = dynamic_cast<dp::sg::algorithm::DenormalizedNormalsResult *>(results[i]);
        plainText += "\tDenormalizedNormalsResult (VertexAttribute with denormalized normals):\n";
        plainText += "\t\tobject code: " + QString( dp::sg::core::objectCodeToName( dnr->objectCode ).c_str() ) + "\n";
        plainText += "\t\tnumber of objects: " + QString::number( dnr->objectCount ) + "\n";
        plainText += "\t\tnumber of objects with denormalized normals: " + QString::number( dnr->denormalizedNormalsCount ) + "\n";
        plainText += "\t\tnumber of denormalized normals in those objects: " + QString::number( dnr->denormalizedNormalsNumber ) + "\n";
      }
      else if ( dynamic_cast<dp::sg::algorithm::EmptyResult *>(results[i]) )
      {
        dp::sg::algorithm::EmptyResult * er = dynamic_cast<dp::sg::algorithm::EmptyResult *>(results[i]);
        plainText += "\tEmptyResult:\n";
        plainText += "\t\tobject code: " + QString( dp::sg::core::objectCodeToName( er->objectCode ).c_str() ) + "\n";
        plainText += "\t\tnumber of objects: " + QString::number( er->objectCount ) + "\n";
        plainText += "\t\tnumber of empty objects: " + QString::number( er->emptyCount ) + "\n";
      }
      else if ( dynamic_cast<dp::sg::algorithm::EquivalenceResult *>(results[i]) )
      {
        dp::sg::algorithm::EquivalenceResult * er = dynamic_cast<dp::sg::algorithm::EquivalenceResult *>(results[i]);
        plainText += "\tEquivalenceResult:\n";
        plainText += "\t\tobject code: " + QString( dp::sg::core::objectCodeToName( er->objectCode ).c_str() ) + "\n";
        plainText += "\t\tnumber of objects: " + QString::number( er->objectCount ) + "\n";
        plainText += "\t\tnumber of equivalent objects: " + QString::number( er->equivalentCount ) + "\n";
        plainText += "\t\tnumber of objects if combined: " + QString::number( er->reducedCount ) + "\n";
      }
      else if ( dynamic_cast<dp::sg::algorithm::IdentityResult *>(results[i]) )
      {
        dp::sg::algorithm::IdentityResult * ir = dynamic_cast<dp::sg::algorithm::IdentityResult *>(results[i]);
        plainText += "\tIdentityResult (Identity Transforms):\n";
        plainText += "\t\tobject code: " + QString( dp::sg::core::objectCodeToName( ir->objectCode ).c_str() ) + "\n";
        plainText += "\t\tnumber of objects: " + QString::number( ir->objectCount ) + "\n";
        plainText += "\t\tnumber of identity Transforms: " + QString::number( ir->identityCount ) + "\n";
      }
      else if ( dynamic_cast<dp::sg::algorithm::MissingResult *>(results[i]) )
      {
        dp::sg::algorithm::MissingResult * mr = dynamic_cast<dp::sg::algorithm::MissingResult *>(results[i]);
        plainText += "\tMissingResult (non-complete Objects):\n";
        plainText += "\t\tobject code: " + QString( dp::sg::core::objectCodeToName( mr->objectCode ).c_str() ) + "\n";
        plainText += "\t\tnumber of objects: " + QString::number( mr->objectCount ) + "\n";
        plainText += "\t\tnumber of incomplete objects: " + QString::number( mr->missingCount ) + "\n";
      }
      else if ( dynamic_cast<dp::sg::algorithm::NullNormalsResult *>(results[i]) )
      {
        dp::sg::algorithm::NullNormalsResult * nnr = dynamic_cast<dp::sg::algorithm::NullNormalsResult *>(results[i]);
        plainText += "\tNullNormalsResult (VertexAttributeSet with null normals):\n";
        plainText += "\t\tobject code: " + QString( dp::sg::core::objectCodeToName( nnr->objectCode ).c_str() ) + "\n";
        plainText += "\t\tnumber of objects: " + QString::number( nnr->objectCount ) + "\n";
        plainText += "\t\tnumber of objects with null normals: " + QString::number( nnr->nullNormalsCount ) + "\n";
        plainText += "\t\tnumber of null normals in those objects: " + QString::number( nnr->nullNormalsNumber ) + "\n";
      }
      else if ( dynamic_cast<dp::sg::algorithm::RedundantIndexSetResult *>(results[i]) )
      {
        dp::sg::algorithm::RedundantIndexSetResult * risr = dynamic_cast<dp::sg::algorithm::RedundantIndexSetResult *>(results[i]);
        plainText += "\tRedundantIndexSetResult:\n";
        plainText += "\t\tnumber of Primitives: " + QString::number( risr->objectCount ) + "\n";
        plainText += "\t\tnumber of Primitives with redundant IndexSet: " + QString::number( risr->redundantIndexSetCount ) + "\n";
      }
      else if ( dynamic_cast<dp::sg::algorithm::RedundantLODRangesResult *>(results[i]) )
      {
        dp::sg::algorithm::RedundantLODRangesResult * rlodrr = dynamic_cast<dp::sg::algorithm::RedundantLODRangesResult *>(results[i]);
        plainText += "\tRedundantLODRangesResult:\n";
        plainText += "\t\tnumber of LODs: " + QString::number( rlodrr->objectCount ) + "\n";
        plainText += "\t\tnumber of LODs with redundant ranges: " + QString::number( rlodrr->redundantLODs ) + "\n";
        plainText += "\t\tnumber of ranges: " + QString::number( rlodrr->lodRanges ) + "\n";
        plainText += "\t\tnumber of redundant ranges: " + QString::number( rlodrr->redundantLODRanges ) + "\n";
      }
      else if ( dynamic_cast<dp::sg::algorithm::SingleChildResult *>(results[i]) )
      {
        dp::sg::algorithm::SingleChildResult * scr = dynamic_cast<dp::sg::algorithm::SingleChildResult *>(results[i]);
        plainText += "\tSingleChildResult:\n";
        plainText += "\t\tobject code: " + QString( dp::sg::core::objectCodeToName( scr->objectCode ).c_str() ) + "\n";
        plainText += "\t\tnumber of objects: " + QString::number( scr->objectCount ) + "\n";
        plainText += "\t\tnumber of objects with a single child: " + QString::number( scr->singleChildCount ) + "\n";
      }
      else if ( dynamic_cast<dp::sg::algorithm::ShortStrippedResult *>(results[i]) )
      {
        dp::sg::algorithm::ShortStrippedResult * ssr = dynamic_cast<dp::sg::algorithm::ShortStrippedResult *>(results[i]);
        plainText += "\tShortStrippedResult:\n";
        plainText += "\t\tobject code: " + QString( dp::sg::core::objectCodeToName( ssr->objectCode ).c_str() ) + "\n";
        plainText += "\t\tnumber of objects: " + QString::number( ssr->objectCount ) + "\n";
        plainText += "\t\tnumber of objects with short strips: " + QString::number( ssr->shortStrippedCount ) + "\n";
        plainText += "\t\tnumber of short strips in those objects: " + QString::number( ssr->shortStrippedNumber ) + "\n";
      }
      else if ( dynamic_cast<dp::sg::algorithm::UnusedVerticesResult *>(results[i]) )
      {
        dp::sg::algorithm::UnusedVerticesResult * uvr = static_cast<dp::sg::algorithm::UnusedVerticesResult*>(results[i]);
        plainText += "\tUnusedVerticesResult:\n";
        plainText += "\t\tobject code: " + QString( dp::sg::core::objectCodeToName( uvr->objectCode ).c_str() ) + "\n";
        plainText += "\t\tnumber of object: " + QString::number( uvr->objectCount ) + "\n";
        plainText += "\t\tnumber of unused vertices: " + QString::number( uvr->unusedVerticesCount ) + "\n";
      }
      else
      {
        dp::sg::algorithm::AnalyzeResult * ar = results[i];
        plainText += "\tAnalyzeResult:\n";
        plainText += "\t\tobject code: " + QString( dp::sg::core::objectCodeToName( ar->objectCode ).c_str() ) + "\n";
        plainText += "\t\tnumber of objects: " + QString::number( ar->objectCount ) + "\n";
      }
      delete results[i];    // delete the result after interpretation
    }
  }
  GetApp()->restoreOverrideCursor();
  return( plainText );
}

bool containsQuadPrimitives( std::vector<dp::sg::core::ObjectWeakPtr> const & vp )
{
  for ( size_t i=0 ; i<vp.size() ; i++ )
  {
    DP_ASSERT( dynamic_cast<dp::sg::core::PrimitiveWeakPtr>(vp[i]) );
    dp::sg::core::PrimitiveType pt = static_cast<dp::sg::core::PrimitiveWeakPtr>(vp[i])->getPrimitiveType();
    if ( ( pt == dp::sg::core::PRIMITIVE_QUADS ) || ( pt == dp::sg::core::PRIMITIVE_QUAD_STRIP ) )
    {
      return( true );
    }
  }
  return( false );
}

bool containsStripablePrimitives( std::vector<dp::sg::core::ObjectWeakPtr> const & vp )
{
  for ( size_t i=0 ; i<vp.size() ; i++ )
  {
    DP_ASSERT( dynamic_cast<dp::sg::core::PrimitiveWeakPtr>(vp[i]) );
    dp::sg::core::PrimitiveType pt = static_cast<dp::sg::core::PrimitiveWeakPtr>(vp[i])->getPrimitiveType();
    if ( ( pt == dp::sg::core::PRIMITIVE_QUADS ) || ( pt == dp::sg::core::PRIMITIVE_TRIANGLES ) )
    {
      return( true );
    }
  }
  return( false );
}

bool containsStrippedPrimitives( std::vector<dp::sg::core::ObjectWeakPtr> const & vp )
{
  for ( size_t i=0 ; i<vp.size() ; i++ )
  {
    DP_ASSERT( dynamic_cast<dp::sg::core::PrimitiveWeakPtr>(vp[i]) );
    dp::sg::core::PrimitiveType pt = static_cast<dp::sg::core::PrimitiveWeakPtr>(vp[i])->getPrimitiveType();
    if (    ( pt == dp::sg::core::PRIMITIVE_LINE_STRIP )
        ||  ( pt == dp::sg::core::PRIMITIVE_LINE_LOOP )
        ||  ( pt == dp::sg::core::PRIMITIVE_TRIANGLE_STRIP )
        ||  ( pt == dp::sg::core::PRIMITIVE_TRIANGLE_FAN )
        ||  ( pt == dp::sg::core::PRIMITIVE_QUAD_STRIP )
        ||  ( pt == dp::sg::core::PRIMITIVE_TRIANGLE_STRIP_ADJACENCY )
        ||  ( pt == dp::sg::core::PRIMITIVE_LINE_STRIP_ADJACENCY ) )
    {
      return( true );
    }
  }
  return( false );
}

bool findOrCreateCameras( dp::sg::ui::ViewStateSharedPtr const& viewState, std::vector<dp::sg::core::CameraSharedPtr> & cameras )
{
  cameras.clear();

  // if there was a viewstate with a camera, always set it as first cam
  dp::sg::core::CameraSharedPtr vsCam;
  unsigned int c = 0;
  if ( viewState )
  {
    if ( viewState->getCamera() )
    {
      vsCam = viewState->getCamera();
      nameCamera( vsCam, "SceneCamera", c );
      cameras.push_back( vsCam );
    }
  }

  dp::sg::core::SceneSharedPtr const& scene = viewState->getScene();
  for ( dp::sg::core::Scene::CameraIterator iter = scene->beginCameras() ; iter != scene->endCameras() ; ++iter )
  {
    // If there was no camera at the ViewState vsCam will be nullptr.
    // Add all cameras to be able to add headlights when wanted.
    if ( vsCam != (*iter) )
    {
      nameCamera( *iter, "SceneCamera", c );
      cameras.push_back( *iter );
    }
  }

  // make SVCameraN start over at zero
  c = 0;

  // make sure we have at least 4
  while ( cameras.size() < 4 )
  {
    dp::sg::core::PerspectiveCameraSharedPtr perspectiveCamera = dp::sg::core::PerspectiveCamera::create();
    {
      nameCamera( perspectiveCamera, "SVCamera", c );

      // zoom cameras if we had to create them
      if ( scene->getRootNode() )
      {
        const dp::math::Sphere3f & bs = scene->getRootNode()->getBoundingSphere();
        if ( isPositive( bs ) )
        {
          perspectiveCamera->zoom( bs, dp::math::PI_QUARTER );
        }
      }
    }

    cameras.push_back( perspectiveCamera );
    scene->addCamera( perspectiveCamera );
  }

  // always return true now because we have probably renamed some cameras as well
  // resetting item models is not that big of a deal
  return true;
}

void nameCamera( const dp::sg::core::CameraSharedPtr & camera, const std::string & baseName, unsigned int & index )
{
  if ( camera->getName().empty() )
  {
    std::ostringstream ss;
    ss << baseName << std::setw(2) << std::setfill('0') << index;
    index++;
    camera->setName( ss.str() );
  }
}

std::string samplerTypeToString( unsigned int samplerType )
{
  DP_ASSERT( samplerType & dp::fx::PT_SAMPLER_TYPE_MASK );
  switch( samplerType & dp::fx::PT_SAMPLER_TYPE_MASK )
  {
    case dp::fx::PT_SAMPLER_1D :                    return( "1D Texture" );
    case dp::fx::PT_SAMPLER_2D :                    return( "2D Texture" );
    case dp::fx::PT_SAMPLER_3D :                    return( "3D Texture" );
    case dp::fx::PT_SAMPLER_CUBE :                  return( "Cube Texture" );
    case dp::fx::PT_SAMPLER_2D_RECT :               return( "2D Rect Texture" );
    case dp::fx::PT_SAMPLER_1D_ARRAY :              return( "1D Texture Array" );
    case dp::fx::PT_SAMPLER_2D_ARRAY :              return( "2D Texture Array" );
    case dp::fx::PT_SAMPLER_BUFFER :                return( "Buffer Texture" );
    case dp::fx::PT_SAMPLER_2D_MULTI_SAMPLE :       return( "2D Multi-Sample Texture" );
    case dp::fx::PT_SAMPLER_2D_MULTI_SAMPLE_ARRAY : return( "2D Multi-Sample Texture Array" );
    case dp::fx::PT_SAMPLER_CUBE_ARRAY :            return( "Cube Texture Array" );
    case dp::fx::PT_SAMPLER_1D_SHADOW :             return( "1D Shadow Texture" );
    case dp::fx::PT_SAMPLER_2D_SHADOW :             return( "2D Shadow Texture" );
    case dp::fx::PT_SAMPLER_2D_RECT_SHADOW :        return( "2D Rect Shadow Texture" );
    case dp::fx::PT_SAMPLER_1D_ARRAY_SHADOW :       return( "1D Shadow Texture Array" );
    case dp::fx::PT_SAMPLER_2D_ARRAY_SHADOW :       return( "2D Shadow Texture Array" );
    case dp::fx::PT_SAMPLER_CUBE_SHADOW :           return( "Cube Shadow Texture" );
    case dp::fx::PT_SAMPLER_CUBE_ARRAY_SHADOW :     return( "Cube Shadow Texture Array" );
    default:
      DP_ASSERT( !"unknown sampler type encountered" );
      return( "" );
  }
}

void setTraversalMasks( dp::sg::core::SceneSharedPtr const & scene, unsigned int mask )
{
  if ( scene )
  {
    dp::sg::algorithm::SearchTraverser searchTraverser;

    searchTraverser.setClassName( "class dp::sg::core::GeoNode" );
    searchTraverser.setBaseClassSearch( true );
    searchTraverser.apply( scene );  

    const std::vector<dp::sg::core::ObjectWeakPtr> & searchResults = searchTraverser.getResults();
    for ( std::vector<dp::sg::core::ObjectWeakPtr>::const_iterator it = searchResults.begin() ; it != searchResults.end() ; ++it )
    {
      dp::util::weakPtr_cast<dp::sg::core::GeoNode>(*it)->setTraversalMask( mask );
    }
  }
}

bool textureTypeIsCompatible( unsigned int id, dp::sg::core::TextureTarget tt )
{
  DP_ASSERT( id & dp::fx::PT_SAMPLER_TYPE_MASK );
  return(   ( ( id & dp::fx::PT_SAMPLER_1D ) && ( tt == dp::sg::core::TT_TEXTURE_1D ) )
        ||  ( ( id & dp::fx::PT_SAMPLER_2D ) && ( tt == dp::sg::core::TT_TEXTURE_2D ) )
        ||  ( ( id & dp::fx::PT_SAMPLER_3D ) && ( tt == dp::sg::core::TT_TEXTURE_3D ) )
        ||  ( ( id & dp::fx::PT_SAMPLER_CUBE ) && ( tt == dp::sg::core::TT_TEXTURE_CUBE ) )
        ||  ( ( id & dp::fx::PT_SAMPLER_1D_ARRAY ) && ( tt == dp::sg::core::TT_TEXTURE_1D_ARRAY ) )
        ||  ( ( id & dp::fx::PT_SAMPLER_2D_ARRAY ) && ( tt == dp::sg::core::TT_TEXTURE_2D_ARRAY ) )
        ||  ( ( id & dp::fx::PT_SAMPLER_2D_RECT ) && ( tt == dp::sg::core::TT_TEXTURE_RECTANGLE ) )
        ||  ( ( id & dp::fx::PT_SAMPLER_CUBE_ARRAY ) && ( tt == dp::sg::core::TT_TEXTURE_CUBE_ARRAY ) )
        ||  ( ( id & dp::fx::PT_SAMPLER_BUFFER ) && ( tt == dp::sg::core::TT_TEXTURE_BUFFER ) ) );
}

std::string textureTargetToString( dp::sg::core::TextureTarget tt )
{
  switch( tt )
  {
    case dp::sg::core::TT_TEXTURE_1D :          return( "1D Texture" );
    case dp::sg::core::TT_TEXTURE_2D :          return( "2D Texture" );
    case dp::sg::core::TT_TEXTURE_3D :          return( "3D Texture" );
    case dp::sg::core::TT_TEXTURE_CUBE :        return( "Cube Texture" );
    case dp::sg::core::TT_TEXTURE_1D_ARRAY :    return( "1D Texture Array" );
    case dp::sg::core::TT_TEXTURE_2D_ARRAY :    return( "2D Texture Array" );
    case dp::sg::core::TT_TEXTURE_RECTANGLE :   return( "Rectangle Texture" );
    case dp::sg::core::TT_TEXTURE_CUBE_ARRAY :  return( "Cube Texture Array" );
    case dp::sg::core::TT_TEXTURE_BUFFER :      return( "Buffer Texture" );
    default:
      DP_ASSERT( !"unknown dp::sg::core::TextureTarget encountered" );
      return( "" );
  }
}
