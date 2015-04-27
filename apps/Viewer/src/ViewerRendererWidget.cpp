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


#include "CommandAddItem.h"
#include "SceneTreeBrowser.h"
#include "Viewer.h"
#include "ViewerCommand.h"
#include "MainWindow.h"
#include "Log.h"

#include <QList>
#include <QMimeData>
#include <dp/math/Vecnt.h>
#include <dp/math/Quatt.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/algorithm/SearchTraverser.h>
#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/sg/renderer/rix/gl/SceneRenderer.h>
#include <dp/sg/renderer/rix/gl/TransparencyManagerOITClosestArray.h>
#include <dp/sg/ui/RendererOptions.h>
#include <dp/sg/io/IO.h>
#include <dp/fx/EffectLibrary.h>
#include <dp/sg/ui/manipulator/CylindricalCameraManipulatorHIDSync.h>
#include <dp/sg/ui/manipulator/TrackballCameraManipulatorHIDSync.h>
#include <dp/sg/ui/manipulator/WalkCameraManipulatorHIDSync.h>
#include <dp/sg/ui/manipulator/FlightCameraManipulatorHIDSync.h>

#include <iostream>
#include <iomanip>

#include <ViewerRendererWidget.h>

using namespace dp::sg::core;
using namespace dp::math;
using namespace std;


dp::gl::RenderContextFormat ViewerRendererWidget::s_glFormat;
unsigned int ViewerRendererWidget::s_cameraCount = 0;
unsigned int ViewerRendererWidget::s_dlightCount = 0;
unsigned int ViewerRendererWidget::s_plightCount = 0;
unsigned int ViewerRendererWidget::s_slightCount = 0;

ViewerRendererWidget::ViewerRendererWidget( QWidget *parent, SceniXQGLWidget *shareWidget )
  : SceniXQGLSceneRendererWidget( parent, s_glFormat, shareWidget )
  , m_currentManipulator(0)
  , m_manipulatorType(MANIPULATOR_NONE)
  , m_rendererType( RENDERER_NONE )
  , m_restartAccumulation(true)  // Start from scratch on initial frame.
  , m_highlightedObject(0)
  , m_canManipulateObject(false)
  , m_sRGBDesired(false)
  , m_isViewport(true)
  , m_frameCount(0)
  , m_oitDepth( 8 )
{
  // set to allow D&D  
  // MMM - need finer control here because we would accept color drops as well
  setAcceptDrops( true );
  setFocusPolicy( Qt::StrongFocus ); // TAB or click for keyboard focus

  m_sceneRendererPipeline = SceneRendererPipeline::create(); // Allow the setSceneRenderer() API to work!

  // use new manipulator
  setManipulatorType( MANIPULATOR_TRACKBALL );

  // create our default viewstate
  m_defaultViewState = dp::sg::ui::ViewState::create();
  setViewState( m_defaultViewState );
}

void ViewerRendererWidget::present()
{
  restartUpdate();
}

void ViewerRendererWidget::clear()
{
  s_cameraCount = 0;
  s_dlightCount = 0;
  s_plightCount = 0;
  s_slightCount = 0;
}

void
ViewerRendererWidget::setIsViewport( bool isViewport )
{
  m_isViewport = isViewport;
}

void 
ViewerRendererWidget::setRendererType( RendererType type )
{
  if( type != m_rendererType )
  {
    clearContextMenu();

    m_rendererType = type;

    addDefaultActions();

    // set up renderer
    switch( type )
    {
      case RENDERER_NONE:
        setSceneRenderer( dp::sg::ui::SceneRendererSharedPtr::null );
        break;

      case RENDERER_RASTERIZE_XBAR:
      {
        addRasterizeActions();
        Viewer * viewer = GetApp();
        dp::sg::renderer::rix::gl::SceneRendererSharedPtr ssrgl =
          dp::sg::renderer::rix::gl::SceneRenderer::create( viewer->getRenderEngine().c_str(), 
                                                            viewer->getShaderManagerType(),
                                                            viewer->getCullingMode(),
                                                            viewer->getTransparencyMode() );
        ssrgl->setEnvironmentSampler( GetApp()->getEnvironmentSampler() );
        ssrgl->setDepthPass( GetApp()->getPreferences()->getDepthPass() );
        setSceneRenderer( ssrgl );
      }
      break;

      default:
        DP_ASSERT( !"Invalid RendererType" );
        break;
    }
    
    present();
  }
}

ViewerRendererWidget::RendererType 
ViewerRendererWidget::getRendererType() const
{
  return m_rendererType;
}

void
ViewerRendererWidget::clearContextMenu()
{
  for( size_t i = 0; i < m_contextMenuEntries.size(); i ++ )
  {
    delete m_contextMenuEntries[i];
  }

  m_contextMenuEntries.clear();
}

void ViewerRendererWidget::addRasterizeActions()
{
}

void
ViewerRendererWidget::addDefaultActions()
{
  QAction * act = new QAction(this);
  act->setSeparator(true);
  act->setText("Renderer");

  m_contextMenuEntries.push_back( act );

  // Render Engine submenu
  {
    m_renderEngineActions.clear();
    QMenu * renderEngineMenu = GetApp()->getMainWindow()->getMenu( MainWindow::MID_RENDER_ENGINE );
    QList<QAction *> actions = renderEngineMenu->actions();
    for ( QList<QAction*>::const_iterator it = actions.begin() ; it != actions.end() ; ++it )
    {
      m_renderEngineActions.push_back( *it );
    }
    m_contextMenuEntries.push_back( renderEngineMenu );
  }
  
  // The Viewports get all actions, the preview widget only the render engine toggle.
  if ( !m_isViewport )
  {
    return;
  }

  act = new QAction(this);
  act->setSeparator(true);
  act->setText("View Manipulation");

  m_contextMenuEntries.push_back( act );

  act = new QAction(this);
  act->setSeparator(true);
  act->setText("Light Sources");

  m_contextMenuEntries.push_back( act );

  m_contextMenuEntries.push_back( GetApp()->getMainWindow()->getMenu( MainWindow::MID_ADD_HEADLIGHT ) );
  m_contextMenuEntries.push_back( GetApp()->getMainWindow()->getMenu( MainWindow::MID_ADD_LIGHT_SOURCE ) );

  act = new QAction(this);
  act->setSeparator(true);
  act->setText("Cameras");

  m_contextMenuEntries.push_back( act );

  act = new QAction( this );
  act->setObjectName(QString::fromUtf8("actionAddCamera"));
  act->setText( "Add Camera" );
  connect( act, SIGNAL(triggered()), this, SLOT(addCamera()) );
  m_contextMenuEntries.push_back( act );

  act = new QAction(this);
  act->setSeparator(true);
  act->setText("Manipulate");

  m_contextMenuEntries.push_back( act );

  act = new QAction( this );
  act->setObjectName(QString::fromUtf8("actionMoveSelectedObject"));
  act->setText( "Move Selected Object Here" );
  connect( act, SIGNAL(triggered()), this, SLOT(moveSelectedObject()) );
  m_contextMenuEntries.push_back( act );

  m_contextMenuEntries.push_back( GetApp()->getMainWindow()->getMenu( MainWindow::MID_CULLING ) );
  m_contextMenuEntries.push_back( GetApp()->getMainWindow()->getMenu( MainWindow::MID_VIEWPORT_FORMAT ) );

  connect( GetApp()->getPreferences(), SIGNAL(environmentEnabledChanged()), this, SLOT(setEnvironmentEnabledChanged()) );
  connect( GetApp(), SIGNAL(environmentChanged()), this, SLOT(updateEnvironment()) );
}

void
ViewerRendererWidget::addRaytraceActions()
{
  // what should we have here??
}

ViewerRendererWidget::~ViewerRendererWidget()
{ 
  // make render context current before destruction
  getRenderContext()->makeCurrent();

  // this will delete the current manipulator
  setManipulatorType( MANIPULATOR_NONE );
}

void
ViewerRendererWidget::setCamera( const CameraSharedPtr & cameraSP )
{
  DP_ASSERT( cameraSP );
  DP_ASSERT( m_viewState );

  // set the camera, and ensure that focus and target distance are set initially
  m_viewState->setCamera( cameraSP );

  if ( m_viewState->getScene()->getRootNode() )
  {
    const Sphere3f & bs = m_viewState->getScene()->getRootNode()->getBoundingSphere();
    float dist = dp::math::distance( cameraSP->getPosition(), bs.getCenter() );
    dist = std::max<float>( dist, 0.1f );

    m_viewState->setTargetDistance( dist );
    // MMM - should we do this for cameras that came from the scene?
    //       if so, we could modify the effect the user had in mind..
    if( cameraSP->getUserData() )
    {
      cameraSP->setFocusDistance( dist );
    }
  }
}

void ViewerRendererWidget::setScene( dp::sg::core::SceneSharedPtr const & scene )
{
  dehighlightAll();

  DP_ASSERT( m_viewState );

  if ( scene )
  {
    m_viewState->setSceneTree( GetApp()->getViewState()->getSceneTree() );

    // reset manipulator in case the manip corrects for scene size and whatnot
    ManipulatorType mt = getManipulatorType();
    setManipulatorType( MANIPULATOR_NONE );
    setManipulatorType( mt );
  }
  else
  {
    // reset scene and camera here
    m_viewState->setSceneTree( dp::sg::xbar::SceneTreeSharedPtr::null );
    m_viewState->setCamera( CameraSharedPtr::null );
  }

  // set scene data
  m_scene = scene;
}

void ViewerRendererWidget::initializeGL()
{
  SceniXQGLSceneRendererWidget::initializeGL();
  glewInit();

  // Now that all OpenGL resources are initialized, configure the pipeline renderer.
  bool success = m_sceneRendererPipeline->init(getRenderContext(), getRenderTarget());
  DP_ASSERT(success);

  // Call the base class, because the SceneRendererPipeline has overridden the setSceneRenderer() function.
  SceniXQGLSceneRendererWidget::setSceneRenderer(m_sceneRendererPipeline);
}

void ViewerRendererWidget::keyPressEvent ( QKeyEvent * keyEvent )
{
  SceniXQGLSceneRendererWidget::keyPressEvent(keyEvent);

  if (!keyEvent->isAccepted() )
  {
    switch( keyEvent->key() )
    {
      case Qt::Key_Space :
        {
          // allow space to toggle continuous rendering
          QAction * action = GetApp()->getMainWindow()->getContinuousRedrawAction();
          DP_ASSERT( action );
          action->setChecked( !action->isChecked() );
        }
        break;
      case Qt::Key_M :
        GetApp()->setTonemapperEnabled( !GetApp()->isTonemapperEnabled() );
        break;
      case Qt::Key_O :
        {
          unsigned int depth = getOITDepth();
          if ( keyEvent->modifiers() & Qt::ShiftModifier )
          {
            setOITDepth( depth + 1 );
          }
          else if ( 1 < depth )
          {
            setOITDepth( depth - 1 );
          }
        }
        break;
      case Qt::Key_T :
        switch( getTransparencyMode() )
        {
          case dp::sg::renderer::rix::gl::TM_NONE :
            setTransparencyMode( dp::sg::renderer::rix::gl::TM_SORTED_BLENDED );
            break;
          case dp::sg::renderer::rix::gl::TM_SORTED_BLENDED :
            setTransparencyMode( dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_CLOSEST_LIST );
            break;
          case dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_CLOSEST_LIST :
            setTransparencyMode( dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_ALL );
            break;
          case dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_ALL :
            setTransparencyMode( dp::sg::renderer::rix::gl::TM_NONE );
            break;
          default :
            DP_ASSERT( !"ViewerRendererWidget::keyPressEvent: unknown transparency mode encountered!" );
            break;
        }
        break;
    }
  }
}

void ViewerRendererWidget::keyReleaseEvent ( QKeyEvent * keyEvent )
{
  SceniXQGLSceneRendererWidget::keyReleaseEvent(keyEvent);
}

void ViewerRendererWidget::mouseDoubleClickEvent ( QMouseEvent * mouseEvent )
{
  SceniXQGLSceneRendererWidget::mouseDoubleClickEvent(mouseEvent);

  if (!mouseEvent->isAccepted() )
  {
    present();
    mouseEvent->accept();
  }
}

void ViewerRendererWidget::mouseMoveEvent ( QMouseEvent * mouseEvent )
{
  SceniXQGLSceneRendererWidget::mouseMoveEvent(mouseEvent);
}

void ViewerRendererWidget::mousePressEvent ( QMouseEvent * mouseEvent )
{
  SceniXQGLSceneRendererWidget::mousePressEvent(mouseEvent);

  if (!mouseEvent->isAccepted() )
  {
    // Exactly Shift+LMB to prevent inadvertent selections during vertigo (Shift+Control+LMB+MMB)
    if( mouseEvent->button()    == Qt::LeftButton && 
        mouseEvent->modifiers() == Qt::ShiftModifier )
    {
      selectObject( mouseEvent );
    }

    mouseEvent->accept();
  }
}

void ViewerRendererWidget::mouseReleaseEvent ( QMouseEvent * mouseEvent )
{
  SceniXQGLSceneRendererWidget::mouseReleaseEvent(mouseEvent);
}

void ViewerRendererWidget::wheelEvent ( QWheelEvent * wheelEvent )
{
  SceniXQGLSceneRendererWidget::wheelEvent(wheelEvent);
}

bool ViewerRendererWidget::isOptionSupported( const std::string & option )
{
  return( m_viewState->getRendererOptions()->hasProperty( option ) );
}

QAction * 
ViewerRendererWidget::findAction( const QString & name )
{
  for( int i = 0; i < actions().size(); i ++ )
  {
    if( actions().at(i)->objectName() == name )
    {
      return actions().at(i);
    }
  }

  return 0;
}

void ViewerRendererWidget::highlightObject( const dp::sg::core::NodeSharedPtr & which )
{
  if( !which )
  {
    return;
  }

  // select everything from here on down
  dp::sg::algorithm::SearchTraverser searchTraverser;
  searchTraverser.setClassName("class dp::sg::core::GeoNode");
  searchTraverser.apply( which );

  const vector<ObjectWeakPtr> & searchResults = searchTraverser.getResults();
  vector<ObjectWeakPtr>::const_iterator it;
  for(it=searchResults.begin(); it!=searchResults.end(); it++)
  {
    highlightGeoNode( (*it)->getSharedPtr<GeoNode>() );
  }
}

void ViewerRendererWidget::highlightGeoNode( const dp::sg::core::GeoNodeSharedPtr & geoNode )
{
  // note, if a primitive is highlighted, then there may be multiple primitives
  m_highlightedObject = geoNode;
  m_selectedGeoNodes.insert( geoNode );

  // set to selected 
  geoNode->setTraversalMask( 0x02 );

  enableHighlighting( true );
}

void ViewerRendererWidget::dehighlightAll()
{
  m_highlightedObject.reset();

  std::set< GeoNodeSharedPtr >::iterator iter = m_selectedGeoNodes.begin();
  while( iter != m_selectedGeoNodes.end() )
  {
    // set to default value
    (*iter)->setTraversalMask( 0x01 );
    ++iter;
  }

  m_selectedGeoNodes.clear();

  enableHighlighting( false );
}

void ViewerRendererWidget::dropEvent( QDropEvent * event )
{
  if( event->source() == this )
  {
    // ignore our own
    return;
  }

  const QMimeData * mimeData = event->mimeData();
  if ( mimeData->hasText() )
  {
    std::string effectName = mimeData->text().toStdString();
    const dp::fx::EffectSpecSharedPtr & materialEffectSpec = dp::fx::EffectLibrary::instance()->getEffectSpec( effectName );
    if ( materialEffectSpec )
    {
      ExecuteCommand( new CommandReplaceEffect( *m_selectedGeoNodes.begin(), GetApp()->getEffectData( effectName ) ) );
    }
  }
  else if ( mimeData->hasFormat( "EffectData" ) )
  {
    // EffectData delivers the GeoNode, as both material and geometry effect needs to be copied
    dp::sg::core::GeoNodeSharedPtr geoNode( (*(reinterpret_cast<dp::sg::core::GeoNodeWeakPtr*>( mimeData->data( "EffectData" ).data() )))->getSharedPtr<GeoNode>() );

    DP_ASSERT( m_selectedGeoNodes.size() == 1 );
    DP_ASSERT( m_selectedGeoNodes.begin()->isPtrTo<GeoNode>() );

    if ( *m_selectedGeoNodes.begin() != geoNode )
    {
      ExecuteCommand( new CommandReplaceEffect( *m_selectedGeoNodes.begin(), geoNode->getMaterialEffect() ) );

      dehighlightAll();
      present();
    }
  }
}

void ViewerRendererWidget::dragEnterEvent( QDragEnterEvent * event )
{
  const QMimeData * mimeData = event->mimeData();
  if ( mimeData->hasText() )
  {
    std::string effectName = mimeData->text().toStdString();
    const dp::fx::EffectSpecSharedPtr & effectSpec = dp::fx::EffectLibrary::instance()->getEffectSpec( effectName );
    if ( effectSpec )
    {
      event->accept();
    }
  }
  else if ( mimeData->hasFormat( "EffectData" ) )
  {
    event->accept();
  }
}

void ViewerRendererWidget::dragLeaveEvent( QDragLeaveEvent * event )
{
  // make sure we deselect everything if the drag leaves the window
  dehighlightAll();
  present();
}

void ViewerRendererWidget::dragMoveEvent( QDragMoveEvent * event )
{
  QMouseEvent e( QEvent::MouseMove, event->pos(), Qt::NoButton, Qt::NoButton, Qt::NoModifier );

  event->setAccepted( selectObject( &e ) );
}

bool ViewerRendererWidget::selectObject( QMouseEvent * mouseEvent )
{
  dp::sg::algorithm::Intersection result;
  bool intersect = intersectObject( m_viewState->getScene()->getRootNode()
                                  , mouseEvent->pos().x(), mouseEvent->pos().y(), result );
  if ( intersect )
  {
    emit objectSelected( result.getPath() );
  }
  return( intersect );
}

void ViewerRendererWidget::currentItemChanged( dp::sg::core::ObjectSharedPtr current, dp::sg::core::ObjectSharedPtr previous )
{
  if ( previous.isPtrTo<GeoNode>() )
  {
    // set to default value
    m_selectedGeoNodes.clear();
    previous.staticCast<GeoNode>()->setTraversalMask( 0x01 );
    enableHighlighting( false );
  }
  if ( current.isPtrTo<GeoNode>() )
  {
    // set to selected 
    m_selectedGeoNodes.insert( current.staticCast<GeoNode>() );
    current.staticCast<GeoNode>()->setTraversalMask( 0x02 );
    enableHighlighting( true );
  }
  m_highlightedObject = current;
  present();
}

bool 
ViewerRendererWidget::intersectObject( const dp::sg::core::NodeSharedPtr & baseSearch,
                                       unsigned int screenX, unsigned int screenY,
                                       dp::sg::algorithm::Intersection & result )
{
  // requires a camera attached to the ViewState
  if ( m_viewState->getCamera().isPtrTo<FrustumCamera>() )
  {
    FrustumCameraSharedPtr const& pCam = m_viewState->getCamera().staticCast<FrustumCamera>();

    // calculate ray origin and direction from the input point
    Vec3f rayOrigin;
    Vec3f rayDir;

    int y = height() - 1 - screenY; // adjust to bottom left origin
    pCam->getPickRay(screenX, y, width(), height(), rayOrigin, rayDir);

    // run the intersect traverser for intersections with the given ray
    dp::sg::algorithm::RayIntersectTraverser rayIntersectTraverser;

    rayIntersectTraverser.setRay(rayOrigin, rayDir);
    rayIntersectTraverser.setViewState( m_viewState );
    rayIntersectTraverser.setViewportSize( width(), height() );
    rayIntersectTraverser.apply( baseSearch );

    if (rayIntersectTraverser.getNumberOfIntersections() > 0)
    {
      result = rayIntersectTraverser.getNearest();
      return true;
    }
  }

  return false;
}

bool
ViewerRendererWidget::menuConstraintsSatisfied( QObject * object )
{
  // check constraints of all context menu options here..
  
  if( object->objectName() == "actionMoveSelectedObject" )
  {
    // Not all objects can be manipulated
    return m_highlightedObject && m_canManipulateObject;
  }

  return true;
}

bool ViewerRendererWidget::actionCheck( QAction * action )
{
  if ( action->objectName() == "actionCulling" )
  {
    return( m_sceneRendererPipeline->isCullingEnabled() );
  }
  DP_ASSERT( false );
  return( true );
}

void 
ViewerRendererWidget::contextMenuEvent( QContextMenuEvent * event )
{
  // ignore "reason" for the moment
  QMenu menu( "Render Controls", this );

  for( size_t i = 0; i < m_contextMenuEntries.size(); i ++ )
  {
    if( menuConstraintsSatisfied( m_contextMenuEntries[i] ) )
    {
      if ( dynamic_cast<QAction*>(m_contextMenuEntries[i]) )
      {
        QAction * action = static_cast<QAction*>(m_contextMenuEntries[i]);
        if ( action->isCheckable() )
        {
          action->setChecked( actionCheck( action ) );
        }
        menu.addAction( static_cast<QAction*>(m_contextMenuEntries[i]) );
      }
      else
      {
        DP_ASSERT( dynamic_cast<QMenu*>(m_contextMenuEntries[i]) );
        menu.addMenu( static_cast<QMenu*>(m_contextMenuEntries[i]) );
      }
    }
  }

  QAction * action = menu.exec( event->globalPos() );
  // renderer engine actions need to be handled _after_ menu.exec returned, as they clear the menu!
  if ( std::find( m_renderEngineActions.begin(), m_renderEngineActions.end(), action ) != m_renderEngineActions.end() )
  {
    triggeredRenderEngineMenu( action );
  }
}


void
ViewerRendererWidget::resizeGL( int width, int height )
{
  m_restartAccumulation = true;
  SceniXQGLSceneRendererWidget::resizeGL(width, height);
}

void
ViewerRendererWidget::paintGL()
{
  if (m_restartAccumulation)
  {
    getSceneRenderer()->restartAccumulation();
    m_restartAccumulation = false;
  }

  // leave me here in case we ever want to override this
  SceniXQGLSceneRendererWidget::paintGL();

  m_frameCount++;
}

bool ViewerRendererWidget::screenShot( const QString & filename )
{
  dp::util::imageToFile( getRenderTarget()->getImage(), filename.toStdString() );
  return false;
}

void moveDirectedLightToCamera( ParameterGroupDataSharedPtr const& pgd, PerspectiveCameraSharedPtr const& pcam )
{
  DP_ASSERT( pgd && pcam );

  DP_VERIFY( pgd->setParameter( "direction", pcam->getDirection() ) );
}

void movePointLightToCamera( ParameterGroupDataSharedPtr const& pgd, PerspectiveCameraSharedPtr const& pcam )
{
  DP_ASSERT( pgd && pcam );

  DP_VERIFY( pgd->setParameter( "position", pcam->getPosition() ) );
}

void moveSpotLightToCamera( ParameterGroupDataSharedPtr const& pgd, PerspectiveCameraSharedPtr const& pcam )
{
  DP_ASSERT( pgd && pcam );
  DP_VERIFY( pgd->setParameter( "position", pcam->getPosition() ) );
  DP_VERIFY( pgd->setParameter( "direction", pcam->getDirection() ) );
}

inline GroupSharedPtr makeRootGroup( SceneSharedPtr const& scene )
{
  GroupSharedPtr group;

  if ( scene->getRootNode().isPtrTo<Group>() )
  {
    group = scene->getRootNode().staticCast<Group>();
  }
  else
  {
    group = Group::create();
    group->addChild( scene->getRootNode() );
    scene->setRootNode( group );
  }

  return group;
}

static void nameLight( const LightSourceSharedPtr lsSP, const std::string & baseName, unsigned int index )
{
  std::ostringstream ss;
  ss << baseName << setw(2) << setfill('0') << index;

  lsSP->setName( ss.str() );
}

void ViewerRendererWidget::addDirectedLight()
{
  LightSourceSharedPtr directedLight = createStandardDirectedLight( Vec3f( 0.0f, 0.0f, -1.0f ), Vec3f( 1.0f, 1.0f, 1.0f ) );
  nameLight( directedLight, "SVDirectedLight", s_dlightCount++ );

  const ParameterGroupDataSharedPtr & pgd = directedLight->getLightEffect()->findParameterGroupData( std::string( "standardDirectedLightParameters" ) );
  DP_ASSERT( pgd );
  moveDirectedLightToCamera( pgd, m_viewState->getCamera().staticCast<PerspectiveCamera>() );
  GroupSharedPtr group = makeRootGroup( m_viewState->getScene() );

  // NOTE: Nothing can be read/write locked during executecommand
  ExecuteCommand( new CommandAddObject( group, directedLight ) );
}

void ViewerRendererWidget::addPointLight()
{
  LightSourceSharedPtr pointLight = createStandardPointLight( Vec3f( 0.0f, 0.0f, 0.0f ), Vec3f( 1.0f, 1.0f, 1.0f ) );
  nameLight( pointLight, "SVPointLight", s_plightCount++ );

  const ParameterGroupDataSharedPtr & pgd = pointLight->getLightEffect()->findParameterGroupData( std::string( "standardPointLightParameters" ) );
  DP_ASSERT( pgd );
  movePointLightToCamera( pgd, m_viewState->getCamera().staticCast<PerspectiveCamera>() );
  GroupSharedPtr group = makeRootGroup( m_viewState->getScene() );

  // NOTE: Nothing can be read/write locked during executecommand
  ExecuteCommand( new CommandAddObject( group, pointLight ) );
}

void ViewerRendererWidget::addSpotLight()
{
  LightSourceSharedPtr spotLight = createStandardSpotLight( Vec3f( 0.0f, 0.0f, 0.0f ), Vec3f( 0.0f, 0.0f, -1.0f ), Vec3f( 1.0f, 1.0f, 1.0f ) );
  nameLight( spotLight, "SVSpotLight", s_slightCount++ );

  const ParameterGroupDataSharedPtr & pgd = spotLight->getLightEffect()->findParameterGroupData( std::string( "standardSpotLightParameters" ) );
  DP_ASSERT( pgd );
  moveSpotLightToCamera( pgd, m_viewState->getCamera().staticCast<PerspectiveCamera>() );
  GroupSharedPtr group = makeRootGroup( m_viewState->getScene() );
  
  // NOTE: Nothing can be read/write locked during executecommand
  ExecuteCommand( new CommandAddObject( group, spotLight ) );
}

static void nameCamera( CameraWeakPtr camWP, const std::string & baseName, unsigned int index )
{
  std::ostringstream ss;
  ss << baseName << setw(2) << setfill('0') << index;

  camWP->setName( ss.str() );
}

void ViewerRendererWidget::addCamera()
{
  PerspectiveCameraSharedPtr const& pcamh = m_viewState->getCamera().staticCast<PerspectiveCamera>();
  SceneSharedPtr const& ssh = m_viewState->getScene();

  if( pcamh )
  {
    PerspectiveCameraSharedPtr newPcamh = pcamh.clone();
    nameCamera( newPcamh.getWeakPtr(), "SVPerspectiveCamera", s_cameraCount++ );

    // so this can be used for camera cycling
    newPcamh->setUserData( nullptr );

    // NOTE: Nothing can be read/write locked during executecommand
    ExecuteCommand( new CommandAddObject( ssh, CameraSharedPtr( newPcamh ) ) );
  }
  else
  {
    LogWarning("Scene's Camera is not a PerspectiveCamera??\n");
  }
}

void ViewerRendererWidget::moveSelectedObject()
{
  unsigned int objectCode = m_highlightedObject->getObjectCode();

  bool modified = false;

  PerspectiveCameraSharedPtr const& pcam = m_viewState->getCamera().staticCast<PerspectiveCamera>();

  // ensure we have an object highlighted!
  DP_ASSERT( m_highlightedObject );

  switch( objectCode )
  {
    case OC_LIGHT_SOURCE:
    {
      EffectDataSharedPtr const& le = m_highlightedObject.staticCast<LightSource>()->getLightEffect();
      const dp::fx::EffectSpecSharedPtr & es = le->getEffectSpec();
      for ( dp::fx::EffectSpec::iterator it = es->beginParameterGroupSpecs() ; it != es->endParameterGroupSpecs() ; ++it )
      {
        const dp::sg::core::ParameterGroupDataSharedPtr & parameterGroupData = le->getParameterGroupData( it );
        if ( parameterGroupData )
        {
          string name = (*it)->getName();
          if ( ( name == "standardDirectedLightParameters" )
            || ( name == "standardPointLightParameters" )
            || ( name == "standardSpotLightParameters" ) )
          {
            if ( name == "standardDirectedLightParameters" )
            {
              moveDirectedLightToCamera( parameterGroupData, pcam );
            }
            else if ( name == "standardPointLightParameters" )
            {
              movePointLightToCamera( parameterGroupData, pcam );
            }
            else
            {
              moveSpotLightToCamera( parameterGroupData, pcam );
            }
            break;
          }
        }
      }
      modified = true;
    }
    break;

    case OC_PERSPECTIVECAMERA:
    {
      PerspectiveCameraSharedPtr const& pc = m_highlightedObject.staticCast<PerspectiveCamera>();
      pc->setPosition( pcam->getPosition() );
      pc->setOrientation( pcam->getOrientation() );
      modified = true;
    }
    break;
  }

  if( modified )
  {
    // send this signal so all windows will update
    // NOTE: nothing should be read/write locked during this call..
    GetApp()->emitViewChanged();
  }
}

void ViewerRendererWidget::restartUpdate()
{
  m_restartAccumulation = true;
  update();
}

void 
ViewerRendererWidget::enableHighlighting( bool onOff )
{
  m_sceneRendererPipeline->enableHighlighting( onOff );
}

dp::sg::ui::SceneRendererSharedPtr ViewerRendererWidget::getSceneRenderer() const
{
  if( m_sceneRendererPipeline )
  {
    return m_sceneRendererPipeline->getSceneRenderer();
  }
  else
  {
    return dp::sg::ui::SceneRendererSharedPtr(0);
  }
}

void ViewerRendererWidget::setSceneRenderer( const dp::sg::ui::SceneRendererSharedPtr & ssr )
{
  DP_ASSERT( m_sceneRendererPipeline );
  m_sceneRendererPipeline->setSceneRenderer( ssr );

  // Unconditionally reset the RendererOptions. (SceneRendererGL2 has different ones than SceneRendererRT.)
  m_viewState->setRendererOptions( dp::sg::ui::RendererOptions::create() );
}

void ViewerRendererWidget::aboutToShowRenderEngineMenu()
{
  DP_ASSERT( dynamic_cast<QMenu*>(sender()) );
  QMenu * menu = static_cast<QMenu*>(sender());
  QList<QAction*> actions = menu->actions();
  for ( QList<QAction*>::const_iterator it = actions.begin() ; it != actions.end() ; ++it )
  {
    (*it)->setChecked( m_rendererType == (*it)->data().toInt() );
  }
}

void ViewerRendererWidget::aboutToShowAntialiasingMenu()
{
  DP_ASSERT( dynamic_cast<QMenu*>(sender()) );
  QMenu * menu = static_cast<QMenu*>(sender());

  unsigned int n = menu->actions().size();
  DP_ASSERT( n );
  dp::gl::RenderContextFormat format = getFormat();

  // disable unavailable formats, starting from the last one,
  // assuming all formats below the first available are also available
  unsigned int i = n - 1;
  for ( ; 0 < i ; i-- )
  {
    QAction * action = menu->actions()[i];
    unsigned int data = action->data().toUInt();

    dp::gl::RenderContextFormat f = format;
    f.setMultisampleCoverage( data >> 8, data & 0xFF );
    if ( f.isAvailable() )
    {
      break;
    }
    action->setEnabled( false );
  }
  for ( ; 0 < i ; --i )
  {
    menu->actions()[i]->setEnabled( true );
  }

  // check the current menu entry
  unsigned int colorSamples, coverageSamples;
  format.getMultisampleCoverage( colorSamples, coverageSamples );

  for ( unsigned int i=0 ; i<n ; i++ )
  {
    QAction * action = menu->actions()[i];
    unsigned int data = action->data().toUInt();
    action->setChecked( ( colorSamples == ( data >> 8 ) ) && ( coverageSamples == ( data & 0xFF ) ) );
    DP_ASSERT( !action->isChecked() || action->isEnabled() );
  }
}

void ViewerRendererWidget::checkViewportFormat30Bit( QAction * action )
{
  dp::gl::RenderContextFormat format = getFormat();
  if ( format.isThirtyBit() )
  {
    action->setChecked( true );
  }
  else
  {
    action->setChecked( false );
    dp::gl::RenderContextFormat f = format;
    f.setThirtyBit( true );
    action->setEnabled( f.isAvailable() );
  }
}

void ViewerRendererWidget::checkViewportFormatSRGB( QAction * action )
{
  dp::gl::RenderContextFormat format = getFormat();
  if ( format.isSRGB() )
  {
    action->setChecked( true );
  }
  else
  {
    action->setChecked( false );
    dp::gl::RenderContextFormat f = format;
    f.setSRGB( true );
    action->setEnabled( f.isAvailable() );
  }
}

void ViewerRendererWidget::checkViewportFormatAntialiasing( QMenu * menu )
{
  bool enable = !GetApp()->isTonemapperEnabled();
  if ( enable )
  {
    dp::gl::RenderContextFormat format = getFormat();
    if ( ! format.getMultisample() )
    {
      dp::gl::RenderContextFormat f = format;
      f.setMultisample( 1 );    // check if multisample is available at all
      enable = f.isAvailable();
    }
  }
  menu->setEnabled( enable );
}

void ViewerRendererWidget::checkViewportFormatStencil( QAction * action )
{
  dp::gl::RenderContextFormat format = getFormat();
  if ( format.isStencil() )
  {
    action->setChecked( true );
  }
  else
  {
    action->setChecked( false );
    dp::gl::RenderContextFormat f = format;
    f.setStencil( true );
    action->setEnabled( f.isAvailable() );
  }
}

void ViewerRendererWidget::checkViewportFormatStereo( QAction * action )
{
  dp::gl::RenderContextFormat format = getFormat();
  if ( format.isStereo() )
  {
    action->setChecked( true );
  }
  else
  {
    action->setChecked( false );
    dp::gl::RenderContextFormat f = format;
    f.setStereo( true );
    action->setEnabled( f.isAvailable() );
  }
}

void ViewerRendererWidget::aboutToShowCullingMenu()
{
  DP_ASSERT( dynamic_cast<QMenu*>(sender()) );
  QMenu * menu = static_cast<QMenu*>(sender());
  QList<QAction*> actions = menu->actions();

  DP_ASSERT( actions.size() == 5 );

  bool enabled = m_sceneRendererPipeline->isCullingEnabled();
  dp::culling::Mode mode = m_sceneRendererPipeline->getCullingMode();
  for ( int i=0 ; i<4 ; i++ )
  {
    actions[i]->setChecked( enabled && ( i == mode ) );
  }
  actions[4]->setChecked( ! enabled );
}

void ViewerRendererWidget::aboutToShowViewportFormatMenu()
{
  DP_ASSERT( dynamic_cast<QMenu*>(sender()) );
  QList<QAction*> actions = static_cast<QMenu*>(sender())->actions();

  DP_ASSERT( actions.size() == 5 );
  DP_ASSERT( actions[0]->text() == "&30 Bit" );
  DP_ASSERT( actions[1]->menu() && actions[1]->menu()->title() == "&Antialiasing" );
  DP_ASSERT( actions[2]->text() == "s&RGB" );
  DP_ASSERT( actions[3]->text() == "&Stencil" );
  DP_ASSERT( actions[4]->text() == "S&tereo" );

  checkViewportFormat30Bit( actions[0] );
  checkViewportFormatAntialiasing( actions[1]->menu() );
  checkViewportFormatSRGB( actions[2] );
  checkViewportFormatStencil( actions[3] );
  checkViewportFormatStereo( actions[4] );
}

void ViewerRendererWidget::triggeredAddLightSourceMenu( QAction * action )
{
  LightSourceSharedPtr lightSource;
  switch ( action->data().toUInt() )
  {
    case 0 :
      addDirectedLight();
      break;
    case 1 :
      addPointLight();
      break;
    case 2 :
      addSpotLight();
      break;
    default :
      DP_ASSERT( false );
      break;
  }
}

void ViewerRendererWidget::triggeredAddHeadlightMenu( QAction * action )
{
  std::string name;
  LightSourceSharedPtr lightSource;
  switch ( action->data().toUInt() )
  {
    case 0 :
      lightSource = createStandardDirectedLight( Vec3f( 1.0f, -1.0f, -1.0f ), Vec3f( 1.0f, 1.0f, 1.0f ) );
      name = "SVDirectedLight";
      break;
    case 1 :
      lightSource = createStandardPointLight( Vec3f( 0.0f, 0.0f, 0.0f ), Vec3f( 1.0f, 1.0f, 1.0f ) );
      name = "SVPointLight";
      break;
    case 2 :
      lightSource = createStandardSpotLight( Vec3f( 0.0f, 0.0f, 0.0f ), Vec3f( 0.0f, 0.0f, -1.0f ), Vec3f( 1.0f, 1.0f, 1.0f ) );
      name = "SVSpotLight";
      break;
    default :
      DP_ASSERT( false );
      break;
  }
  if ( lightSource )
  {
    lightSource->setName( name );
  }
  DP_ASSERT( m_viewState );
  ExecuteCommand( new CommandAddObject( m_viewState->getCamera(), lightSource ) );
}

void ViewerRendererWidget::triggeredCullingMenu( QAction * action )
{
  if ( action->isChecked() )
  {
    dp::culling::Mode mode;
    if ( action->text() == "Off" )
    {
      m_sceneRendererPipeline->setCullingEnabled( false );
    }
    else
    {
      QString text = action->text();
      if ( text == "Auto" )
      {
        mode = dp::culling::MODE_AUTO;
      }
      else if ( text == "CPU" )
      {
        mode = dp::culling::MODE_CPU;
      }
      else if ( text == "CUDA" )
      {
        mode = dp::culling::MODE_CUDA;
      }
      else if ( text == "OpenGL Compute" )
      {
        mode = dp::culling::MODE_OPENGL_COMPUTE;
      }
      else
      {
        DP_ASSERT( !"unknown culling mode selected" );
      }
      m_sceneRendererPipeline->setCullingEnabled( true );
      m_sceneRendererPipeline->setCullingMode( mode );
    }
  }
}

void ViewerRendererWidget::triggeredViewportFormat30Bit( bool checked )
{
  dp::gl::RenderContextFormat format = getFormat();
  if( format.isThirtyBit() != checked )
  {
    format.setThirtyBit( checked );
    setFormat( format );
    restartUpdate();
  }
}

void ViewerRendererWidget::triggeredViewportFormatSRGB( bool checked )
{
  dp::gl::RenderContextFormat format = getFormat();
  if( format.isSRGB() != checked )
  {
    format.setSRGB( checked );
    setFormat( format );
    m_sRGBDesired = checked;
    restartUpdate();
  }
}

void ViewerRendererWidget::triggeredRenderEngineMenu( QAction * action )
{
  if ( action->isChecked() )
  {
    setRendererType( (ViewerRendererWidget::RendererType)action->data().toUInt() );
  }
}

void ViewerRendererWidget::triggeredAntialiasingMenu( QAction * action )
{
  if ( action->isChecked() )
  {
    dp::gl::RenderContextFormat format = getFormat();
    unsigned int data = action->data().toUInt();
    format.setMultisampleCoverage( data >> 8, data & 0xFF );
    setFormat( format );
    restartUpdate();
  }
}

void ViewerRendererWidget::triggeredViewportFormatStencil( bool checked )
{
  dp::gl::RenderContextFormat format = getFormat();
  if( format.isStencil() != checked )
  {
    format.setStencil( checked );
    setFormat( format );
    restartUpdate();
  }
}

void ViewerRendererWidget::triggeredViewportFormatStereo( bool checked )
{
  dp::gl::RenderContextFormat format = getFormat();
  if( format.isStereo() != checked )
  {
    format.setStereo( checked );
    setFormat( format );
    restartUpdate();
  }
}

void ViewerRendererWidget::setEnvironmentEnabledChanged()
{
  m_sceneRendererPipeline->setEnvironmentRenderingEnabled( GetApp()->getPreferences()->getEnvironmentEnabled() );
}

void ViewerRendererWidget::updateEnvironment()
{
  if (m_sceneRendererPipeline)
  {
    m_sceneRendererPipeline->updateEnvironment();
  }
}

dp::sg::renderer::rix::gl::TransparencyMode ViewerRendererWidget::getTransparencyMode() const
{
  DP_ASSERT( m_sceneRendererPipeline );
  return( m_sceneRendererPipeline->getTransparencyMode() );
}

void ViewerRendererWidget::setTransparencyMode( dp::sg::renderer::rix::gl::TransparencyMode mode )
{
  DP_ASSERT( m_sceneRendererPipeline );
  DP_ASSERT( mode != getTransparencyMode() );
  m_sceneRendererPipeline->setTransparencyMode( mode );
  restartUpdate();
}

unsigned int ViewerRendererWidget::getOITDepth() const
{
  return( m_oitDepth );
}

void ViewerRendererWidget::setOITDepth( unsigned int depth )
{
  if ( m_oitDepth != depth )
  {
    m_oitDepth = depth;
    dp::sg::renderer::rix::gl::SceneRendererSharedPtr const& sceneRenderer = getSceneRenderer().staticCast<dp::sg::renderer::rix::gl::SceneRenderer>();
    if ( sceneRenderer )
    {
      sceneRenderer->getTransparencyManager()->setLayersCount( m_oitDepth );
      restartUpdate();
    }
  }
}

void ViewerRendererWidget::setTonemapperEnabled( bool enabled )
{
  m_sceneRendererPipeline->setTonemapperEnabled( enabled );
  restartUpdate();
}

bool ViewerRendererWidget::isTonemapperEnabled() const
{
  return( m_sceneRendererPipeline->isTonemapperEnabled() );
}

TonemapperValues ViewerRendererWidget::getTonemapperValues() const
{
  return m_sceneRendererPipeline->getTonemapperValues();
}

void ViewerRendererWidget::setTonemapperValues( const TonemapperValues& tonemapperValues )
{
  m_sceneRendererPipeline->setTonemapperValues( tonemapperValues );
  restartUpdate();
}

void ViewerRendererWidget::setManipulatorType( ManipulatorType mt  )
{
  if( m_manipulatorType != mt )
  {
    m_manipulatorType = mt;
    delete m_currentManipulator;
    m_currentManipulator = nullptr;
    // reset manipulator in case new returns same pointer
    setManipulator( nullptr );
    setContinuousUpdate( false );

    switch( m_manipulatorType )
    {
      case MANIPULATOR_TRACKBALL:
      {
        dp::sg::ui::manipulator::TrackballCameraManipulatorHIDSync * manip = new dp::sg::ui::manipulator::TrackballCameraManipulatorHIDSync();
        // MMM - shouldn't the SceniXSceneRendererWidget set these too, so we don't have to
        //       do it for every one?
        manip->setHID( this );
        manip->setRenderTarget( getRenderTarget() );
        m_currentManipulator = manip;
      }
      break;

      case MANIPULATOR_CYLINDRICAL:
      {
        dp::sg::ui::manipulator::CylindricalCameraManipulatorHIDSync * manip = new dp::sg::ui::manipulator::CylindricalCameraManipulatorHIDSync();
        manip->setHID( this );
        manip->setRenderTarget( getRenderTarget() );
        m_currentManipulator = manip;
      }
      break;

      case MANIPULATOR_FLY:
      {
        dp::sg::ui::manipulator::FlightCameraManipulatorHIDSync * manip = new dp::sg::ui::manipulator::FlightCameraManipulatorHIDSync();

        manip->setHID( this );
        manip->setRenderTarget( getRenderTarget() );

        DP_ASSERT( m_viewState );
        DP_ASSERT( m_viewState->getScene() );

        float sceneRadius = 5.f; // set to 5 so end result is 1 m/s if no root
        if (m_viewState->getScene()->getRootNode())
        {
          const Sphere3f & bs = m_viewState->getScene()->getRootNode()->getBoundingSphere();
          sceneRadius = bs.getRadius();
        }

        // set speed so that it is possible to traverse the database in 10 seconds
        manip->setSpeed( sceneRadius / 5.f ); 

        m_currentManipulator = manip;
        setContinuousUpdate( true );
      }
      break;

      case MANIPULATOR_WALK:
      {
        dp::sg::ui::manipulator::WalkCameraManipulatorHIDSync * manip = new dp::sg::ui::manipulator::WalkCameraManipulatorHIDSync();

        manip->setHID( this );
        manip->setRenderTarget( getRenderTarget() );

        DP_ASSERT( m_viewState );
        DP_ASSERT( m_viewState->getScene() );

        float sceneRadius = 5.0f; // set to 5.0 so end result is 0.5 m/s if no root
        if (m_viewState->getScene()->getRootNode())
        {
          const Sphere3f & bs = m_viewState->getScene()->getRootNode()->getBoundingSphere();
          sceneRadius = bs.getRadius();
        }

        // set speed so that it is possible to traverse the database in 20 seconds
        manip->setSpeed( sceneRadius / 10.f ); 

        m_currentManipulator = manip;
        setContinuousUpdate( true );
      }
      break;

      case MANIPULATOR_NONE:
      default:
        // do nothing for now
        break;
    }

    // may still be NULL -> thats OK
    setManipulator( m_currentManipulator );
  }
}

ViewerRendererWidget::ManipulatorType ViewerRendererWidget::getManipulatorType() const
{
  return m_manipulatorType;
}

void ViewerRendererWidget::onRenderTargetChanged( const dp::gl::RenderTargetSharedPtr &oldTarget, const dp::gl::RenderTargetSharedPtr &newTarget )
{
  switch( getRendererType() )
  {
    case RENDERER_RASTERIZE_XBAR :
      {
        Viewer * viewer = GetApp();
        setSceneRenderer( dp::sg::renderer::rix::gl::SceneRenderer::create( viewer->getRenderEngine().c_str(), viewer->getShaderManagerType(), viewer->getCullingMode() ) );
      }
      break;
    default :
      DP_ASSERT( !"ViewerRendererWidget::onRenderTargetChanged: unknown renderType" );
      break;
  }
}

dp::gl::RenderContextFormat ViewerRendererWidget::getFormat() const
{
  dp::gl::RenderContextFormat f = SceniXQGLSceneRendererWidget::getFormat();
  f.setSRGB( m_sRGBDesired && f.isSRGB() );
  return( f );
}
