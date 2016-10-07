// Copyright (c) 2013-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <QDialogButtonBox>
#include <QEvent>
#include <QFileDialog>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QMimeData>
#include <QDrag>
#include "CommandAddItem.h"
#include "CommandReplaceItem.h"
#include "ScenePropertiesWidget.h"
#include "SceneTreeBrowser.h"
#include "SceneTreeItem.h"
#include "Viewer.h"
#include <dp/fx/ParameterGroupSpec.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/ParameterGroupData.h>
#include <dp/sg/core/PipelineData.h>

using namespace dp::math;
using namespace dp::sg::core;

SceneTreeBrowser::SceneTreeBrowser( QWidget * parent )
  : QDockWidget( "Scene Tree", parent )
  , m_objectObserver(this)
{
  setObjectName( "Scene Tree" );
  setAcceptDrops( true );

  m_tree = new QTreeWidget();
  m_tree->setHeaderHidden( true );
  connect( m_tree, SIGNAL(currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)), this, SLOT(currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)) );
  connect( m_tree, SIGNAL(itemExpanded(QTreeWidgetItem*)), this, SLOT(itemExpanded(QTreeWidgetItem*)) );
  connect( m_tree, SIGNAL(itemPressed(QTreeWidgetItem*,int)), this, SLOT(itemPressed(QTreeWidgetItem*,int)) );
  setWidget( m_tree );
  connect(GetApp(), SIGNAL(sceneTreeChanged()), this, SLOT(updateTree()));
}

SceneTreeBrowser::~SceneTreeBrowser()
{
  DP_ASSERT( m_tree );
  if ( m_tree->currentItem() )
  {
    DP_ASSERT( dynamic_cast<SceneTreeItem*>(m_tree->currentItem()) );
    DP_ASSERT( static_cast<SceneTreeItem*>(m_tree->currentItem())->getObject() );
    static_cast<SceneTreeItem*>(m_tree->currentItem())->getObject()->detach( &m_objectObserver );
  }
}

QTreeWidget * SceneTreeBrowser::getTree() const
{
  return( m_tree );
}

void SceneTreeBrowser::selectObject( dp::sg::core::PathSharedPtr const& path )
{
  if ( isVisible() )
  {
    DP_ASSERT( dynamic_cast<SceneTreeItem*>(m_tree->topLevelItem( 0 )) );
    SceneTreeItem * item = static_cast<SceneTreeItem *>(m_tree->topLevelItem( 0 ));
    for ( unsigned int i=0 ; i<path->getLength() ; i++ )
    {
      if ( item->childCount() == 0 )
      {
        item->expandItem();
      }
      bool found = false;
      for ( int j=0 ; j<item->childCount() && !found ; j++ )
      {
        if ( static_cast<SceneTreeItem*>(item->child( j ))->getObject() == path->getFromHead( i ) )
        {
          item = static_cast<SceneTreeItem*>(item->child( j ));
          found = true;
        }
      }
      DP_ASSERT( found );
    }
    m_tree->setCurrentItem( item );
  }
}

void SceneTreeBrowser::setScene( SceneSharedPtr const & scene )
{
  m_tree->clear();
  if ( scene )
  {
    m_tree->addTopLevelItem( new SceneTreeItem( scene ) );
  }
}

std::vector<dp::fx::ParameterGroupSpec::iterator> getEmptySamplerParameters( const ParameterGroupDataSharedPtr & parameterGroupData )
{
  std::vector<dp::fx::ParameterGroupSpec::iterator> samplerParameters;
  const dp::fx::ParameterGroupSpecSharedPtr & pgs = parameterGroupData->getParameterGroupSpec();
  for ( dp::fx::ParameterGroupSpec::iterator it = pgs->beginParameterSpecs() ; it != pgs->endParameterSpecs() ; ++it )
  {
    if ( ( ( it->first.getType() & dp::fx::PT_POINTER_TYPE_MASK ) == dp::fx::PT_SAMPLER_PTR ) && ! parameterGroupData->getParameter<SamplerSharedPtr>( it ) )
    {
      samplerParameters.push_back( it );
    }
  }
  return( samplerParameters );
}

void SceneTreeBrowser::contextMenuEvent( QContextMenuEvent * event )
{
  QDockWidget::contextMenuEvent( event );

  if ( dynamic_cast<SceneTreeItem*>(m_tree->currentItem()) && dynamic_cast<SceneTreeItem*>(m_tree->currentItem())->getObject() )
  {
    SceneTreeItem * currentItem = static_cast<SceneTreeItem*>(m_tree->currentItem());

    QMenu menu( "Tree Context Menu", this );
    dp::sg::core::ObjectCode objectCode = currentItem->getObject()->getObjectCode();
    switch( objectCode )
    {
      case ObjectCode::GEO_NODE :
        {
          QAction * pipelineAction = menu.addAction( "&Show shader pipeline ..." );
          connect( pipelineAction, SIGNAL(triggered()), this, SLOT(triggeredShowShaderPipeline()) );
        }
        break;

      case ObjectCode::PRIMITIVE :
        {
          VertexAttributeSetSharedPtr vas = std::static_pointer_cast<Primitive>(currentItem->getObject())->getVertexAttributeSet();
          if ( !vas->getVertexAttribute( VertexAttributeSet::AttributeID::TEXCOORD0 ).getBuffer() )
          {
            QMenu * subMenu = menu.addMenu( "Generate Texture &Coordinates" );
            subMenu->addAction( "Cylindrical" );
            subMenu->actions().back()->setData( static_cast<unsigned int>(TextureCoordType::CYLINDRICAL) );
            subMenu->addAction( "Planar" );
            subMenu->actions().back()->setData( static_cast<unsigned int>(TextureCoordType::PLANAR) );
            subMenu->addAction( "Spherical" );
            subMenu->actions().back()->setData( static_cast<unsigned int>(TextureCoordType::SPHERICAL) );
            connect( subMenu, SIGNAL( triggered( QAction * ) ), this, SLOT( triggeredGenerateTextureCoordinatesMenu( QAction * ) ) );
          }

          if (      vas->getVertexAttribute( VertexAttributeSet::AttributeID::TEXCOORD0 ).getBuffer()
              &&  ! vas->getVertexAttribute( VertexAttributeSet::AttributeID::TEXCOORD6 ).getBuffer()
              &&  ! vas->getVertexAttribute( VertexAttributeSet::AttributeID::TEXCOORD7 ).getBuffer() )
          {
            QAction * action = menu.addAction( "Generate &Tangent Space" );
            connect( action, SIGNAL(triggered()), this, SLOT(triggeredGenerateTangentSpace()) );
          }
        }
        break;

      case ObjectCode::PARAMETER_GROUP_DATA :
        {
          ParameterGroupDataSharedPtr parameterGroupData = std::static_pointer_cast<ParameterGroupData>(currentItem->getObject());
          std::vector<dp::fx::ParameterGroupSpec::iterator> samplerParameters = getEmptySamplerParameters( parameterGroupData );
          if ( ! samplerParameters.empty() )
          {
            QMenu * subMenu = menu.addMenu( "&Add Sampler for Parameter" );
            for ( size_t i=0 ; i<samplerParameters.size() ; i++ )
            {
              subMenu->addAction( QApplication::translate( VIEWER_APPLICATION_NAME, samplerParameters[i]->first.getName().c_str() ) );
            }
            connect( subMenu, SIGNAL(triggered(QAction*)), this, SLOT(triggeredAddSamplerMenu(QAction*)) );
          }
        }
        break;

      case ObjectCode::PARALLEL_CAMERA :
      case ObjectCode::PERSPECTIVE_CAMERA :
      case ObjectCode::MATRIX_CAMERA :
        {
          QMenu * subMenu = menu.addMenu( "&Add Headlight" );
          subMenu->addAction( "&Directed Light" );
          subMenu->addAction( "&Point Light" );
          subMenu->addAction( "&Spot Light" );
          for ( int i=0 ; i<subMenu->actions().size() ; ++i )
          {
            subMenu->actions()[i]->setData( i );
          }
          connect( subMenu, SIGNAL(triggered(QAction*)), this, SLOT(triggeredAddHeadlightMenu(QAction*)) );
        }
        break;

      case ObjectCode::PIPELINE_DATA :
        {
          QAction * saveAction = menu.addAction( "&Save EffectData ..." );
          connect( saveAction, SIGNAL(triggered()), this, SLOT(triggeredSaveEffectData()) );

          QAction * replaceAction = menu.addAction( "&Replace by Clone" );
          connect( replaceAction, SIGNAL(triggered()), this, SLOT(triggeredReplaceByClone()) );
        }
        break;
    }

    if ( objectCode != ObjectCode::SCENE )
    {
      if ( ! menu.isEmpty() )
      {
        menu.addSeparator();
      }
      std::ostringstream oss;
      oss << "&Delete " << objectCodeToName( objectCode );
      QAction * deleteAction = menu.addAction( oss.str().c_str() );
      connect( deleteAction, SIGNAL(triggered()), this, SLOT(triggeredDeleteObject()) );
    }
    menu.exec( event->globalPos() );
  }
}

void SceneTreeBrowser::currentItemChanged( QTreeWidgetItem * current, QTreeWidgetItem * previous )
{
  if ( previous )
  {
    DP_ASSERT( dynamic_cast<SceneTreeItem*>(previous) );
    DP_ASSERT( static_cast<SceneTreeItem*>(previous)->getObject() );
    static_cast<SceneTreeItem*>(previous)->getObject()->detach( &m_objectObserver );
  }

  if ( current )
  {
    DP_ASSERT( dynamic_cast<SceneTreeItem*>(current) );
    DP_ASSERT( static_cast<SceneTreeItem*>(current)->getObject() );
    static_cast<SceneTreeItem*>(current)->getObject()->attach( &m_objectObserver );
  }

  emit currentItemChanged( dynamic_cast<SceneTreeItem*>(current)  ? static_cast<SceneTreeItem*>(current)->getObject()  : ObjectSharedPtr()
                         , dynamic_cast<SceneTreeItem*>(previous) ? static_cast<SceneTreeItem*>(previous)->getObject() : ObjectSharedPtr() );
}

void SceneTreeBrowser::itemExpanded( QTreeWidgetItem * item )
{
  DP_ASSERT( dynamic_cast<SceneTreeItem*>(item) );
  static_cast<SceneTreeItem*>(item)->expandItem();
}

void SceneTreeBrowser::itemPressed( QTreeWidgetItem * item, int column )
{
  DP_ASSERT( dynamic_cast<SceneTreeItem*>(item) );
  if ( ( QApplication::mouseButtons() & Qt::LeftButton ) && ( static_cast<SceneTreeItem*>(item)->getObject()->getObjectCode() == ObjectCode::PIPELINE_DATA ) )
  {
    // start drag'n'drop on EffectData
    // We don't set the EffectData as the MimeData, but the GeoNode, as both the material of the GeoNode and the geometry of the Primitive need to be copied!
    dp::fx::EffectSpec::Type type = std::static_pointer_cast<dp::sg::core::PipelineData>(static_cast<SceneTreeItem*>(item)->getObject())->getEffectSpec()->getType();
    if ( type == dp::fx::EffectSpec::Type::PIPELINE )
    {
      DP_ASSERT( item->parent() );
      item = item->parent();

      DP_ASSERT( std::dynamic_pointer_cast<GeoNode>(static_cast<SceneTreeItem*>(item)->getObject()) );
      GeoNodeSharedPtr geoNode = std::static_pointer_cast<GeoNode>(static_cast<SceneTreeItem*>(item)->getObject());
      QByteArray qba( reinterpret_cast<char*>(&geoNode), sizeof(void*) );

      QMimeData * mimeData = new QMimeData;
      mimeData->setData( "EffectData", qba );

      QDrag * drag = new QDrag( this );
      drag->setMimeData( mimeData );
      drag->exec();
    }
  }
}

void SceneTreeBrowser::triggeredAddHeadlightMenu( QAction * action )
{
  std::string name;
  LightSourceSharedPtr lightSource;
  switch ( action->data().toUInt() )
  {
    case 0 :
      lightSource = createStandardDirectedLight( Vec3f( 0.0f, 0.0f, -1.0f ), Vec3f( 1.0f, 1.0f, 1.0f ) );
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
  DP_ASSERT( lightSource );
  lightSource->setName( name );

  DP_ASSERT( dynamic_cast<SceneTreeItem*>(m_tree->currentItem()) );
  DP_ASSERT( std::dynamic_pointer_cast<Camera>(static_cast<SceneTreeItem*>(m_tree->currentItem())->getObject()) );
  ExecuteCommand( new CommandAddItem( static_cast<SceneTreeItem*>(m_tree->currentItem()), new SceneTreeItem( lightSource ) ) );
}

void SceneTreeBrowser::triggeredAddSamplerMenu( QAction * action )
{
  DP_ASSERT( dynamic_cast<SceneTreeItem*>(m_tree->currentItem()) );
  DP_ASSERT( std::dynamic_pointer_cast<ParameterGroupData>(static_cast<SceneTreeItem*>(m_tree->currentItem())->getObject()) );

  SamplerSharedPtr sampler = Sampler::create();
  sampler->setName( action->text().toStdString() );

  ExecuteCommand( new CommandAddItem( static_cast<SceneTreeItem*>(m_tree->currentItem()), new SceneTreeItem( sampler ) ) );
}

void SceneTreeBrowser::triggeredDeleteObject()
{
  DP_ASSERT( dynamic_cast<SceneTreeItem*>(m_tree->currentItem()) );
  DP_ASSERT( dynamic_cast<SceneTreeItem*>(m_tree->currentItem()->parent()) );

  ExecuteCommand( new CommandAddItem( static_cast<SceneTreeItem*>(m_tree->currentItem()->parent()), static_cast<SceneTreeItem*>(m_tree->currentItem()), false ) );
}

void SceneTreeBrowser::triggeredGenerateTangentSpace()
{
  DP_ASSERT( dynamic_cast<SceneTreeItem*>(m_tree->currentItem()) && dynamic_cast<SceneTreeItem*>(m_tree->currentItem())->getObject() );
  SceneTreeItem * currentItem = static_cast<SceneTreeItem*>(m_tree->currentItem());
  DP_ASSERT( std::dynamic_pointer_cast<Primitive>(currentItem->getObject()) );

  ExecuteCommand(new CommandGenerateTangentSpace(std::static_pointer_cast<Primitive>(currentItem->getObject())));
}

void SceneTreeBrowser::triggeredGenerateTextureCoordinatesMenu( QAction * action )
{
  DP_ASSERT( dynamic_cast<SceneTreeItem*>(m_tree->currentItem()) && dynamic_cast<SceneTreeItem*>(m_tree->currentItem())->getObject() );
  SceneTreeItem * currentItem = static_cast<SceneTreeItem*>(m_tree->currentItem());
  DP_ASSERT( std::dynamic_pointer_cast<Primitive>(currentItem->getObject()) );

  TextureCoordType tct = static_cast<TextureCoordType>( action->data().toInt() );
  ExecuteCommand(new CommandGenerateTextureCoordinates(std::static_pointer_cast<Primitive>(currentItem->getObject()), tct));
}

const char * domainCodeToName( dp::fx::Domain domain )
{
  switch( domain )
  {
    case dp::fx::Domain::VERTEX                  : return( "Vertex Shader" );
    case dp::fx::Domain::FRAGMENT                : return( "Fragment Shader" );
    case dp::fx::Domain::GEOMETRY                : return( "Geometry Shader" );
    case dp::fx::Domain::TESSELLATION_CONTROL    : return( "Tessellation Control Shader" );
    case dp::fx::Domain::TESSELLATION_EVALUATION : return( "Tessellation Evaluation Shader" );
    default :
      DP_ASSERT( false );
      return( "Unknown Shader" );
  }
}

void SceneTreeBrowser::triggeredReplaceByClone()
{
  DP_ASSERT( dynamic_cast<SceneTreeItem*>(m_tree->currentItem()) && dynamic_cast<SceneTreeItem*>(m_tree->currentItem())->getObject() );
  SceneTreeItem * currentItem = static_cast<SceneTreeItem*>(m_tree->currentItem());
  DP_ASSERT( currentItem->parent() && dynamic_cast<SceneTreeItem*>(currentItem->parent()) );

  ExecuteCommand(new CommandReplaceItem(static_cast<SceneTreeItem*>(currentItem->parent()), currentItem, new SceneTreeItem(std::static_pointer_cast<dp::sg::core::Object>(currentItem->getObject()->clone())), &m_objectObserver));
}

void SceneTreeBrowser::triggeredSaveEffectData()
{
  DP_ASSERT( dynamic_cast<SceneTreeItem*>(m_tree->currentItem()) && dynamic_cast<SceneTreeItem*>(m_tree->currentItem())->getObject() );
  DP_ASSERT( std::dynamic_pointer_cast<dp::sg::core::PipelineData>(static_cast<SceneTreeItem*>(m_tree->currentItem())->getObject()) );

  dp::sg::core::PipelineDataSharedPtr pipelineData = std::static_pointer_cast<dp::sg::core::PipelineData>(static_cast<SceneTreeItem*>(m_tree->currentItem())->getObject());
  std::string pipelineName = pipelineData->getEffectSpec()->getName();
  QString fileName = QFileDialog::getSaveFileName( this, tr( "Save PipelineData" ), QString( pipelineName.c_str() ) + QString( ".xml" ), tr( "XML (*.xml)" ) );
  if ( !fileName.isEmpty() )
  {
    pipelineData->save( fileName.toStdString() );
  }
}

void SceneTreeBrowser::triggeredShowShaderPipeline()
{
  DP_ASSERT( dynamic_cast<SceneTreeItem*>(m_tree->currentItem()) && dynamic_cast<SceneTreeItem*>(m_tree->currentItem())->getObject() );
  DP_ASSERT(std::static_pointer_cast<GeoNode>(static_cast<SceneTreeItem*>(m_tree->currentItem())->getObject()));
  GeoNodeSharedPtr geoNode = std::static_pointer_cast<GeoNode>(static_cast<SceneTreeItem*>(m_tree->currentItem())->getObject());

  DP_ASSERT( GetApp() && GetApp()->getMainWindow() && GetApp()->getMainWindow()->getCurrentViewport() );
  ViewerRendererWidget * vrw = GetApp()->getMainWindow()->getCurrentViewport();
  dp::sg::ui::SceneRendererSharedPtr sceneRenderer = vrw->getSceneRenderer();
  DP_ASSERT( sceneRenderer );
  std::map<dp::fx::Domain,std::string> sources[2];
  sources[0] = sceneRenderer->getShaderSources( geoNode, false );
  sources[1] = sceneRenderer->getShaderSources( geoNode, true );

  if ( sources[0].empty() && sources[1].empty() )
  {
    QMessageBox messageBox( QMessageBox::Information, tr( "Show Shader Pipeline" ), tr( "No Shader sources available." ), QMessageBox::Ok );
    messageBox.exec();
  }
  else
  {
    static std::string passName[2] = { "Forward Pass", "Depth Pass" };
    QTabWidget * topTab = new QTabWidget();
    for ( int i=0 ; i<2 ; i++ )
    {
      QTabWidget * tab = new QTabWidget();
      for ( std::map<dp::fx::Domain,std::string>::const_iterator it = sources[i].begin() ; it != sources[i].end() ; ++it )
      {
        QPlainTextEdit * edit = new QPlainTextEdit( tr( it->second.c_str() ) );
        edit->setReadOnly( true );
        edit->setLineWrapMode( QPlainTextEdit::NoWrap );
        tab->addTab( edit, tr( domainCodeToName( it->first ) ) );
      }
      topTab->addTab( tab, tr( passName[i].c_str() ) );
    }

    QDialogButtonBox * buttonBox = new QDialogButtonBox( QDialogButtonBox::Close );

    QVBoxLayout * layout = new QVBoxLayout();
    layout->addWidget( topTab );
    layout->addWidget( buttonBox );

    QDialog * dialog = new QDialog( this );
    dialog->setWindowTitle( QString( "Shader Pipeline: " ) + QString( geoNode->getMaterialPipeline()->getEffectSpec()->getName().c_str() ) );
    dialog->setLayout( layout );

    connect( buttonBox, SIGNAL(rejected()), dialog, SLOT(reject()) );

    dialog->show();
  }
}

void SceneTreeBrowser::updateTree()
{
  if ( isVisible() )
  {
    DP_ASSERT( dynamic_cast<SceneTreeItem*>(m_tree->topLevelItem( 0 )) );
    SceneTreeItem * item = static_cast<SceneTreeItem *>(m_tree->topLevelItem( 0 ));
    item->update();
  }
}


SceneTreeBrowser::ObjectObserver::ObjectObserver( SceneTreeBrowser * stb )
  : m_stb(stb)
{
}

void SceneTreeBrowser::ObjectObserver::onNotify( const dp::util::Event &event, dp::util::Payload *payload )
{
  DP_ASSERT( m_stb->getTree() && m_stb->getTree()->currentItem() );
  DP_ASSERT( dynamic_cast<SceneTreeItem*>(m_stb->getTree()->currentItem()) );
  SceneTreeItem * item = static_cast<SceneTreeItem*>(m_stb->getTree()->currentItem());
  DP_ASSERT( item->getObject() );

  std::string name = item->getObject()->getName();
  if ( name.empty() )
  {
    name = "unnamed " + objectCodeToName( item->getObject()->getObjectCode() );
  }

  if ( name != item->text( 0 ).toStdString() )
  {
    item->setText( 0, name.c_str() );
  }
}

void SceneTreeBrowser::ObjectObserver::onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload )
{
}
