// Copyright NVIDIA Corporation 2013
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


#include "SceneTreeItem.h"
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Object.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Scene.h>

using namespace dp::sg::core;

SceneTreeItem::SceneTreeItem( ObjectSharedPtr const & object )
  : m_object(object)
{
  dp::sg::core::ObjectCode objectCode = m_object->getObjectCode();

  std::string name = m_object->getName();
  if ( name.empty() )
  {
    name = "unnamed " + objectCodeToName( objectCode );
  }
  setText( 0, name.c_str() );
  setToolTip( 0, QString( objectCodeToName( objectCode ).c_str() ) );

  setChildIndicatorPolicy();

  QPixmap pixmap( 16, 16 );
  switch( objectCode )
  {
    case OC_GEONODE :
      pixmap.load( ":/images/SubNode.png" );
      break;
    case OC_GROUP :
      pixmap.load( ":/images/Group.png" );
      break;
    case OC_LOD :
      pixmap.load( ":/images/LevelOfDetail.png" );
      break;
    case OC_TRANSFORM :
      pixmap.load( ":/images/Transform.png" );
      break;
    case OC_LIGHT_SOURCE :
      DP_ASSERT( m_object.isPtrTo<LightSource>() );
      {
        LightSourceSharedPtr const& lightSource = m_object.staticCast<LightSource>();
        if ( isStandardDirectedLight( lightSource ) )
        {
          pixmap.load( ":/images/DirectedLight.png" );
        }
        else if ( isStandardPointLight( lightSource ) )
        {
          pixmap.load( ":/images/PointLight.png" );
        }
        else
        {
          DP_ASSERT( isStandardSpotLight( lightSource ) );
          pixmap.load( ":/images/SpotLight.png" );
        }
      }
      break;
    case OC_VERTEX_ATTRIBUTE_SET :
      pixmap.load( ":/images/Drawable.png" );
      break;
    case OC_PRIMITIVE :
      DP_ASSERT( m_object.isPtrTo<Primitive>() );
      {
        PrimitiveSharedPtr const& p = m_object.staticCast<Primitive>();
        switch( p->getPrimitiveType() )
        {
          case PRIMITIVE_QUADS:
            pixmap.load( ":/images/Quads.png" );
            break;
          case PRIMITIVE_QUAD_STRIP :
            pixmap.load( ":/images/QuadMeshes.png" );
            break;
          case PRIMITIVE_TRIANGLES :
          case PRIMITIVE_TRIANGLE_STRIP :
          case PRIMITIVE_TRIANGLE_FAN :
            pixmap.load( ":/images/Triangles.png" );
            break;
          case PRIMITIVE_LINES :
          case PRIMITIVE_LINE_STRIP :
            pixmap.load( ":/images/Lines.png" );
            break;
          case PRIMITIVE_POINTS :
            pixmap.load( ":/images/Points.png" );
            break;
          default :
            pixmap.load( ":/images/DefaultNode.png" );
            break;
        }
      }
      break;
    case OC_PARALLELCAMERA :
      pixmap.load( ":/images/ParallelCamera.png" );
      break;
    case OC_PERSPECTIVECAMERA :
      pixmap.load( ":/images/PerspectiveCamera.png" );
      break;
    case OC_MATRIXCAMERA :
      pixmap.load( ":/images/Camera.png" );
      break;
    case OC_SCENE :
      pixmap.load( ":/images/MainNode.png" );
      break;
    case OC_SWITCH :
    case OC_BILLBOARD :
    case OC_CLIPPLANE :
    case OC_INDEX_SET :
    case OC_EFFECT_DATA :
    case OC_PARAMETER_GROUP_DATA :
    case OC_SAMPLER :
      pixmap.load( ":/images/DefaultNode.png" );
      break;
    default :
    case OC_INVALID :
      DP_ASSERT( false );
      break;
  }
  if ( !pixmap.isNull() )
  {
    setIcon( 0, QIcon( pixmap ) );
  }
}

void SceneTreeItem::expandItem()
{
  if ( childCount() == 0 )
  {
    switch( m_object->getObjectCode() )
    {
      case OC_GEONODE :
        {
          GeoNodeSharedPtr const& gn = m_object.staticCast<GeoNode>();
          if ( gn->getMaterialEffect() )
          {
            addChild( new SceneTreeItem( gn->getMaterialEffect() ) );
          }
          if ( gn->getPrimitive() )
          {
            addChild( new SceneTreeItem( gn->getPrimitive() ) );
          }
        }
        break;
      case OC_GROUP :
      case OC_LOD :
      case OC_SWITCH :
      case OC_TRANSFORM :
      case OC_BILLBOARD :
        {
          GroupSharedPtr const& g = m_object.staticCast<Group>();
          for ( Group::ChildrenIterator it = g->beginChildren() ; it != g->endChildren() ; ++it )
          {
            addChild( new SceneTreeItem( *it ) );
          }
        }
        break;
      case OC_LIGHT_SOURCE :
        {
          LightSourceSharedPtr const& ls = m_object.staticCast<LightSource>();
          if ( ls->getLightEffect() )
          {
            addChild( new SceneTreeItem( ls->getLightEffect() ) );
          }
        }
        break;
      case OC_PARALLELCAMERA :
      case OC_PERSPECTIVECAMERA :
      case OC_MATRIXCAMERA :
        {
          CameraSharedPtr const& c = m_object.staticCast<Camera>();
          for ( Camera::HeadLightIterator it = c->beginHeadLights() ; it != c->endHeadLights() ; ++it )
          {
            addChild( new SceneTreeItem( *it ) );
          }
        }
        break;
      case OC_PRIMITIVE :
        {
          PrimitiveSharedPtr const& p = m_object.staticCast<Primitive>();
          if ( p->getIndexSet() )
          {
            addChild( new SceneTreeItem( p->getIndexSet() ) );
          }
          if ( p->getVertexAttributeSet() )
          {
            addChild( new SceneTreeItem( p->getVertexAttributeSet() ) );
          }
        }
        break;
      case OC_EFFECT_DATA :
        {
          EffectDataSharedPtr const& ed = m_object.staticCast<EffectData>();
          dp::fx::EffectSpecSharedPtr const & es = ed->getEffectSpec();
          for ( dp::fx::EffectSpec::iterator it = es->beginParameterGroupSpecs() ; it != es->endParameterGroupSpecs() ; ++it )
          {
            if ( ed->getParameterGroupData( it ) )
            {
              addChild( new SceneTreeItem( ed->getParameterGroupData( it ) ) );
            }
          }
        }
        break;
      case OC_PARAMETER_GROUP_DATA :
        {
          ParameterGroupDataSharedPtr const& pgd = m_object.staticCast<ParameterGroupData>();
          dp::fx::ParameterGroupSpecSharedPtr const & pgs = pgd->getParameterGroupSpec();
          for ( dp::fx::ParameterGroupSpec::iterator it = pgs->beginParameterSpecs() ; it != pgs->endParameterSpecs() ; ++it )
          {
            if ( ( ( it->first.getType() & dp::fx::PT_POINTER_TYPE_MASK ) == dp::fx::PT_SAMPLER_PTR )
              && ( it->first.getAnnotation().find( "Hidden" ) == std::string::npos ) )
            {
              addChild( new SceneTreeItem( pgd->getParameter<SamplerSharedPtr>( it ) ) );
            }
          }
        }
        break;
      case OC_SCENE :
        {
          SceneSharedPtr const& s = m_object.staticCast<Scene>();
          for ( Scene::CameraIterator it = s->beginCameras() ; it != s->endCameras() ; ++it )
          {
            addChild( new SceneTreeItem( *it ) );
          }
          if ( s->getRootNode() )
          {
            addChild( new SceneTreeItem( s->getRootNode() ) );
          }
        }
        break;
      case OC_INVALID :
      case OC_CLIPPLANE :
      case OC_VERTEX_ATTRIBUTE_SET :    // do we need to introduce OC_VERTEX_ATTRIBUTE, to continue display here?
      case OC_INDEX_SET :
      case OC_SAMPLER :
      default :
        DP_ASSERT( false );
        break;
    }
  }
}

ObjectSharedPtr const & SceneTreeItem::getObject() const
{
  return( m_object );
}

void SceneTreeItem::setChildIndicatorPolicy()
{
  unsigned int objectCode = m_object->getObjectCode();

  bool showIndicator = false;
  switch( objectCode )
  {
    case OC_GEONODE :
      DP_ASSERT( m_object.isPtrTo<GeoNode>() );
      {
        GeoNodeSharedPtr const& gn = m_object.staticCast<GeoNode>();
        showIndicator = ( gn->getMaterialEffect() || gn->getPrimitive() );
      }
      break;
    case OC_GROUP :
    case OC_LOD :
    case OC_SWITCH :
    case OC_TRANSFORM :
    case OC_BILLBOARD :
      DP_ASSERT( m_object.isPtrTo<Group>() );
      {
        GroupSharedPtr const& g = m_object.staticCast<Group>();
        showIndicator = ( g->getNumberOfChildren() || g->getNumberOfClipPlanes() );
      }
      break;
    case OC_LIGHT_SOURCE :
      DP_ASSERT( m_object.isPtrTo<LightSource>() );
      showIndicator = !!m_object.staticCast<LightSource>()->getLightEffect();
      break;
    case OC_PERSPECTIVECAMERA :
    case OC_PARALLELCAMERA :
    case OC_MATRIXCAMERA :
      DP_ASSERT( m_object.isPtrTo<Camera>() );
      showIndicator = ( 0 < m_object.staticCast<Camera>()->getNumberOfHeadLights() );
      break;
    case OC_PRIMITIVE :
      DP_ASSERT( m_object.isPtrTo<Primitive>() );
      {
        PrimitiveSharedPtr const& p = m_object.staticCast<Primitive>();
        showIndicator = ( p->getIndexSet() || p->getVertexAttributeSet() );
      }
      break;
    case OC_EFFECT_DATA :
      DP_ASSERT( m_object.isPtrTo<EffectData>() );
      showIndicator = !!m_object.staticCast<EffectData>()->getNumberOfParameterGroupData();
      break;
    case OC_PARAMETER_GROUP_DATA :
      DP_ASSERT( m_object.isPtrTo<ParameterGroupData>() );
      {
        ParameterGroupDataSharedPtr const& pgd = m_object.staticCast<ParameterGroupData>();
        dp::fx::ParameterGroupSpecSharedPtr const & pgs = pgd->getParameterGroupSpec();
        for ( dp::fx::ParameterGroupSpec::iterator it = pgs->beginParameterSpecs() ; it != pgs->endParameterSpecs() && !showIndicator ; ++it )
        {
          showIndicator = ( ( it->first.getType() & dp::fx::PT_POINTER_TYPE_MASK ) == dp::fx::PT_SAMPLER_PTR )
                      &&  ( it->first.getAnnotation().find( "Hidden" ) == std::string::npos );
        }
      }
      break;
    case OC_SCENE :
      DP_ASSERT( m_object.isPtrTo<Scene>() );
      {
        SceneSharedPtr const& s = m_object.staticCast<Scene>();
        showIndicator = ( s->getNumberOfCameras() || s->getRootNode() );
      }
      break;
    case OC_CLIPPLANE :
    case OC_VERTEX_ATTRIBUTE_SET :    // do we need to introduce OC_VERTEX_ATTRIBUTE, to continue display here?
    case OC_INDEX_SET :
    case OC_SAMPLER :
      break;
    default :
    case OC_INVALID :
      DP_ASSERT( false );
      break;
  }
  QTreeWidgetItem::setChildIndicatorPolicy( showIndicator ? QTreeWidgetItem::ShowIndicator : QTreeWidgetItem::DontShowIndicator );
}

void SceneTreeItem::setObject( ObjectSharedPtr const & object )
{
  m_object = object;
  update();
}

void SceneTreeItem::update()
{
  if ( childCount() != 0 )
  {
    std::set<dp::sg::core::ObjectSharedPtr> objects;
    switch( m_object->getObjectCode() )
    {
      case OC_GEONODE :
        {
          GeoNodeSharedPtr const& gn = m_object.staticCast<GeoNode>();
          if ( gn->getMaterialEffect() )
          {
            objects.insert( gn->getMaterialEffect() );
          }
          if ( gn->getPrimitive() )
          {
            objects.insert( gn->getPrimitive() );
          }
        }
        break;
      case OC_GROUP :
      case OC_LOD :
      case OC_SWITCH :
      case OC_TRANSFORM :
      case OC_BILLBOARD :
        {
          GroupSharedPtr const& g = m_object.staticCast<Group>();
          for ( Group::ChildrenIterator it = g->beginChildren() ; it != g->endChildren() ; ++it )
          {
            objects.insert( *it );
          }
        }
        break;
      case OC_LIGHT_SOURCE :
        {
          LightSourceSharedPtr const& ls = m_object.staticCast<LightSource>();
          if ( ls->getLightEffect() )
          {
            objects.insert( ls->getLightEffect() );
          }
        }
        break;
      case OC_PARALLELCAMERA :
      case OC_PERSPECTIVECAMERA :
      case OC_MATRIXCAMERA :
        {
          CameraSharedPtr const& c = m_object.staticCast<Camera>();
          for ( Camera::HeadLightIterator it = c->beginHeadLights() ; it != c->endHeadLights() ; ++it )
          {
            objects.insert( *it );
          }
        }
        break;
      case OC_PRIMITIVE :
        {
          PrimitiveSharedPtr const& p = m_object.staticCast<Primitive>();
          if ( p->getIndexSet() )
          {
            objects.insert( p->getIndexSet() );
          }
          if ( p->getVertexAttributeSet() )
          {
            objects.insert( p->getVertexAttributeSet() );
          }
        }
        break;
      case OC_EFFECT_DATA :
        {
          EffectDataSharedPtr const& ed = m_object.staticCast<EffectData>();
          dp::fx::EffectSpecSharedPtr const & es = ed->getEffectSpec();
          for ( dp::fx::EffectSpec::iterator it = es->beginParameterGroupSpecs() ; it != es->endParameterGroupSpecs() ; ++it )
          {
            if ( ed->getParameterGroupData( it ) )
            {
              objects.insert( ed->getParameterGroupData( it ) );
            }
          }
        }
        break;
      case OC_PARAMETER_GROUP_DATA :
        {
          ParameterGroupDataSharedPtr const& pgd = m_object.staticCast<ParameterGroupData>();
          dp::fx::ParameterGroupSpecSharedPtr const & pgs = pgd->getParameterGroupSpec();
          for ( dp::fx::ParameterGroupSpec::iterator it = pgs->beginParameterSpecs() ; it != pgs->endParameterSpecs() ; ++it )
          {
            if ( ( it->first.getType() & dp::fx::PT_POINTER_TYPE_MASK ) == dp::fx::PT_SAMPLER_PTR )
            {
              objects.insert( pgd->getParameter<SamplerSharedPtr>( it ) );
            }
          }
        }
        break;
      case OC_SCENE :
        {
          SceneSharedPtr const& s = m_object.staticCast<Scene>();
          for ( Scene::CameraIterator it = s->beginCameras() ; it != s->endCameras() ; ++it )
          {
            objects.insert( *it );
          }
          if ( s->getRootNode() )
          {
            objects.insert( s->getRootNode() );
          }
        }
        break;
      case OC_INVALID:
      case OC_CLIPPLANE :
      case OC_VERTEX_ATTRIBUTE_SET :
      case OC_INDEX_SET :
      case OC_SAMPLER :
      default :
        DP_ASSERT( false );
        break;
    }

    for ( int i=0 ; i<childCount() ; i++ )
    {
      std::set<dp::sg::core::ObjectSharedPtr>::iterator it = objects.find( static_cast<SceneTreeItem*>(child( i ))->getObject() );
      if ( it == objects.end() )
      {
        QTreeWidgetItem * twi = takeChild( i-- );
        DP_ASSERT( twi );
        delete twi;
      }
      else
      {
        static_cast<SceneTreeItem*>(child( i ))->update();
        objects.erase( it );
      }
    }
    for ( std::set<dp::sg::core::ObjectSharedPtr>::iterator it = objects.begin() ; it != objects.end() ; ++it )
    {
      addChild( new SceneTreeItem( *it ) );
    }
  }
}
