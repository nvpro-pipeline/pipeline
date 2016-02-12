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


#include "SceneTreeItem.h"
#include "Viewer.h"
#include <dp/fx/EffectSpec.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Object.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Scene.h>

using namespace dp::sg::core;

SceneTreeItem::SceneTreeItem(ObjectSharedPtr const & object)
  : m_object(object)
{
  dp::sg::core::ObjectCode objectCode = m_object->getObjectCode();

  std::string name = stripNameSpaces(m_object->getName());
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
    case ObjectCode::GEO_NODE :
      pixmap.load( ":/images/SubNode.png" );
      break;
    case ObjectCode::GROUP :
      pixmap.load( ":/images/Group.png" );
      break;
    case ObjectCode::LOD :
      pixmap.load( ":/images/LevelOfDetail.png" );
      break;
    case ObjectCode::TRANSFORM :
      pixmap.load( ":/images/Transform.png" );
      break;
    case ObjectCode::LIGHT_SOURCE :
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
    case ObjectCode::VERTEX_ATTRIBUTE_SET :
      pixmap.load( ":/images/Drawable.png" );
      break;
    case ObjectCode::PRIMITIVE :
      DP_ASSERT( m_object.isPtrTo<Primitive>() );
      {
        PrimitiveSharedPtr const& p = m_object.staticCast<Primitive>();
        switch( p->getPrimitiveType() )
        {
          case PrimitiveType::QUADS:
            pixmap.load( ":/images/Quads.png" );
            break;
          case PrimitiveType::QUAD_STRIP :
            pixmap.load( ":/images/QuadMeshes.png" );
            break;
          case PrimitiveType::TRIANGLES :
          case PrimitiveType::TRIANGLE_STRIP :
          case PrimitiveType::TRIANGLE_FAN :
            pixmap.load( ":/images/Triangles.png" );
            break;
          case PrimitiveType::LINES :
          case PrimitiveType::LINE_STRIP :
            pixmap.load( ":/images/Lines.png" );
            break;
          case PrimitiveType::POINTS :
            pixmap.load( ":/images/Points.png" );
            break;
          default :
            pixmap.load( ":/images/DefaultNode.png" );
            break;
        }
      }
      break;
    case ObjectCode::PARALLEL_CAMERA :
      pixmap.load( ":/images/ParallelCamera.png" );
      break;
    case ObjectCode::PERSPECTIVE_CAMERA :
      pixmap.load( ":/images/PerspectiveCamera.png" );
      break;
    case ObjectCode::MATRIX_CAMERA :
      pixmap.load( ":/images/Camera.png" );
      break;
    case ObjectCode::SCENE :
      pixmap.load( ":/images/MainNode.png" );
      break;
    case ObjectCode::SWITCH :
    case ObjectCode::BILLBOARD :
    case ObjectCode::CLIP_PLANE :
    case ObjectCode::INDEX_SET :
    case ObjectCode::PARAMETER_GROUP_DATA :
    case ObjectCode::PIPELINE_DATA :
    case ObjectCode::SAMPLER :
      pixmap.load( ":/images/DefaultNode.png" );
      break;
    default :
    case ObjectCode::INVALID :
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
      case ObjectCode::GEO_NODE :
        {
          GeoNodeSharedPtr const& gn = m_object.staticCast<GeoNode>();
          if ( gn->getMaterialPipeline() )
          {
            addChild( new SceneTreeItem( gn->getMaterialPipeline() ) );
          }
          if ( gn->getPrimitive() )
          {
            addChild( new SceneTreeItem( gn->getPrimitive() ) );
          }
        }
        break;
      case ObjectCode::GROUP :
      case ObjectCode::LOD :
      case ObjectCode::SWITCH :
      case ObjectCode::TRANSFORM :
      case ObjectCode::BILLBOARD :
        {
          GroupSharedPtr const& g = m_object.staticCast<Group>();
          for ( Group::ChildrenIterator it = g->beginChildren() ; it != g->endChildren() ; ++it )
          {
            addChild( new SceneTreeItem( *it ) );
          }
        }
        break;
      case ObjectCode::LIGHT_SOURCE :
        {
          LightSourceSharedPtr const& ls = m_object.staticCast<LightSource>();
          if ( ls->getLightPipeline() )
          {
            addChild( new SceneTreeItem( ls->getLightPipeline() ) );
          }
        }
        break;
      case ObjectCode::PARALLEL_CAMERA :
      case ObjectCode::PERSPECTIVE_CAMERA :
      case ObjectCode::MATRIX_CAMERA :
        {
          CameraSharedPtr const& c = m_object.staticCast<Camera>();
          for ( Camera::HeadLightIterator it = c->beginHeadLights() ; it != c->endHeadLights() ; ++it )
          {
            addChild( new SceneTreeItem( *it ) );
          }
        }
        break;
      case ObjectCode::PRIMITIVE :
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
      case ObjectCode::PARAMETER_GROUP_DATA :
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
      case ObjectCode::PIPELINE_DATA :
        {
          dp::sg::core::PipelineDataSharedPtr const& pd = m_object.staticCast<dp::sg::core::PipelineData>();
          dp::fx::EffectSpecSharedPtr const & es = pd->getEffectSpec();
          for ( dp::fx::EffectSpec::iterator it = es->beginParameterGroupSpecs() ; it != es->endParameterGroupSpecs() ; ++it )
          {
            if ( pd->getParameterGroupData( it ) )
            {
              addChild( new SceneTreeItem( pd->getParameterGroupData( it ) ) );
            }
          }
        }
        break;
      case ObjectCode::SCENE :
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
      case ObjectCode::INVALID :
      case ObjectCode::CLIP_PLANE :
      case ObjectCode::VERTEX_ATTRIBUTE_SET :    // do we need to introduce OC_VERTEX_ATTRIBUTE, to continue display here?
      case ObjectCode::INDEX_SET :
      case ObjectCode::SAMPLER :
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
  dp::sg::core::ObjectCode objectCode = m_object->getObjectCode();

  bool showIndicator = false;
  switch( objectCode )
  {
    case ObjectCode::GEO_NODE :
      DP_ASSERT( m_object.isPtrTo<GeoNode>() );
      {
        GeoNodeSharedPtr const& gn = m_object.staticCast<GeoNode>();
        showIndicator = ( gn->getMaterialPipeline() || gn->getPrimitive() );
      }
      break;
    case ObjectCode::GROUP :
    case ObjectCode::LOD :
    case ObjectCode::SWITCH :
    case ObjectCode::TRANSFORM :
    case ObjectCode::BILLBOARD :
      DP_ASSERT( m_object.isPtrTo<Group>() );
      {
        GroupSharedPtr const& g = m_object.staticCast<Group>();
        showIndicator = ( g->getNumberOfChildren() || g->getNumberOfClipPlanes() );
      }
      break;
    case ObjectCode::LIGHT_SOURCE :
      DP_ASSERT( m_object.isPtrTo<LightSource>() );
      showIndicator = !!m_object.staticCast<LightSource>()->getLightPipeline();
      break;
    case ObjectCode::PERSPECTIVE_CAMERA :
    case ObjectCode::PARALLEL_CAMERA :
    case ObjectCode::MATRIX_CAMERA :
      DP_ASSERT( m_object.isPtrTo<Camera>() );
      showIndicator = ( 0 < m_object.staticCast<Camera>()->getNumberOfHeadLights() );
      break;
    case ObjectCode::PRIMITIVE :
      DP_ASSERT( m_object.isPtrTo<Primitive>() );
      {
        PrimitiveSharedPtr const& p = m_object.staticCast<Primitive>();
        showIndicator = ( p->getIndexSet() || p->getVertexAttributeSet() );
      }
      break;
    case ObjectCode::PARAMETER_GROUP_DATA :
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
    case ObjectCode::PIPELINE_DATA :
      DP_ASSERT( m_object.isPtrTo<dp::sg::core::PipelineData>() );
      showIndicator = !!m_object.staticCast<dp::sg::core::PipelineData>()->getNumberOfParameterGroupData();
      break;
    case ObjectCode::SCENE :
      DP_ASSERT( m_object.isPtrTo<Scene>() );
      {
        SceneSharedPtr const& s = m_object.staticCast<Scene>();
        showIndicator = ( s->getNumberOfCameras() || s->getRootNode() );
      }
      break;
    case ObjectCode::CLIP_PLANE :
    case ObjectCode::VERTEX_ATTRIBUTE_SET :    // do we need to introduce OC_VERTEX_ATTRIBUTE, to continue display here?
    case ObjectCode::INDEX_SET :
    case ObjectCode::SAMPLER :
      break;
    default :
    case ObjectCode::INVALID :
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
      case ObjectCode::GEO_NODE :
        {
          GeoNodeSharedPtr const& gn = m_object.staticCast<GeoNode>();
          if ( gn->getMaterialPipeline() )
          {
            objects.insert( gn->getMaterialPipeline() );
          }
          if ( gn->getPrimitive() )
          {
            objects.insert( gn->getPrimitive() );
          }
        }
        break;
      case ObjectCode::GROUP :
      case ObjectCode::LOD :
      case ObjectCode::SWITCH :
      case ObjectCode::TRANSFORM :
      case ObjectCode::BILLBOARD :
        {
          GroupSharedPtr const& g = m_object.staticCast<Group>();
          for ( Group::ChildrenIterator it = g->beginChildren() ; it != g->endChildren() ; ++it )
          {
            objects.insert( *it );
          }
        }
        break;
      case ObjectCode::LIGHT_SOURCE :
        {
          LightSourceSharedPtr const& ls = m_object.staticCast<LightSource>();
          if ( ls->getLightPipeline() )
          {
            objects.insert( ls->getLightPipeline() );
          }
        }
        break;
      case ObjectCode::PARALLEL_CAMERA :
      case ObjectCode::PERSPECTIVE_CAMERA :
      case ObjectCode::MATRIX_CAMERA :
        {
          CameraSharedPtr const& c = m_object.staticCast<Camera>();
          for ( Camera::HeadLightIterator it = c->beginHeadLights() ; it != c->endHeadLights() ; ++it )
          {
            objects.insert( *it );
          }
        }
        break;
      case ObjectCode::PRIMITIVE :
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
      case ObjectCode::PARAMETER_GROUP_DATA :
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
      case ObjectCode::PIPELINE_DATA :
        {
          dp::sg::core::PipelineDataSharedPtr const& pd = m_object.staticCast<dp::sg::core::PipelineData>();
          dp::fx::EffectSpecSharedPtr const & es = pd->getEffectSpec();
          for ( dp::fx::EffectSpec::iterator it = es->beginParameterGroupSpecs() ; it != es->endParameterGroupSpecs() ; ++it )
          {
            if ( pd->getParameterGroupData( it ) )
            {
              objects.insert( pd->getParameterGroupData( it ) );
            }
          }
        }
        break;
      case ObjectCode::SCENE :
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
      case ObjectCode::INVALID:
      case ObjectCode::CLIP_PLANE :
      case ObjectCode::VERTEX_ATTRIBUTE_SET :
      case ObjectCode::INDEX_SET :
      case ObjectCode::SAMPLER :
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
