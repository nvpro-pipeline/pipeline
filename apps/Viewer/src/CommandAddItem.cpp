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


#include "CommandAddItem.h"
#include "Viewer.h"
#include <dp/sg/core/Camera.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/Scene.h>

QString makeText( bool add, dp::sg::core::ObjectSharedPtr const& parent, dp::sg::core::ObjectSharedPtr const& child );
bool add( dp::sg::core::ObjectSharedPtr const& parent, dp::sg::core::ObjectSharedPtr const& child );
bool remove( dp::sg::core::ObjectSharedPtr const& parent, dp::sg::core::ObjectSharedPtr const& child );


CommandAddItem::CommandAddItem( SceneTreeItem * parent, SceneTreeItem * child, bool add )
  : ViewerCommand( ViewerCommand::UPDATE_SCENE_TREE )
  , m_add(add)
  , m_parent(parent)
  , m_child(child)
  , m_childTaken(add)
{
  setText( makeText( m_add, m_parent->getObject(), m_child->getObject() ) );
}

CommandAddItem::~CommandAddItem()
{
  if ( m_childTaken )
  {
    delete m_child;
  }
}

void CommandAddItem::adjustItems( bool add )
{
  if ( m_parent->isExpanded() )
  {
    if ( add )
    {
      DP_ASSERT( m_childTaken );
      m_parent->addChild( m_child );
      m_childTaken = false;
    }
    else
    {
      DP_ASSERT( !m_childTaken );
      m_parent->takeChild( m_parent->indexOfChild( m_child ) );
      m_childTaken = true;
    }
  }
  m_parent->setChildIndicatorPolicy();
}

bool CommandAddItem::doRedo()
{
  adjustItems( m_add );
  return( m_add ? add( m_parent->getObject(), m_child->getObject() ) : remove( m_parent->getObject(), m_child->getObject() ) );
}

bool CommandAddItem::doUndo()
{
  // undoing an "add" is equivalent with removing
  // undoing an "remove" is equivalent with adding
  adjustItems( !m_add );
  return( m_add ? remove( m_parent->getObject(), m_child->getObject() ) : add( m_parent->getObject(), m_child->getObject() ) );
}


CommandAddObject::CommandAddObject( dp::sg::core::ObjectSharedPtr const& parent, dp::sg::core::ObjectSharedPtr const& child, bool add )
  : ViewerCommand( ViewerCommand::UPDATE_SCENE_TREE )
  , m_add(add)
  , m_parent(parent)
  , m_child(child)
{
  setText( makeText( m_add, parent, child ) );
}

CommandAddObject::~CommandAddObject()
{
}

bool CommandAddObject::doRedo()
{
  return( m_add ? add( m_parent, m_child ) : remove( m_parent, m_child ) );
}

bool CommandAddObject::doUndo()
{
  return( m_add ? remove( m_parent, m_child ) : add( m_parent, m_child ) );
}

QString makeText( bool add, dp::sg::core::ObjectSharedPtr const& parent, dp::sg::core::ObjectSharedPtr const& child )
{
  if ( add )
  {
    return( "Add " + QString( dp::sg::core::objectCodeToName( child->getObjectCode() ).c_str() )
          + " To " + QString( dp::sg::core::objectCodeToName( parent->getObjectCode() ).c_str() ) );
  }
  else
  {
    return( "Remove " + QString( dp::sg::core::objectCodeToName( child->getObjectCode() ).c_str() )
          + " From "  + QString( dp::sg::core::objectCodeToName( parent->getObjectCode() ).c_str() ) );
  }
}

bool add( dp::sg::core::ObjectSharedPtr const& parent, dp::sg::core::ObjectSharedPtr const& child )
{
  switch( parent->getObjectCode() )
  {
    case dp::sg::core::OC_GEONODE :
      if ( child.isPtrTo<dp::sg::core::EffectData>() )
      {
        parent.staticCast<dp::sg::core::GeoNode>()->setMaterialEffect( child.staticCast<dp::sg::core::EffectData>() );
      }
      else
      {
        DP_ASSERT( child.isPtrTo<dp::sg::core::Primitive>() );
        parent.staticCast<dp::sg::core::GeoNode>()->setPrimitive( child.staticCast<dp::sg::core::Primitive>() );
      }
      break;
    case dp::sg::core::OC_GROUP :
    case dp::sg::core::OC_LOD :
    case dp::sg::core::OC_SWITCH :
    case dp::sg::core::OC_TRANSFORM :
    case dp::sg::core::OC_BILLBOARD :
      if ( child.isPtrTo<dp::sg::core::Node>() )
      {
        parent.staticCast<dp::sg::core::Group>()->addChild( child.staticCast<dp::sg::core::Node>() );
      }
      else
      {
        DP_ASSERT( child.isPtrTo<dp::sg::core::ClipPlane>() );
        parent.staticCast<dp::sg::core::Group>()->addClipPlane( child.staticCast<dp::sg::core::ClipPlane>() );
      }
      break;
    case dp::sg::core::OC_PRIMITIVE :
      if ( child.isPtrTo<dp::sg::core::IndexSet>() )
      {
        parent.staticCast<dp::sg::core::Primitive>()->setIndexSet( child.staticCast<dp::sg::core::IndexSet>() );
      }
      else
      {
        DP_ASSERT( child.isPtrTo<dp::sg::core::VertexAttributeSet>() );
        parent.staticCast<dp::sg::core::Primitive>()->setVertexAttributeSet( child.staticCast<dp::sg::core::VertexAttributeSet>() );
      }
      break;
    case dp::sg::core::OC_EFFECT_DATA :
      DP_ASSERT( child.isPtrTo<dp::sg::core::ParameterGroupData>() );
      parent.staticCast<dp::sg::core::EffectData>()->setParameterGroupData( child.staticCast<dp::sg::core::ParameterGroupData>() );
      break;
    case dp::sg::core::OC_PARAMETER_GROUP_DATA :
      DP_ASSERT( child->getObjectCode() == dp::sg::core::OC_SAMPLER );
      {
        dp::sg::core::ParameterGroupDataSharedPtr const& pgd = parent.staticCast<dp::sg::core::ParameterGroupData>();
        const dp::fx::SmartParameterGroupSpec & pgs = pgd->getParameterGroupSpec();
        dp::fx::ParameterGroupSpec::iterator it = pgs->findParameterSpec( child.staticCast<dp::sg::core::Sampler>()->getName() );
        DP_ASSERT( it != pgs->endParameterSpecs() );
        pgd->setParameter( it, child.staticCast<dp::sg::core::Sampler>() );
      }
      break;
    case dp::sg::core::OC_PARALLELCAMERA :
    case dp::sg::core::OC_PERSPECTIVECAMERA :
    case dp::sg::core::OC_MATRIXCAMERA :
      DP_ASSERT( child->getObjectCode() == dp::sg::core::OC_LIGHT_SOURCE );
      parent.staticCast<dp::sg::core::Camera>()->addHeadLight( child.staticCast<dp::sg::core::LightSource>() );
      break;
    case dp::sg::core::OC_SCENE :
      if ( child.isPtrTo<dp::sg::core::Camera>() )
      {
        parent.staticCast<dp::sg::core::Scene>()->addCamera( child.staticCast<dp::sg::core::Camera>() );
      }
      else
      {
        DP_ASSERT( child.isPtrTo<dp::sg::core::Node>() );
        parent.staticCast<dp::sg::core::Scene>()->setRootNode( child.staticCast<dp::sg::core::Node>() );
      }
      break;
    default :
      DP_ASSERT( false );
      break;
  }

  GetApp()->outputStatistics();

  return true;
}

bool remove( dp::sg::core::ObjectSharedPtr const& parent, dp::sg::core::ObjectSharedPtr const& child )
{
  switch( parent->getObjectCode() )
  {
    case dp::sg::core::OC_GEONODE :
      if ( child.isPtrTo<dp::sg::core::EffectData>() )
      {
        parent.staticCast<dp::sg::core::GeoNode>()->setMaterialEffect( dp::sg::core::EffectDataSharedPtr() );
      }
      else
      {
        DP_ASSERT( child.isPtrTo<dp::sg::core::Primitive>() );
        parent.staticCast<dp::sg::core::GeoNode>()->setPrimitive( dp::sg::core::PrimitiveSharedPtr() );
      }
      break;
    case dp::sg::core::OC_GROUP :
    case dp::sg::core::OC_LOD :
    case dp::sg::core::OC_SWITCH :
    case dp::sg::core::OC_TRANSFORM :
    case dp::sg::core::OC_BILLBOARD :
      if ( child.isPtrTo<dp::sg::core::Node>() )
      {
        parent.staticCast<dp::sg::core::Group>()->removeChild( child.staticCast<dp::sg::core::Node>() );
      }
      else
      {
        DP_ASSERT( child.isPtrTo<dp::sg::core::ClipPlane>() );
        parent.staticCast<dp::sg::core::Group>()->removeClipPlane( child.staticCast<dp::sg::core::ClipPlane>() );
      }
      break;
    case dp::sg::core::OC_PRIMITIVE :
      if ( child.isPtrTo<dp::sg::core::IndexSet>() )
      {
        parent.staticCast<dp::sg::core::Primitive>()->setIndexSet( dp::sg::core::IndexSetSharedPtr() );
      }
      else
      {
        DP_ASSERT( child.isPtrTo<dp::sg::core::VertexAttributeSet>() );
        parent.staticCast<dp::sg::core::Primitive>()->setVertexAttributeSet( dp::sg::core::VertexAttributeSetSharedPtr() );
      }
      break;
    case dp::sg::core::OC_EFFECT_DATA :
      DP_ASSERT( child.isPtrTo<dp::sg::core::ParameterGroupData>() );
      {
        dp::sg::core::ParameterGroupDataSharedPtr const& pgd = child.staticCast<dp::sg::core::ParameterGroupData>();
        dp::sg::core::EffectDataSharedPtr const& ed = parent.staticCast<dp::sg::core::EffectData>();
        dp::fx::SmartEffectSpec const & es = ed->getEffectSpec();
        DP_ASSERT( es->findParameterGroupSpec( pgd->getParameterGroupSpec() ) != es->endParameterGroupSpecs() );
        parent.staticCast<dp::sg::core::EffectData>()->setParameterGroupData( es->findParameterGroupSpec( pgd->getParameterGroupSpec() ), dp::sg::core::ParameterGroupDataSharedPtr() );
      }
      break;
    case dp::sg::core::OC_PARAMETER_GROUP_DATA :
      DP_ASSERT( child->getObjectCode() == dp::sg::core::OC_SAMPLER );
      {
        dp::sg::core::ParameterGroupDataSharedPtr const& pgd = parent.staticCast<dp::sg::core::ParameterGroupData>();
        const dp::fx::SmartParameterGroupSpec & pgs = pgd->getParameterGroupSpec();
        dp::fx::ParameterGroupSpec::iterator it = pgs->findParameterSpec( child.staticCast<dp::sg::core::Sampler>()->getName() );
        DP_ASSERT( it != pgs->endParameterSpecs() );
        pgd->setParameter( it, dp::sg::core::SamplerSharedPtr() );
      }
      break;
    case dp::sg::core::OC_PARALLELCAMERA :
    case dp::sg::core::OC_PERSPECTIVECAMERA :
    case dp::sg::core::OC_MATRIXCAMERA :
      DP_ASSERT( child->getObjectCode() == dp::sg::core::OC_LIGHT_SOURCE );
      parent.staticCast<dp::sg::core::Camera>()->removeHeadLight( child.staticCast<dp::sg::core::LightSource>() );
      break;
    case dp::sg::core::OC_SCENE :
      if ( child.isPtrTo<dp::sg::core::Camera>() )
      {
        DP_VERIFY( parent.staticCast<dp::sg::core::Scene>()->removeCamera( child.staticCast<dp::sg::core::Camera>() ) );
      }
      else
      {
        DP_ASSERT( child.isPtrTo<dp::sg::core::Node>() );
        parent.staticCast<dp::sg::core::Scene>()->setRootNode( dp::sg::core::NodeSharedPtr() );
      }
      break;
    default :
      DP_ASSERT( false );
      break;
  }

  GetApp()->outputStatistics();

  return true;
}

