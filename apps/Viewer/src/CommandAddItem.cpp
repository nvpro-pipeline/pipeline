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


#include "CommandAddItem.h"
#include "Viewer.h"
#include <dp/sg/core/Camera.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Scene.h>

QString makeText( bool add, dp::sg::core::ObjectSharedPtr const& parent, dp::sg::core::ObjectSharedPtr const& child );
bool add( dp::sg::core::ObjectSharedPtr const& parent, dp::sg::core::ObjectSharedPtr const& child );
bool remove( dp::sg::core::ObjectSharedPtr const& parent, dp::sg::core::ObjectSharedPtr const& child );


CommandAddItem::CommandAddItem( SceneTreeItem * parent, SceneTreeItem * child, bool add )
  : ViewerCommand( ViewerCommand::UpdateFlag::SCENE_TREE )
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
  : ViewerCommand( ViewerCommand::UpdateFlag::SCENE_TREE )
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
    case dp::sg::core::ObjectCode::GEO_NODE :
      if ( std::dynamic_pointer_cast<dp::sg::core::PipelineData>(child) )
      {
        std::static_pointer_cast<dp::sg::core::GeoNode>(parent)->setMaterialPipeline(std::static_pointer_cast<dp::sg::core::PipelineData>(child));
      }
      else
      {
        DP_ASSERT( std::dynamic_pointer_cast<dp::sg::core::Primitive>(child) );
        std::static_pointer_cast<dp::sg::core::GeoNode>(parent)->setPrimitive(std::static_pointer_cast<dp::sg::core::Primitive>(child));
      }
      break;
    case dp::sg::core::ObjectCode::GROUP :
    case dp::sg::core::ObjectCode::LOD :
    case dp::sg::core::ObjectCode::SWITCH :
    case dp::sg::core::ObjectCode::TRANSFORM :
    case dp::sg::core::ObjectCode::BILLBOARD :
      if ( std::dynamic_pointer_cast<dp::sg::core::Node>(child) )
      {
        std::static_pointer_cast<dp::sg::core::Group>(parent)->addChild(std::static_pointer_cast<dp::sg::core::Node>(child));
      }
      else
      {
        DP_ASSERT( std::dynamic_pointer_cast<dp::sg::core::ClipPlane>(child) );
        std::static_pointer_cast<dp::sg::core::Group>(parent)->addClipPlane(std::static_pointer_cast<dp::sg::core::ClipPlane>(child));
      }
      break;
    case dp::sg::core::ObjectCode::PRIMITIVE :
      if ( std::dynamic_pointer_cast<dp::sg::core::IndexSet>(child) )
      {
        std::static_pointer_cast<dp::sg::core::Primitive>(parent)->setIndexSet(std::static_pointer_cast<dp::sg::core::IndexSet>(child));
      }
      else
      {
        DP_ASSERT( std::dynamic_pointer_cast<dp::sg::core::VertexAttributeSet>(child) );
        std::static_pointer_cast<dp::sg::core::Primitive>(parent)->setVertexAttributeSet(std::static_pointer_cast<dp::sg::core::VertexAttributeSet>(child));
      }
      break;
    case dp::sg::core::ObjectCode::PARAMETER_GROUP_DATA :
      DP_ASSERT( child->getObjectCode() == dp::sg::core::ObjectCode::SAMPLER );
      {
        dp::sg::core::ParameterGroupDataSharedPtr pgd = std::static_pointer_cast<dp::sg::core::ParameterGroupData>(parent);
        const dp::fx::ParameterGroupSpecSharedPtr & pgs = pgd->getParameterGroupSpec();
        dp::fx::ParameterGroupSpec::iterator it = pgs->findParameterSpec(std::static_pointer_cast<dp::sg::core::Sampler>(child)->getName());
        DP_ASSERT( it != pgs->endParameterSpecs() );
        pgd->setParameter(it, std::static_pointer_cast<dp::sg::core::Sampler>(child));
      }
      break;
    case dp::sg::core::ObjectCode::PARALLEL_CAMERA :
    case dp::sg::core::ObjectCode::PERSPECTIVE_CAMERA :
    case dp::sg::core::ObjectCode::MATRIX_CAMERA :
      DP_ASSERT( child->getObjectCode() == dp::sg::core::ObjectCode::LIGHT_SOURCE );
      std::static_pointer_cast<dp::sg::core::Camera>(parent)->addHeadLight(std::static_pointer_cast<dp::sg::core::LightSource>(child));
      break;
    case dp::sg::core::ObjectCode::PIPELINE_DATA :
      DP_ASSERT( std::dynamic_pointer_cast<dp::sg::core::ParameterGroupData>(child) );
      std::static_pointer_cast<dp::sg::core::PipelineData>(parent)->setParameterGroupData(std::static_pointer_cast<dp::sg::core::ParameterGroupData>(child));
      break;
    case dp::sg::core::ObjectCode::SCENE :
      if ( std::dynamic_pointer_cast<dp::sg::core::Camera>(child) )
      {
        std::static_pointer_cast<dp::sg::core::Scene>(parent)->addCamera(std::static_pointer_cast<dp::sg::core::Camera>(child));
      }
      else
      {
        DP_ASSERT( std::dynamic_pointer_cast<dp::sg::core::Node>(child) );
        std::static_pointer_cast<dp::sg::core::Scene>(parent)->setRootNode(std::static_pointer_cast<dp::sg::core::Node>(child));
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
    case dp::sg::core::ObjectCode::GEO_NODE :
      if ( std::dynamic_pointer_cast<dp::sg::core::PipelineData>(child) )
      {
        std::static_pointer_cast<dp::sg::core::GeoNode>(parent)->setMaterialPipeline(dp::sg::core::PipelineDataSharedPtr());
      }
      else
      {
        DP_ASSERT( std::dynamic_pointer_cast<dp::sg::core::Primitive>(child) );
        std::static_pointer_cast<dp::sg::core::GeoNode>(parent)->setPrimitive(dp::sg::core::PrimitiveSharedPtr());
      }
      break;
    case dp::sg::core::ObjectCode::GROUP :
    case dp::sg::core::ObjectCode::LOD :
    case dp::sg::core::ObjectCode::SWITCH :
    case dp::sg::core::ObjectCode::TRANSFORM :
    case dp::sg::core::ObjectCode::BILLBOARD :
      if ( std::dynamic_pointer_cast<dp::sg::core::Node>(child) )
      {
        std::static_pointer_cast<dp::sg::core::Group>(parent)->removeChild(std::static_pointer_cast<dp::sg::core::Node>(child));
      }
      else
      {
        DP_ASSERT( std::dynamic_pointer_cast<dp::sg::core::ClipPlane>(child) );
        std::static_pointer_cast<dp::sg::core::Group>(parent)->removeClipPlane(std::static_pointer_cast<dp::sg::core::ClipPlane>(child));
      }
      break;
    case dp::sg::core::ObjectCode::PRIMITIVE :
      if ( std::dynamic_pointer_cast<dp::sg::core::IndexSet>(child) )
      {
        std::static_pointer_cast<dp::sg::core::Primitive>(parent)->setIndexSet(dp::sg::core::IndexSetSharedPtr());
      }
      else
      {
        DP_ASSERT( std::dynamic_pointer_cast<dp::sg::core::VertexAttributeSet>(child) );
        std::static_pointer_cast<dp::sg::core::Primitive>(parent)->setVertexAttributeSet(dp::sg::core::VertexAttributeSetSharedPtr());
      }
      break;
    case dp::sg::core::ObjectCode::PARAMETER_GROUP_DATA :
      DP_ASSERT( child->getObjectCode() == dp::sg::core::ObjectCode::SAMPLER );
      {
        dp::sg::core::ParameterGroupDataSharedPtr pgd = std::static_pointer_cast<dp::sg::core::ParameterGroupData>(parent);
        const dp::fx::ParameterGroupSpecSharedPtr & pgs = pgd->getParameterGroupSpec();
        dp::fx::ParameterGroupSpec::iterator it = pgs->findParameterSpec(std::static_pointer_cast<dp::sg::core::Sampler>(child)->getName());
        DP_ASSERT( it != pgs->endParameterSpecs() );
        pgd->setParameter( it, dp::sg::core::SamplerSharedPtr() );
      }
      break;
    case dp::sg::core::ObjectCode::PARALLEL_CAMERA :
    case dp::sg::core::ObjectCode::PERSPECTIVE_CAMERA :
    case dp::sg::core::ObjectCode::MATRIX_CAMERA :
      DP_ASSERT( child->getObjectCode() == dp::sg::core::ObjectCode::LIGHT_SOURCE );
      std::static_pointer_cast<dp::sg::core::Camera>(parent)->removeHeadLight(std::static_pointer_cast<dp::sg::core::LightSource>(child));
      break;
    case dp::sg::core::ObjectCode::PIPELINE_DATA :
      DP_ASSERT( std::dynamic_pointer_cast<dp::sg::core::ParameterGroupData>(child) );
      {
        dp::sg::core::ParameterGroupDataSharedPtr pgd = std::static_pointer_cast<dp::sg::core::ParameterGroupData>(child);
        dp::sg::core::PipelineDataSharedPtr pd = std::static_pointer_cast<dp::sg::core::PipelineData>(parent);
        dp::fx::EffectSpecSharedPtr const & es = pd->getEffectSpec();
        DP_ASSERT( es->findParameterGroupSpec( pgd->getParameterGroupSpec() ) != es->endParameterGroupSpecs() );
        std::static_pointer_cast<dp::sg::core::PipelineData>(parent)->setParameterGroupData(es->findParameterGroupSpec(pgd->getParameterGroupSpec()), dp::sg::core::ParameterGroupDataSharedPtr());
      }
      break;
    case dp::sg::core::ObjectCode::SCENE :
      if ( std::dynamic_pointer_cast<dp::sg::core::Camera>(child) )
      {
        DP_VERIFY(std::static_pointer_cast<dp::sg::core::Scene>(parent)->removeCamera(std::static_pointer_cast<dp::sg::core::Camera>(child)));
      }
      else
      {
        DP_ASSERT( std::dynamic_pointer_cast<dp::sg::core::Node>(child) );
        std::static_pointer_cast<dp::sg::core::Scene>(parent)->setRootNode(dp::sg::core::NodeSharedPtr());
      }
      break;
    default :
      DP_ASSERT( false );
      break;
  }

  GetApp()->outputStatistics();

  return true;
}

