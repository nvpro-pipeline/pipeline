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


#include "CommandReplaceItem.h"
#include "Viewer.h"
#include <dp/sg/core/Camera.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/ParameterGroupData.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Scene.h>

using namespace dp::sg::core;

CommandReplaceItem::CommandReplaceItem( SceneTreeItem * parent, SceneTreeItem * oldChild, SceneTreeItem * newChild, dp::util::Observer * observer )
  : ViewerCommand( ViewerCommand::UpdateFlag::SCENE_TREE )
  , m_parent(parent)
  , m_oldChild(oldChild)
  , m_newChild(newChild)
  , m_holdsOldChild(false)
  , m_observer(observer)
{
  setText( "Replace " + QString( objectCodeToName( oldChild->getObject()->getObjectCode() ).c_str() )
         + " with "   + QString( objectCodeToName( newChild->getObject()->getObjectCode() ).c_str() ) );
}

CommandReplaceItem::~CommandReplaceItem()
{
  delete ( m_holdsOldChild ? m_oldChild : m_newChild );
}

bool CommandReplaceItem::doRedo()
{
  return( doReplace( m_oldChild, m_newChild ) );
}

bool CommandReplaceItem::doUndo()
{
  return( doReplace( m_newChild, m_oldChild ) );
}

bool CommandReplaceItem::doReplace( SceneTreeItem * oldChild, SceneTreeItem * newChild )
{
  switch( m_parent->getObject()->getObjectCode() )
  {
    case ObjectCode::GEO_NODE :
      DP_ASSERT(std::static_pointer_cast<GeoNode>(m_parent->getObject())->getMaterialPipeline() == std::static_pointer_cast<dp::sg::core::PipelineData>(oldChild->getObject()));
      if ( std::dynamic_pointer_cast<dp::sg::core::PipelineData>(newChild->getObject()) )
      {
        std::static_pointer_cast<GeoNode>(m_parent->getObject())->setMaterialPipeline(std::static_pointer_cast<dp::sg::core::PipelineData>(newChild->getObject()));
      }
      else
      {
        DP_ASSERT( std::dynamic_pointer_cast<Primitive>(m_newChild->getObject()) );
        std::static_pointer_cast<GeoNode>(m_parent->getObject())->setPrimitive(std::static_pointer_cast<Primitive>(newChild->getObject()));
      }
      break;
    case ObjectCode::GROUP :
    case ObjectCode::LOD :
    case ObjectCode::SWITCH :
    case ObjectCode::TRANSFORM :
    case ObjectCode::BILLBOARD :
      {
        GroupSharedPtr g = std::static_pointer_cast<Group>(m_parent->getObject());
        if ( std::dynamic_pointer_cast<Node>(newChild->getObject()) )
        {
          DP_ASSERT(g->findChild(g->beginChildren(), std::static_pointer_cast<Node>(oldChild->getObject())) != g->endChildren());
          g->replaceChild(std::static_pointer_cast<Node>(newChild->getObject()), std::static_pointer_cast<Node>(oldChild->getObject()));
        }
        else
        {
          DP_ASSERT( std::dynamic_pointer_cast<ClipPlane>(newChild->getObject()) );
          DP_ASSERT( g->findClipPlane( std::static_pointer_cast<ClipPlane>(oldChild->getObject()) ) != g->endClipPlanes() );
          g->removeClipPlane(std::static_pointer_cast<ClipPlane>(oldChild->getObject()));
          g->addClipPlane(std::static_pointer_cast<ClipPlane>(newChild->getObject()));
        }
      }
      break;
    case ObjectCode::PRIMITIVE :
      if ( std::dynamic_pointer_cast<IndexSet>(newChild->getObject()) )
      {
        std::static_pointer_cast<Primitive>(m_parent->getObject())->setIndexSet(std::static_pointer_cast<IndexSet>(newChild->getObject()));
      }
      else
      {
        DP_ASSERT( std::dynamic_pointer_cast<VertexAttributeSet>(newChild->getObject()) );
        std::static_pointer_cast<Primitive>(m_parent->getObject())->setVertexAttributeSet(std::static_pointer_cast<VertexAttributeSet>(newChild->getObject()));
      }
      break;
    case ObjectCode::PARAMETER_GROUP_DATA :
      DP_ASSERT( newChild->getObject()->getObjectCode() == ObjectCode::SAMPLER );
      {
        ParameterGroupDataSharedPtr pgd = std::static_pointer_cast<ParameterGroupData>(m_parent->getObject());
        const dp::fx::ParameterGroupSpecSharedPtr & pgs = pgd->getParameterGroupSpec();
        dp::fx::ParameterGroupSpec::iterator it = pgs->findParameterSpec(std::static_pointer_cast<Sampler>(newChild->getObject())->getName());
        DP_ASSERT( it != pgs->endParameterSpecs() );
        DP_ASSERT(pgs->findParameterSpec(std::static_pointer_cast<Sampler>(oldChild->getObject())->getName()) == it);
        pgd->setParameter(it, std::static_pointer_cast<Sampler>(newChild->getObject()));
      }
      break;
    case ObjectCode::PARALLEL_CAMERA :
    case ObjectCode::PERSPECTIVE_CAMERA :
    case ObjectCode::MATRIX_CAMERA :
      DP_ASSERT( m_newChild->getObject()->getObjectCode() == ObjectCode::LIGHT_SOURCE );
      std::static_pointer_cast<Camera>(m_parent->getObject())->replaceHeadLight(std::static_pointer_cast<LightSource>(newChild->getObject()), std::static_pointer_cast<LightSource>(oldChild->getObject()));
      break;
    case ObjectCode::PIPELINE_DATA :
      DP_ASSERT( std::dynamic_pointer_cast<ParameterGroupData>(newChild->getObject()) );
      DP_ASSERT(std::static_pointer_cast<ParameterGroupData>(newChild->getObject())->getParameterGroupSpec() == std::static_pointer_cast<ParameterGroupData>(oldChild->getObject())->getParameterGroupSpec());
      std::static_pointer_cast<dp::sg::core::PipelineData>(m_parent->getObject())->setParameterGroupData(std::static_pointer_cast<ParameterGroupData>(newChild->getObject()));
      break;
    case ObjectCode::SCENE :
      if ( std::dynamic_pointer_cast<Camera>(m_newChild->getObject()) )
      {
        SceneSharedPtr s = std::static_pointer_cast<Scene>(m_parent->getObject());
        s->removeCamera(std::static_pointer_cast<Camera>(oldChild->getObject()));
        s->addCamera(std::static_pointer_cast<Camera>(newChild->getObject()));
      }
      else
      {
        DP_ASSERT( std::dynamic_pointer_cast<Node>(newChild->getObject()) );
        std::static_pointer_cast<Scene>(m_parent->getObject())->setRootNode(std::static_pointer_cast<Node>(newChild->getObject()));
      }
      break;
    default :
      DP_ASSERT( false );
      break;
  }

  if ( m_parent->isExpanded() )
  {
    int index = m_parent->indexOfChild( oldChild );
    DP_ASSERT( 0 <= index );
    m_parent->takeChild( index );
    m_parent->insertChild( index, newChild );
    m_holdsOldChild = !m_holdsOldChild;
  }
  m_parent->setChildIndicatorPolicy();
  GetApp()->outputStatistics();

  ObjectSharedPtr s = std::static_pointer_cast<Object>(oldChild->getObject());
  if ( s->isAttached( m_observer ) )
  {
    s->detach( m_observer );
    std::static_pointer_cast<Object>(newChild->getObject())->attach(m_observer);
  }

  return true;
}

