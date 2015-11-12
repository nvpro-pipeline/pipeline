// Copyright (c) 2013-2015, NVIDIA CORPORATION. All rights reserved.
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
  : ViewerCommand( ViewerCommand::UPDATE_SCENE_TREE )
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
    case OC_GEONODE :
      DP_ASSERT( m_parent->getObject().staticCast<GeoNode>()->getMaterialPipeline() == oldChild->getObject().staticCast<dp::sg::core::PipelineData>() );
      if ( newChild->getObject().isPtrTo<dp::sg::core::PipelineData>() )
      {
        m_parent->getObject().staticCast<GeoNode>()->setMaterialPipeline( newChild->getObject().staticCast<dp::sg::core::PipelineData>() );
      }
      else
      {
        DP_ASSERT( m_newChild->getObject().isPtrTo<Primitive>() );
        m_parent->getObject().staticCast<GeoNode>()->setPrimitive( newChild->getObject().staticCast<Primitive>() );
      }
      break;
    case OC_GROUP :
    case OC_LOD :
    case OC_SWITCH :
    case OC_TRANSFORM :
    case OC_BILLBOARD :
      {
        GroupSharedPtr const& g = m_parent->getObject().staticCast<Group>();
        if ( newChild->getObject().isPtrTo<Node>() )
        {
          DP_ASSERT( g->findChild( g->beginChildren(), oldChild->getObject().staticCast<Node>() ) != g->endChildren() );
          g->replaceChild( newChild->getObject().staticCast<Node>(), oldChild->getObject().staticCast<Node>() );
        }
        else
        {
          DP_ASSERT( newChild->getObject().isPtrTo<ClipPlane>() );
          DP_ASSERT( g->findClipPlane( oldChild->getObject().staticCast<ClipPlane>() ) != g->endClipPlanes() );
          g->removeClipPlane( oldChild->getObject().staticCast<ClipPlane>() );
          g->addClipPlane( newChild->getObject().staticCast<ClipPlane>() );
        }
      }
      break;
    case OC_PRIMITIVE :
      if ( newChild->getObject().isPtrTo<IndexSet>() )
      {
        m_parent->getObject().staticCast<Primitive>()->setIndexSet( newChild->getObject().staticCast<IndexSet>() );
      }
      else
      {
        DP_ASSERT( newChild->getObject().isPtrTo<VertexAttributeSet>() );
        m_parent->getObject().staticCast<Primitive>()->setVertexAttributeSet( newChild->getObject().staticCast<VertexAttributeSet>() );
      }
      break;
    case OC_PARAMETER_GROUP_DATA :
      DP_ASSERT( newChild->getObject()->getObjectCode() == OC_SAMPLER );
      {
        ParameterGroupDataSharedPtr const& pgd = m_parent->getObject().staticCast<ParameterGroupData>();
        const dp::fx::ParameterGroupSpecSharedPtr & pgs = pgd->getParameterGroupSpec();
        dp::fx::ParameterGroupSpec::iterator it = pgs->findParameterSpec( newChild->getObject().staticCast<Sampler>()->getName() );
        DP_ASSERT( it != pgs->endParameterSpecs() );
        DP_ASSERT( pgs->findParameterSpec( oldChild->getObject().staticCast<Sampler>()->getName() ) == it );
        pgd->setParameter( it, newChild->getObject().staticCast<Sampler>() );
      }
      break;
    case OC_PARALLELCAMERA :
    case OC_PERSPECTIVECAMERA :
    case OC_MATRIXCAMERA :
      DP_ASSERT( m_newChild->getObject()->getObjectCode() == OC_LIGHT_SOURCE );
      m_parent->getObject().staticCast<Camera>()->replaceHeadLight( newChild->getObject().staticCast<LightSource>(), oldChild->getObject().staticCast<LightSource>() );
      break;
    case OC_PIPELINE_DATA :
      DP_ASSERT( newChild->getObject().isPtrTo<ParameterGroupData>() );
      DP_ASSERT( newChild->getObject().staticCast<ParameterGroupData>()->getParameterGroupSpec() == oldChild->getObject().staticCast<ParameterGroupData>()->getParameterGroupSpec() );
      m_parent->getObject().staticCast<dp::sg::core::PipelineData>()->setParameterGroupData( newChild->getObject().staticCast<ParameterGroupData>() );
      break;
    case OC_SCENE :
      if ( m_newChild->getObject().isPtrTo<Camera>() )
      {
        SceneSharedPtr const& s = m_parent->getObject().staticCast<Scene>();
        s->removeCamera( oldChild->getObject().staticCast<Camera>() );
        s->addCamera( newChild->getObject().staticCast<Camera>() );
      }
      else
      {
        DP_ASSERT( newChild->getObject().isPtrTo<Node>() );
        m_parent->getObject().staticCast<Scene>()->setRootNode( newChild->getObject().staticCast<Node>() );
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

  ObjectSharedPtr const& s = oldChild->getObject().staticCast<Object>();
  if ( s->isAttached( m_observer ) )
  {
    s->detach( m_observer );
    newChild->getObject().staticCast<Object>()->attach( m_observer );
  }

  return true;
}

