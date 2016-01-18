// Copyright (c) 2002-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/IndexSet.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/algorithm/EliminateTraverser.h>

using namespace dp::sg::core;

using std::pair;
using std::set;
using std::vector;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      DEFINE_STATIC_PROPERTY( EliminateTraverser, EliminateTargets );

      BEGIN_REFLECTION_INFO( EliminateTraverser )
        DERIVE_STATIC_PROPERTIES( EliminateTraverser, OptimizeTraverser );
        INIT_STATIC_PROPERTY_RW( EliminateTraverser, EliminateTargets, EliminateTraverser::TargetMask, Semantic::VALUE, value, value );
      END_REFLECTION_INFO

      EliminateTraverser::EliminateTraverser( void )
      : m_eliminateTargets(Target::ALL)
      {
      }

      EliminateTraverser::~EliminateTraverser( void )
      {
      }

      void EliminateTraverser::doApply( const NodeSharedPtr & root )
      {
        DP_ASSERT(root);

        OptimizeTraverser::doApply( root );

        NodeSharedPtr oldRoot(root);
        {
          if (  ( ( m_eliminateTargets & Target::LOD )                                    && ( oldRoot.isPtrTo<LOD>() ) )
            ||  ( ( m_eliminateTargets & ( Target::GROUP | Target::GROUP_SINGLE_CHILD ) ) && ( oldRoot.isPtrTo<Group>() ) ) )
          {
            GroupSharedPtr const& group = oldRoot.staticCast<Group>();
            if ( m_scene &&  isOneChildCandidate( group ) ) // Apply this optimization only if we have valid scene handle,
                                                            // and root, as a group node, has only one child
            {
              NodeSharedPtr newRoot = *group->beginChildren();
              m_scene->setRootNode( newRoot );
              m_root = newRoot;
            }
          }
        }
        m_objects.clear();
      }

      void EliminateTraverser::handleBillboard( Billboard *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleBillboard( p );
          if ( m_eliminateTargets & Target::GROUP )
          {
            eliminateGroups( dynamic_cast<Group *>( p ) );
          }
          else if ( m_eliminateTargets & Target::GROUP_SINGLE_CHILD )
          {
            eliminateSingleChildChildren( p, ObjectCode::GROUP );
          }
          if ( m_eliminateTargets & Target::LOD )
          {
            eliminateSingleChildChildren( p, ObjectCode::LOD );
          }
        }
      }

      void EliminateTraverser::handleGroup( Group *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleGroup( p );
          if ( m_eliminateTargets & Target::GROUP )
          {
            eliminateGroups( p );
          }
          else if ( m_eliminateTargets & Target::GROUP_SINGLE_CHILD )
          {
            eliminateSingleChildChildren( p, ObjectCode::GROUP );
          }
          if ( m_eliminateTargets & Target::LOD )
          {
            eliminateSingleChildChildren( p, ObjectCode::LOD );
          }
        }
      }

      void EliminateTraverser::handleLOD( LOD *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleLOD( p );
          if ( m_eliminateTargets & ( Target::GROUP | Target::GROUP_SINGLE_CHILD ) )
          {
            eliminateSingleChildChildren( dynamic_cast<Group *>( p ), ObjectCode::GROUP );
          }
          if ( m_eliminateTargets & Target::LOD )
          {
            eliminateSingleChildChildren( p, ObjectCode::LOD );
          }
        }
      }

      bool isFlatIndexSet( const IndexSetSharedPtr & is, unsigned int offset, unsigned int count )
      {
        IndexSet::ConstIterator<unsigned int> isci( is, offset );
        for ( unsigned int i=0, idx = offset ; i<count ; i++, idx++ )
        {
          if ( isci[i] != idx )
          {
            return( false );
          }
        }
        return( true );
      }

      void EliminateTraverser::handlePrimitive( Primitive * p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handlePrimitive( p );
          if (    optimizationAllowed( p->getSharedPtr<Primitive>() )
              &&  ( m_eliminateTargets & Target::INDEX_SET )
              &&  p->isIndexed()
              &&  ( p->getIndexSet()->getNumberOfIndices() == p->getVertexAttributeSet()->getNumberOfVertices() )
              &&  isFlatIndexSet( p->getIndexSet(), p->getElementOffset(), p->getElementCount() ) )
          {
            p->setIndexSet( IndexSetSharedPtr::null );
            setTreeModified();
          }
        }
      }

      void EliminateTraverser::handleSwitch( Switch *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleSwitch( p );
          if ( m_eliminateTargets & ( Target::GROUP | Target::GROUP_SINGLE_CHILD ) )
          {
            eliminateSingleChildChildren( dynamic_cast<Group *>( p ), ObjectCode::GROUP );
          }
          if ( m_eliminateTargets & Target::LOD )
          {
            eliminateSingleChildChildren( p, ObjectCode::LOD );
          }
        }
      }

      void EliminateTraverser::handleTransform( Transform *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleTransform( p );
          if ( m_eliminateTargets & Target::GROUP )
          {
            eliminateGroups( dynamic_cast<Group *>( p ) );
          }
          else if ( m_eliminateTargets & Target::GROUP_SINGLE_CHILD )
          {
            eliminateSingleChildChildren( p, ObjectCode::GROUP );
          }
          if ( m_eliminateTargets & Target::LOD )
          {
            eliminateSingleChildChildren( p, ObjectCode::LOD );
          }
        }
      }

      void EliminateTraverser::eliminateGroups( Group *p )
      {
        std::set<GroupSharedPtr> groups;

        if( !optimizationAllowed( p->getSharedPtr<Group>() ) )
        {
          return;
        }

        for ( Group::ChildrenIterator gci = p->beginChildren() ; gci != p->endChildren() ; ++gci )
        {
          if ( gci->isPtrTo<Group>() )
          {
            bool isJoint = false;
            if( gci->isPtrTo<Transform>() )
            {
              isJoint = gci->staticCast<Transform>()->isJoint();
            }

            GroupSharedPtr const& group = gci->staticCast<Group>();
            if (    ( getIgnoreNames() || group->getName().empty() )        // only unnamed or if names are to be ignored
                &&  optimizationAllowed( group )                            // only if optimization is allowed
                &&  (   (   ( group->getObjectCode() == ObjectCode::GROUP )          // replace a Group (and only a Group)
                        &&  ( group->getNumberOfClipPlanes() == 0 ))        // - without clip planes
                    ||  (   ( group->getNumberOfChildren() == 0 )           // or remove a group derived without children
                        &&   !isJoint  )                                    // - if it's not a joint
                    )
                )
            {
              groups.insert( group );
            }
          }
        }

        if ( !groups.empty() )
        {
          vector<NodeSharedPtr> newChildren;

          for ( Group::ChildrenIterator gci = p->beginChildren() ; gci != p->endChildren() ; ++gci )
          {
            if ( !gci->isPtrTo<Group>() || ( groups.find( gci->inplaceCast<dp::sg::core::Group>() ) == groups.end() ) )
            {
              newChildren.push_back( *gci );
            }
            else
            {
              DP_ASSERT( gci->isPtrTo<Group>() );
              GroupSharedPtr const& group = gci->staticCast<Group>();
              for ( Group::ChildrenIterator grandChild = group->beginChildren() ; grandChild != group->endChildren() ; ++grandChild )
              {
                (*grandChild)->addHints( group->getHints() );
                newChildren.push_back( *grandChild );
              }
            }
          }
          p->clearChildren();
          for ( unsigned int i=0 ; i<newChildren.size() ; i++ )
          {
            p->addChild( newChildren[i] );
          }
          setTreeModified();
        }
      }

      void EliminateTraverser::eliminateSingleChildChildren( Group *p, dp::sg::core::ObjectCode objectCode )
      {
        vector<GroupSharedPtr> groups;

        if( !optimizationAllowed( p->getSharedPtr<Group>() ) )
        {
          return;
        }

        for ( Group::ChildrenIterator gci = p->beginChildren() ; gci != p->endChildren() ; ++gci )
        {
          if( gci->isPtrTo<Group>() )
          {
            GroupSharedPtr const& group = gci->staticCast<Group>();
            if ( group->getObjectCode() == objectCode )
            {
              if ( isOneChildCandidate( group ) )
              {
                groups.push_back( group );
              }
            }
          }
        }

        if ( groups.size() )
        {
          for ( vector<GroupSharedPtr>::iterator i=groups.begin() ; i!=groups.end() ; ++i )
          {
            // Need to use GroupLock on *i here, because itself will be replaced as child. During
            // replacement the child will be write-locked to remove its actual parent as parent. A
            // GroupLock would cause a deadlock in this special case!
            p->replaceChild( *(*i)->beginChildren(), *i );
          }
          setTreeModified();
        }
      }

      bool EliminateTraverser::isOneChildCandidate( GroupSharedPtr const& p )
      {
        return(   ( getIgnoreNames() || p->getName().empty() )
              &&  optimizationAllowed( p )
              &&  ( p->getNumberOfClipPlanes() == 0 )
              &&  ( p->getNumberOfChildren() == 1 ) );
      }

    } // namespace algorithm
  } // namespace sp
} // namespace dp
