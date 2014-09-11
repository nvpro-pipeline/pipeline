// Copyright NVIDIA Corporation 2010-2013
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


#include <dp/sg/xbar/DrawableManager.h>
#include <dp/sg/xbar/SceneTree.h>
#include <dp/sg/xbar/inc/UpdateTransformVisitor.h>
#include <dp/sg/xbar/inc/UpdateObjectVisitor.h>
#include <dp/sg/xbar/inc/SceneTreeGenerator.h>

// observers
#include <dp/sg/xbar/inc/GeoNodeObserver.h>
#include <dp/sg/xbar/inc/ObjectObserver.h>
#include <dp/sg/xbar/inc/SwitchObserver.h>
#include <dp/sg/xbar/inc/TransformObserver.h>
#include <dp/sg/xbar/inc/SceneObserver.h>


#include <dp/math/Boxnt.h>
#include <dp/math/Trafo.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/Camera.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/Node.h>
#include <dp/sg/core/Object.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>

#include <dp/util/FrameProfiler.h>

#include <algorithm>

using namespace dp::math;
using namespace dp::util;
using namespace dp::sg::core;

using namespace std;

namespace dp
{
  namespace sg
  {

    namespace xbar
    {

      SceneTree::SceneTree( dp::sg::core::SceneSharedPtr const & scene )
        : m_scene( scene )
        , m_rootNode( scene->getRootNode() )
        , m_dirty( false )
        , m_switchObserver( SwitchObserver::create() )
      {
        m_objectObserver = ObjectObserver::create( this );
        m_geoNodeObserver = GeoNodeObserver::create( this );
        m_transformObserver = TransformObserver::create( this );
        m_sceneObserver = SceneObserver::create( this );

        // push an identity transform onto the transform tree so objects without transforms get the identity
        TransformTreeNode TransformTreeSentinel;
        setIdentity( TransformTreeSentinel.m_worldMatrix );
        setIdentity( TransformTreeSentinel.m_localMatrix );
        TransformTreeIndex transformTreeSentinel = m_transformTree.insertNode( TransformTreeSentinel, ~0, ~0 );

        ObjectTreeNode objectTreeSentinel;
        objectTreeSentinel.m_clipPlaneGroup = ClipPlaneGroup::create();
        m_objectTreeSentinel = m_objectTree.insertNode( objectTreeSentinel, ~0, ~0 );

        m_objectTree[m_objectTreeSentinel].m_transformIndex = transformTreeSentinel;
        m_transformTree[transformTreeSentinel].m_objectTreeIndex = m_objectTreeSentinel;

        SceneTreeGenerator rlg( this );
        rlg.setCurrentTransformTreeData( transformTreeSentinel, ~0 );
        rlg.setCurrentObjectTreeData( m_objectTreeSentinel, ~0 );
        rlg.apply( scene );

        // root node is first child below sentinel
        m_objectTreeRootNode = m_objectTree[m_objectTreeSentinel].m_firstChild;
      }

      SceneTree::~SceneTree()
      {
        m_sceneObserver.reset();

        // remove all drawables from DrawableManager
        ObjectTreeIndex treeSize = ObjectTreeIndex(m_objectTree.size());
        for ( ObjectTreeIndex index = 0; index < treeSize;++index )
        {
          drawableInstanceRemove( index );
        }
      }

      SceneTreeSharedPtr SceneTree::create( SceneSharedPtr const & scene )
      {
        return new SceneTree( scene );
      }

      dp::sg::core::SceneSharedPtr const & SceneTree::getScene() const
      {
        return m_scene;
      }

      void SceneTree::addSubTree( const NodeWeakPtr& root, 
        ObjectTreeIndex parentIndex, ObjectTreeIndex leftSibling, 
        TransformTreeIndex parentTransform, TransformTreeIndex leftSiblingTransform )
      {
        SceneTreeGenerator rlg( this );

        rlg.setCurrentObjectTreeData( parentIndex, leftSibling );
        rlg.setCurrentTransformTreeData( parentTransform, leftSiblingTransform );

        rlg.apply( root->getSharedPtr<Node>() );
      }

      void SceneTree::replaceSubTree( const NodeWeakPtr &node, ObjectTreeIndex objectIndex )
      {
        ObjectTreeIndex objectParent = m_objectTree[objectIndex].m_parentIndex;

        DP_ASSERT( objectParent != ~0 );

        // search left ObjectTree sibling
        TransformTreeIndex leftSiblingTransformIndex = ~0;
        TransformTreeIndex parentTransformIndex = m_objectTree[objectParent].m_transformIndex;

        ObjectTreeIndex objectLeftSibling = ~0;
        ObjectTreeIndex currentIndex = m_objectTree[objectParent].m_firstChild;
        while ( currentIndex != objectIndex )
        {
          if( m_objectTree[currentIndex].m_transformIndex != parentTransformIndex )
          {
            leftSiblingTransformIndex = m_objectTree[currentIndex].m_transformIndex;
          }

          objectLeftSibling = currentIndex;
          currentIndex = m_objectTree[currentIndex].m_nextSibling;
        }

        // remove group in both trees
        removeObjectTreeIndex( objectIndex );

        // add group back to tree
        addSubTree( node, objectParent, objectLeftSibling, parentTransformIndex, leftSiblingTransformIndex );
      }

      void SceneTree::addRendererOptions( const dp::sg::ui::RendererOptionsSharedPtr& rendererOptions )
      {
      }

      void SceneTree::update( dp::sg::ui::ViewStateSharedPtr const& vs )
      {
        ObjectTreeIndexSet dirtyGeoNodes;
        m_geoNodeObserver->popDirtyGeoNodes(dirtyGeoNodes);
        for ( ObjectTreeIndexSet::const_iterator it = dirtyGeoNodes.begin(); it != dirtyGeoNodes.end(); ++it )
        {
          drawableInstanceUpdate( *it );
        }

        {
          dp::util::ProfileEntry p("Update TransformTree");
          updateTransformTree( vs );
        }

        {
          dp::util::ProfileEntry p("Update ObjectTree");
          updateObjectTree( vs );
        }
      }

      void SceneTree::updateTransformTree( dp::sg::ui::ViewStateSharedPtr const& vs )
      {
        //
        // first step: update node local information
        // 

        // update dirty transforms from transform observer
        {
          const TransformObserver::DirtyPayloads& cd = m_transformObserver->getDirtyPayloads();

          TransformObserver::DirtyPayloads::const_iterator it, it_end = cd.end();
          for( it = cd.begin(); it != it_end; ++it )
          {
            TransformTreeIndex index = (*it)->m_index;
            TransformTreeNode& node = m_transformTree[index];
            DP_ASSERT( node.m_transform != nullptr );

            const Trafo& t = node.m_transform->getTrafo();
            node.m_localMatrix = t.getMatrix();

            const Vec3f& s( t.getScaling() );
            node.setLocalBits( TransformTreeNode::ISMIRRORTRANSFORM, s[0]*s[1]*s[2] < 0.0f );

            m_transformTree.markDirty( index, TransformTreeNode::DEFAULT_DIRTY );

            // mark the transform's corresponding object tree node's bounding volume dirty
            DP_ASSERT( node.m_objectTreeIndex != ~0 );

            (*it)->m_dirty = false;
          }
          m_transformObserver->clearDirtyPayloads();
        }

        // update dynamic transforms
        {
          TransformTreeIndexSet::const_iterator it, it_end = m_dynamicTransformIndices.end();
          for( it=m_dynamicTransformIndices.begin(); it!=it_end; ++it )
          {
            TransformTreeIndex index = *it;
            TransformTreeNode& node = m_transformTree[index];

            if( node.m_transform )
            {
              Trafo t = node.m_transform->getTrafo();
              node.m_localMatrix = t.getMatrix();

              const Vec3f& s( t.getScaling() );
              node.setLocalBits( TransformTreeNode::ISMIRRORTRANSFORM, s[0]*s[1]*s[2] < 0.0f );
            }
            m_transformTree.markDirty( index, TransformTreeNode::DEFAULT_DIRTY );
          }
        }

        //
        // second step: update resulting node-world information
        // 

        m_changedTransforms.clear();
        UpdateTransformVisitor visitor( m_transformTree, *this, vs->getCamera(), m_changedTransforms );
        PreOrderTreeTraverser<TransformTree, UpdateTransformVisitor> traverser;

        traverser.processDirtyList( m_transformTree, visitor, TransformTreeNode::DEFAULT_DIRTY);
        m_transformTree.m_dirtyObjects.clear();
      }

      void SceneTree::updateObjectTree( dp::sg::ui::ViewStateSharedPtr const& vs )
      {   
        //
        // first step: update node-local information
        //

        // update dirty object hints & masks
        {
          ObjectObserver::NewCacheData cd;
          m_objectObserver->popNewCacheData( cd );

          ObjectObserver::NewCacheData::const_iterator it, it_end = cd.end();
          for( it=cd.begin(); it!=it_end; ++it )
          {
            ObjectTreeNode& node = m_objectTree[ it->first ];
            node.m_localHints = it->second.m_hints;
            node.m_localMask = it->second.m_mask;

            m_objectTree.markDirty( it->first, ObjectTreeNode::DEFAULT_DIRTY );
          }
        }

        // update dirty switch information
        ObjectTreeIndexSet dirtySwitches;
        m_switchObserver->popDirtySwitches( dirtySwitches );
        if( !dirtySwitches.empty() )
        {
          ObjectTreeIndexSet::iterator it, it_end = dirtySwitches.end();
          for( it=dirtySwitches.begin(); it!=it_end; ++it )
          {
            ObjectTreeIndex index = *it;

            SwitchWeakPtr swp = m_objectTree.m_switchNodes[ index ];
            DP_ASSERT( swp );

            ObjectTreeIndex childIndex = m_objectTree[index].m_firstChild;
            // counter for the i-th child
            size_t i = 0;

            while( childIndex != ~0 )
            {
              ObjectTreeNode& childNode = m_objectTree[childIndex];
              DP_ASSERT( childNode.m_parentIndex == index );

              bool newActive = swp->isActive( checked_cast<unsigned int>(i) );
              if ( childNode.m_localActive != newActive )
              {
                childNode.m_localActive = newActive;
                m_objectTree.markDirty( childIndex, ObjectTreeNode::DEFAULT_DIRTY );
              }

              childIndex = childNode.m_nextSibling;
              ++i;
            }  
          }
        }

        // update all lods
        if( !m_objectTree.m_LODs.empty() )
        {
          float scaleFactor = vs->getLODRangeScale();
          const Mat44f& worldToView = vs->getCamera()->getWorldToViewMatrix();

          std::map< ObjectTreeIndex, LODWeakPtr >::iterator it, it_end = m_objectTree.m_LODs.end();
          for( it = m_objectTree.m_LODs.begin(); it != it_end; ++it )
          {
            ObjectTreeIndex index = it->first;
            const ObjectTreeNode& node = m_objectTree[ index ];

            const Mat44f modelToWorld = getTransformMatrix( node.m_transformIndex );
            const Mat44f modelToView = modelToWorld * worldToView;
            ObjectTreeIndex activeIndex = it->second->getLODToUse( modelToView, scaleFactor );

            ObjectTreeIndex childIndex = m_objectTree[index].m_firstChild;
            // counter for the i-th child
            size_t i = 0;

            while( childIndex != ~0 )
            {
              ObjectTreeNode& childNode = m_objectTree[childIndex];
              DP_ASSERT( childNode.m_parentIndex == index );

              bool newActive = activeIndex == i;
              if ( childNode.m_localActive != newActive )
              {
                childNode.m_localActive = newActive;
                m_objectTree.markDirty( childIndex, ObjectTreeNode::DEFAULT_DIRTY );
              }

              childIndex = childNode.m_nextSibling;
              ++i;
            }  
          }
        }

        //
        // second step: update resulting node-world information
        // 

        UpdateObjectVisitor objectVisitor( m_objectTree, this );
        PreOrderTreeTraverser<ObjectTree, UpdateObjectVisitor> objectTraverser;
        
        objectTraverser.processDirtyList( m_objectTree, objectVisitor, ObjectTreeNode::DEFAULT_DIRTY );
        m_objectTree.m_dirtyObjects.clear();
      }

      TransformTreeIndex SceneTree::addTransform( const TransformTreeNode & node, TransformTreeIndex parentIndex, TransformTreeIndex siblingIndex )
      {
        // add transform to transform tree
        TransformTreeIndex index = m_transformTree.insertNode( node, parentIndex, siblingIndex );

        // observe transform
        if( node.m_transform )
        {
          m_transformObserver->attach( node.m_transform, index );
        }

        return index;
      }

      ObjectTreeIndex SceneTree::addObject( const ObjectTreeNode & node, ObjectTreeIndex parentIndex, ObjectTreeIndex siblingIndex )
      {
        // add object to object tree
        ObjectTreeIndex index = m_objectTree.insertNode( node, parentIndex, siblingIndex );

        // observe object
        DP_ASSERT( node.m_object );
        m_objectObserver->attach( node.m_object, index );

        return index;
      }

      void SceneTree::markTransformDynamic( TransformTreeIndex index )
      {
        m_dynamicTransformIndices.insert(index);
      }

      void SceneTree::addLOD( LODSharedPtr const& lod, ObjectTreeIndex index )
      {
        DP_ASSERT( m_objectTree.m_LODs.find(index) == m_objectTree.m_LODs.end() );
        m_objectTree.m_LODs[index] = lod.getWeakPtr();
      }

      void SceneTree::addSwitch( const SwitchSharedPtr& s, ObjectTreeIndex index )
      {
        DP_ASSERT( m_objectTree.m_switchNodes.find(index) == m_objectTree.m_switchNodes.end() );
        m_objectTree.m_switchNodes[index] = s.getWeakPtr();

        // attach switch observer to switch
        m_switchObserver->attach( s, index );
      }

      void SceneTree::addGeoNode( ObjectTreeIndex index )
      {
        // attach observer
        m_objectTree[index].m_isDrawable = true;
        m_geoNodeObserver->attach( weakPtr_cast<GeoNode>(m_objectTree[index].m_object), index );
        notify( EventObject( index, m_objectTree[index], EventObject::Added ) );
      }

      void SceneTree::addLightSource( ObjectTreeIndex index )
      {
        m_lightSources.insert(index);
      }

      void SceneTree::drawableInstanceUpdate( ObjectTreeIndex index )
      {
        DP_ASSERT( m_objectTree[index].m_isDrawable );
        notify( EventObject( index, m_objectTree[index], EventObject::Changed) );
      }

      void SceneTree::drawableInstanceRemove( ObjectTreeIndex index )
      {
        if ( m_objectTree[index].m_isDrawable )
        {
          notify( EventObject( index, m_objectTree[index], EventObject::Removed) );
          m_geoNodeObserver->detach( index );
        }
      }

      void SceneTree::removeObjectTreeIndex( ObjectTreeIndex index )
      {
        // initialize the trafo index for the trafo search with the parent's trafo index
        DP_ASSERT( index != m_objectTreeSentinel && "cannot remove root node" );
        TransformTreeIndex trafoIndex = m_objectTree[m_objectTree[index].m_parentIndex].m_transformIndex;

        // vector for stack-simulation to eliminate overhead of std::stack
        m_objectIndexStack.resize( m_objectTree.size() );
        size_t begin = 0;
        size_t end   = 0;

        // start traversal at index
        m_objectIndexStack[end] = index;
        ++end;

        while( begin != end ) 
        {
          ObjectTreeIndex currentIndex = m_objectIndexStack[begin];
          ++begin;
          ObjectTreeNode& current = m_objectTree[currentIndex];

          if ( isPtrTo<LightSource>( m_objectTree[currentIndex].m_object ) )
          {
            DP_VERIFY( m_lightSources.erase( currentIndex ) == 1 );
          }

          drawableInstanceRemove( currentIndex );

          current.m_clipPlaneGroup.reset();

          // detach current index from object observer
          m_objectObserver->detach( currentIndex );

          // TODO: add observer flag to specify which observers must be detached?
          std::map< ObjectTreeIndex, SwitchWeakPtr >::iterator itSwitch = m_objectTree.m_switchNodes.find( currentIndex );
          if ( itSwitch != m_objectTree.m_switchNodes.end() )
          {
            m_switchObserver->detach( currentIndex );
            m_objectTree.m_switchNodes.erase( itSwitch );
          }

          std::map< ObjectTreeIndex, LODWeakPtr >::iterator itLod = m_objectTree.m_LODs.find( currentIndex );
          if ( itLod != m_objectTree.m_LODs.end() )
          {
            m_objectTree.m_LODs.erase( itLod );
          }

          // check if a transform needs to be removed
          DP_ASSERT( current.m_parentIndex != ~0 );
          const ObjectTreeNode& curParent = m_objectTree[current.m_parentIndex];
          // only remove the topmost transforms below or at index (so transforms are only removed once)
          if(  current.m_transformIndex != curParent.m_transformIndex // transform indices differ
            && curParent.m_transformIndex == trafoIndex )             // parent index is parent index of removal root
          {
            removeTransformTreeIndex( current.m_transformIndex );
          }

          current.m_object = nullptr;

          // insert all children into stack for further traversal
          ObjectTreeIndex child = current.m_firstChild;
          while( child != ~0 )
          {
            m_objectIndexStack[end] = child;
            ++end;
            child = m_objectTree[child].m_nextSibling;
          }
        }

        // delete the node and its children from the object tree
        m_objectTree.deleteNode( index );
      }

      void SceneTree::removeTransformTreeIndex( TransformTreeIndex index )
      {
        // vector for stack-simulation to eliminate overhead of std::stack
        m_transformIndexStack.resize( m_transformTree.size() );
        size_t begin = 0;
        size_t end   = 0;

        // start traversal at index
        m_transformIndexStack[end] = index;
        ++end;

        while( begin != end )
        {
          TransformTreeIndex currentIndex = m_transformIndexStack[begin];
          ++begin;
          TransformTreeNode& current = m_transformTree[currentIndex];

          // detach current index from transform observer
          m_transformObserver->detach( currentIndex );
          // remove reference on this index in dynamic transform set
          m_dynamicTransformIndices.erase( index );

          // insert all children into stack
          TransformTreeIndex child = current.m_firstChild;
          while( child != ~0 )
          {
            m_transformIndexStack[end] = child;
            ++end;
            child = m_transformTree[child].m_nextSibling;
          }
        }

        // delete the node and its children from the transform tree
        m_transformTree.deleteNode( index );
      }

      TransformTree const& SceneTree::getTransformTree() const
      {
        return m_transformTree;
      }

      ObjectTree& SceneTree::getObjectTree()
      {
        return m_objectTree;
      }

      TransformTreeNode& SceneTree::getTransformTreeNodeInternal( TransformTreeIndex index )
      {
        return m_transformTree[index];
      }

      TransformTreeNode const& SceneTree::getTransformTreeNode( TransformTreeIndex index ) const
      {
        return m_transformTree[index];
      }

      ObjectTreeNode& SceneTree::getObjectTreeNode( ObjectTreeIndex index )
      {
        return m_objectTree[index];
      }

      void SceneTree::notifyTransformUpdated( TransformTreeIndex index, TransformTreeNode const& node )
      {
        notify( EventTransform( index, node ) );
      }

      void SceneTree::onRootNodeChanged()
      {
        replaceSubTree( m_scene->getRootNode().getWeakPtr(), m_objectTreeRootNode );
        m_rootNode = m_scene->getRootNode();
      }

    } // namespace xbar
  } // namespace sg
} // namespace dp
