// Copyright NVIDIA Corporation 2010-2015
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
#include <dp/sg/xbar/inc/UpdateObjectVisitor.h>
#include <dp/sg/xbar/inc/SceneTreeGenerator.h>

// observers
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
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>

#include <dp/util/FrameProfiler.h>
#include <dp/util/BitArray.h>

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
        , m_firstTransformUpdate(true)
      {
      }

      SceneTree::~SceneTree()
      {
        m_sceneObserver.reset();
      }

      SceneTreeSharedPtr SceneTree::create( SceneSharedPtr const & scene )
      {
        SceneTreeSharedPtr st = std::shared_ptr<SceneTree>( new SceneTree( scene ) );
        st->init();
        return( st );
      }

      void SceneTree::init()
      {
        m_objectObserver = ObjectObserver::create( shared_from_this() );
        m_transformObserver = TransformObserver::create( shared_from_this() );
        m_sceneObserver = SceneObserver::create( shared_from_this() );

        // push a sentinel root group in the vector to avoid special cases for the real root-node later on
        ObjectTreeNode objectTreeSentinel;
        objectTreeSentinel.m_transform = allocateTransform();
        objectTreeSentinel.m_transformLevel = -1;
        objectTreeSentinel.m_transformParent = -1;
        objectTreeSentinel.m_clipPlaneGroup = ClipPlaneGroup::create();
        m_objectTreeSentinel = m_objectTree.insertNode( objectTreeSentinel, ~0, ~0 );

        TransformEntry te;
        te.local = cIdentity44f;
        te.world = cIdentity44f;
        m_transforms.push_back(te);

        SceneTreeGenerator rlg( this->shared_from_this() );
        rlg.setCurrentObjectTreeData( m_objectTreeSentinel, ~0 );
        rlg.apply( m_scene );

        // root node is first child below sentinel
        m_objectTreeRootNode = m_objectTree[m_objectTreeSentinel].m_firstChild;
      }

      dp::sg::core::SceneSharedPtr const & SceneTree::getScene() const
      {
        return m_scene;
      }

      void SceneTree::addSubTree( NodeSharedPtr const& root, ObjectTreeIndex parentIndex, ObjectTreeIndex leftSibling)
      {
        SceneTreeGenerator rlg( this->shared_from_this() );

        rlg.setCurrentObjectTreeData( parentIndex, leftSibling );
        rlg.apply( root );
      }

      void SceneTree::replaceSubTree( NodeSharedPtr const& node, ObjectTreeIndex objectIndex )
      {
        ObjectTreeIndex objectParent = m_objectTree[objectIndex].m_parentIndex;

        DP_ASSERT( objectParent != ~0 );

        // search left ObjectTree sibling
        ObjectTreeIndex objectLeftSibling = ~0;
        ObjectTreeIndex currentIndex = m_objectTree[objectParent].m_firstChild;
        while ( currentIndex != objectIndex )
        {
          objectLeftSibling = currentIndex;
          currentIndex = m_objectTree[currentIndex].m_nextSibling;
        }

        // remove group in both trees
        removeObjectTreeIndex( objectIndex );

        // add group back to tree
        addSubTree( node, objectParent, objectLeftSibling);
      }

      void SceneTree::addRendererOptions( const dp::sg::ui::RendererOptionsSharedPtr& rendererOptions )
      {
      }

      void SceneTree::update(dp::sg::core::CameraSharedPtr const& camera, float lodScaleRange)
      {
        // for now it is important to update the transform tree first to clear the DIRTY_TRANSFORM bit
        {
          dp::util::ProfileEntry p("Update TransformTree");
          updateTransformTree(camera);
        }

        {
          dp::util::ProfileEntry p("Update ObjectTree");
          updateObjectTree(camera, lodScaleRange);
        }
      }

      void SceneTree::updateTransformTree(dp::sg::core::CameraSharedPtr const& camera)
      {
        dp::util::BitArray dirty(m_transforms.size());
        dirty.clear();
        TransformObserver::DirtyPayloads const & dirtyPayloads = m_transformObserver->getDirtyPayloads();
        for (TransformObserver::DirtyPayloads::const_iterator it = dirtyPayloads.begin(); it != dirtyPayloads.end(); ++it)
        {
          ObjectTreeIndex index = (*it)->m_index;
          ObjectTreeNode const& otn = getObjectTreeNode(index);
          m_transforms[otn.m_transform].local = otn.m_object.getSharedPtr().inplaceCast<dp::sg::core::Transform>()->getMatrix();
          dirty.enableBit(otn.m_transform);
          // TODO make efficient dirty
          (*it)->m_dirty = false;
        }
        m_transformObserver->clearDirtyPayloads();

        int level = 0;
        m_changedTransforms.clear();

        for (TransformLevel const &transformLevel : m_transformLevels)
        {
          // update billboards
          for (BillboardListEntry const &billboardEntry : transformLevel.billboardListEntries)
          {
            dirty.enableBit(billboardEntry.transform);

            dp::math::Mat44f parentMatrix = m_transforms[billboardEntry.parent].world;
            parentMatrix.invert();
            dp::math::Trafo t = billboardEntry.billboard->getTrafo(camera, parentMatrix);
            m_transforms[billboardEntry.transform].local = t.getMatrix(); 

            m_transforms[billboardEntry.transform].world = m_transforms[billboardEntry.transform].local * m_transforms[billboardEntry.parent].world;
            m_changedTransforms.push_back(billboardEntry.transform);
            notifyTransformUpdated(billboardEntry.transform, m_transforms[billboardEntry.transform]);
          }

          // update transforms
          for (TransformListEntry const &transformEntry : transformLevel.transformListEntries)
          {
            if (dirty.getBit(transformEntry.parent) || dirty.getBit(transformEntry.transform))
            {
              dirty.enableBit(transformEntry.transform);
              m_transforms[transformEntry.transform].world = m_transforms[transformEntry.transform].local * m_transforms[transformEntry.parent].world;
              m_changedTransforms.push_back(transformEntry.transform);
              notifyTransformUpdated(transformEntry.transform, m_transforms[transformEntry.transform]);
            }
          }
        }

        if (m_firstTransformUpdate) {
          m_changedTransforms.push_back(0);
          notifyTransformUpdated(0, m_transforms[0]);
          m_firstTransformUpdate = false;
        }
      }

      void SceneTree::updateObjectTree(dp::sg::core::CameraSharedPtr const& camera, float lodRangeScale)
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

            SwitchSharedPtr ssp = m_objectTree.m_switchNodes[ index ].getSharedPtr();
            DP_ASSERT( ssp );

            ObjectTreeIndex childIndex = m_objectTree[index].m_firstChild;
            // counter for the i-th child
            size_t i = 0;

            while( childIndex != ~0 )
            {
              ObjectTreeNode& childNode = m_objectTree[childIndex];
              DP_ASSERT( childNode.m_parentIndex == index );

              bool newActive = ssp->isActive( dp::checked_cast<unsigned int>(i) );
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
          const Mat44f& worldToView = camera->getWorldToViewMatrix();

          std::map< ObjectTreeIndex, LODWeakPtr >::iterator it, it_end = m_objectTree.m_LODs.end();
          for( it = m_objectTree.m_LODs.begin(); it != it_end; ++it )
          {
            ObjectTreeIndex index = it->first;
            const ObjectTreeNode& node = m_objectTree[ index ];

            const Mat44f modelToWorld = getTransformMatrix(node.m_transform);
            const Mat44f modelToView = modelToWorld * worldToView;
            ObjectTreeIndex activeIndex = it->second->getLODToUse( modelToView, lodRangeScale );

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

      ObjectTreeIndex SceneTree::addObject( const ObjectTreeNode & node, ObjectTreeIndex parentIndex, ObjectTreeIndex siblingIndex )
      {
        // add object to object tree
        ObjectTreeIndex index = m_objectTree.insertNode( node, parentIndex, siblingIndex );

        // observe object
        m_objectObserver->attach( node.m_object.getSharedPtr(), index );

        if (node.m_object.getSharedPtr().isPtrTo<dp::sg::core::Transform>())
        {
          m_transformObserver->attach(node.m_object.getSharedPtr().inplaceCast<dp::sg::core::Transform>(), index);
        }

        return index;
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
        notify( EventObject( index, m_objectTree[index], EventObject::Added ) );
      }

      void SceneTree::addLightSource( ObjectTreeIndex index )
      {
        m_lightSources.insert(index);
      }

      void SceneTree::removeObjectTreeIndex( ObjectTreeIndex index )
      {
        // initialize the trafo index for the trafo search with the parent's trafo index
        DP_ASSERT( index != m_objectTreeSentinel && "cannot remove root node" );

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

          if ( m_objectTree[currentIndex].m_object.getSharedPtr().isPtrTo<dp::sg::core::LightSource>() )
          {
            DP_VERIFY( m_lightSources.erase( currentIndex ) == 1 );
          }

          if ( m_objectTree[currentIndex].m_isDrawable )
          {
            notify( EventObject( currentIndex, m_objectTree[currentIndex], EventObject::Removed) );
            m_objectTree[index].m_isDrawable = false;
          }

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
          if(current.m_isTransform)
          {
            if (current.m_isBillboard)
            {
              // remove from billboard list
              BillboardListEntries &billboardEntries = m_transformLevels[current.m_transformLevel].billboardListEntries;
              for (size_t index = 0; index < billboardEntries.size(); ++index)
              {
                if (billboardEntries[index].transform == current.m_transform)
                {
                  billboardEntries[index] = billboardEntries.back();
                  billboardEntries.pop_back();
                  break;
                }
              }
            }
            else
            {
              // remove from transform list
              TransformListEntries &transformEntries = m_transformLevels[current.m_transformLevel].transformListEntries;
              for (size_t index = 0; index < transformEntries.size(); ++index)
              {
                if (transformEntries[index].transform == current.m_transform)
                {
                  transformEntries[index] = transformEntries.back();
                  transformEntries.pop_back();
                  break;
                }
              }
            }

            freeTransform(current.m_transform);
            if (!current.m_isBillboard) {
              m_transformObserver->detach(currentIndex);
            }
          }

          current.m_object.reset();

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

      ObjectTree& SceneTree::getObjectTree()
      {
        return m_objectTree;
      }

      ObjectTreeNode& SceneTree::getObjectTreeNode( ObjectTreeIndex index )
      {
        return m_objectTree[index];
      }

      void SceneTree::notifyTransformUpdated(TransformIndex index, dp::sg::xbar::SceneTree::TransformEntry const& node)
      {
        notify( EventTransform( index, node ) );
      }

      void SceneTree::onRootNodeChanged()
      {
        replaceSubTree( m_scene->getRootNode(), m_objectTreeRootNode );
        m_rootNode = m_scene->getRootNode();
      }

      TransformIndex SceneTree::allocateTransform()
      {
        TransformIndex firstFree = checked_cast<TransformIndex>(m_transformFreeVector.countLeadingZeroes());
        if (firstFree == m_transformFreeVector.getSize()) {
          m_transformFreeVector.resize(m_transformFreeVector.getSize() + 65536, true); // add space for 65536 new bits
        }
        DP_ASSERT(firstFree != m_transformFreeVector.getSize());
        m_transformFreeVector.disableBit(firstFree);
        return firstFree;
      }

      void SceneTree::freeTransform(TransformIndex transformIndex) {
        DP_ASSERT(transformIndex < m_transformFreeVector.getSize());
        m_transformFreeVector.enableBit(transformIndex);
      }


    } // namespace xbar
  } // namespace sg
} // namespace dp
