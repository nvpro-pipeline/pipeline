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


#include <dp/sg/xbar/inc/GeneratorState.h>
#include <dp/sg/xbar/SceneTree.h>

#include <dp/math/Boxnt.h>
#include <dp/math/Trafo.h>
#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/Camera.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>

using namespace dp::sg::core;

using namespace std;

namespace
{
  template <typename T>
  bool compareLessSharedPtrDereferenced(const T &lhs, const T &rhs )
  {
    return *lhs < *rhs;
  }
}

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      GeneratorState::GeneratorState( SceneTreeSharedPtr const& sceneTree )
        : m_sceneTree( sceneTree )
      {
      }

      void GeneratorState::setCurrentObjectTreeData( ObjectTreeIndex parentIndex, ObjectTreeIndex siblingIndex )
      {
        m_objectParentSiblingStack.push( make_pair( parentIndex, siblingIndex ) );

        ObjectTreeNode &node = m_sceneTree->getObjectTree()[parentIndex];

        // push ClipPlaneGroup state of parent as starting state
        m_clipPlaneGroups.push_back( node.m_clipPlaneGroup );
      }

      ObjectTreeIndex GeneratorState::insertNode( ObjectSharedPtr const& o )
      {
        ObjectTreeIndex parentIndex = getParentObjectIndex();
        ObjectTreeIndex siblingIndex = getSiblingObjectIndex();

        // Create node and fill with information
        ObjectTreeNode node;
        node.m_object         = o.getWeakPtr();
        node.m_clipPlaneGroup = m_clipPlaneGroups.back();

        // add node to tree
        ObjectTreeIndex index = m_sceneTree->addObject( node, parentIndex, siblingIndex );

        // update info for next node insertion
        if( !m_objectParentSiblingStack.empty() )
        {
          m_objectParentSiblingStack.top().second = index;
        } 

        return index;
      }

      void GeneratorState::pushObject( GroupSharedPtr const& group )
      {
        ObjectTreeIndex index = insertNode( group );

        m_objectParentSiblingStack.push( make_pair( index, ~0 ) );

        // TODO It's most likely best to move this logic to the SceneTree
        ObjectTree &ot = m_sceneTree->getObjectTree();
        ObjectTreeNode &otn = m_sceneTree->getObjectTreeNode(index);
        ObjectTreeNode &otnParent = m_sceneTree->getObjectTreeNode(otn.m_parentIndex);

        // handle Transform nodes
        if (group.isPtrTo<dp::sg::core::Transform>() || group.isPtrTo <dp::sg::core::Billboard>())
        {
          if (group.isPtrTo<dp::sg::core::Billboard>())
          {
            otn.m_isBillboard = true;
          }
          otn.m_isTransform = true;
          otn.m_transformLevel = otnParent.m_transformLevel + 1;
          otn.m_transform = m_sceneTree->allocateTransform();
          dp::sg::xbar::SceneTree::TransformEntry te;
          if (group.isPtrTo<dp::sg::core::Transform>()) {
            te.local = group.inplaceCast<dp::sg::core::Transform>()->getMatrix();
          }
          else
          {
            te.local = dp::math::cIdentity44f;
          }
          m_sceneTree->m_transforms.push_back(te);

          if (m_sceneTree->m_transformLevels.size() < (otn.m_transformLevel + 1)) {
            m_sceneTree->m_transformLevels.resize(otn.m_transformLevel + 1);
          }

          // add entry to list of transforms to work through
          if (group.isPtrTo<dp::sg::core::Billboard>())
          {
            dp::sg::xbar::SceneTree::BillboardListEntry ble;
            ble.parent = otnParent.m_transform;
            ble.transform = otn.m_transform;
            ble.billboard = group.inplaceCast<dp::sg::core::Billboard>();
            m_sceneTree->m_transformLevels[otn.m_transformLevel].billboardListEntries.push_back(ble);
          }
          else
          {
            dp::sg::xbar::SceneTree::TransformListEntry tle;
            tle.parent = otnParent.m_transform;
            tle.transform = otn.m_transform;
            m_sceneTree->m_transformLevels[otn.m_transformLevel].transformListEntries.push_back(tle);
          }
        }

        // if the parent objectNode is a transform set it as parent
        // otherwise copy the transform information as parent
        if (otnParent.m_isTransform) {
          otn.m_transformParent = otn.m_parentIndex;
        }
      }

      void GeneratorState::pushObject( LODSharedPtr const& lod )
      {
        ObjectTreeIndex index = insertNode( lod );

        m_objectParentSiblingStack.push( make_pair( index, ~0 ) );

        m_sceneTree->addLOD( lod, index );
      }

      void GeneratorState::pushObject( SwitchSharedPtr const& sw )
      {
        ObjectTreeIndex index = insertNode( sw );

        m_objectParentSiblingStack.push( make_pair( index, ~0 ) );

        m_sceneTree->addSwitch( sw, index );
      }

      void GeneratorState::popObject()
      {
        m_objectParentSiblingStack.pop();
      }

      ObjectTreeIndex GeneratorState::addGeoNode( GeoNodeSharedPtr const& geoNode )
      {
        // only insert the node, don't alter parent information (node is a leaf)
        ObjectTreeIndex index = insertNode( geoNode );

        ObjectTreeNode &otn = m_sceneTree->getObjectTreeNode(index);
        ObjectTreeNode &otnParent = m_sceneTree->getObjectTreeNode(otn.m_parentIndex);
        otn.m_transform = otnParent.m_transform;
        otn.m_transformLevel = otnParent.m_transformLevel;

        m_sceneTree->addGeoNode( index );

        return index;
      }

      ObjectTreeIndex GeneratorState::addLightSource( LightSourceSharedPtr const& lightSource )
      {
        // only insert the node, don't alter parent information (node is a leaf)
        ObjectTreeIndex index = insertNode( lightSource );

        m_sceneTree->addLightSource( index );

        return index;
      }

      ObjectTreeIndex GeneratorState::getParentObjectIndex() const
      {
        if( !m_objectParentSiblingStack.empty() )
        {
          return m_objectParentSiblingStack.top().first;
        }
        else
        {
          return ~0;
        }
      }

      ObjectTreeIndex GeneratorState::getSiblingObjectIndex() const
      {
        if( !m_objectParentSiblingStack.empty() )
        {
          return m_objectParentSiblingStack.top().second;
        }
        else
        {
          return ~0;
        }
      }

      void GeneratorState::pushClipPlaneSet()
      {
        m_clipPlaneGroups.push_back( ClipPlaneGroup::create( m_clipPlaneGroups.back()));

        // store LightGroup in current node
        ObjectTreeNode &node = m_sceneTree->getObjectTree()[getParentObjectIndex()];
        node.m_clipPlaneGroup = m_clipPlaneGroups.back();
      }

      void GeneratorState::popClipPlaneSet()
      {
        DP_ASSERT( !m_clipPlaneGroups.empty() );

        m_clipPlaneGroups.pop_back();
      }

      void GeneratorState::addClipPlane( const ClipPlaneInstanceSharedPtr & clipPlane )
      {
        DP_ASSERT( !m_clipPlaneGroups.empty() );

        // add clipPlane to the current set
        m_clipPlaneGroups.back()->add( clipPlane );
      }

      size_t GeneratorState::getNumberOfClipPlanes() const
      {
        DP_ASSERT( !m_clipPlaneGroups.empty() );
        return m_clipPlaneGroups.back()->getVector().size();
      }

      const ClipPlaneInstanceSharedPtr & GeneratorState::getClipPlane( unsigned int i ) const
      {
        DP_ASSERT( !m_clipPlaneGroups.empty() );
        return m_clipPlaneGroups.back()->getVector().at(i);
      }

      // FIXME should be const, but would require mutable.
      const SmartClipPlaneGroup& GeneratorState::getClipPlaneGroup()
      {
        DP_ASSERT( !m_clipPlaneGroups.empty() );

        return m_clipPlaneGroups.back();
      }

    } // namespace xbar
  } // namespace sg
} // namespace dp
