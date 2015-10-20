// Copyright (c) 2010-2015, NVIDIA CORPORATION. All rights reserved.
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


#pragma once

#include <dp/math/Trafo.h>
#include <dp/sg/xbar/xbar.h>
#include <dp/sg/xbar/ObjectTree.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/ClipPlane.h>
#include <dp/sg/ui/RendererOptions.h>
#include <dp/sg/xbar/TransformTree.h>

#include <vector>
#include <stack>
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      class SceneTree;

      class UpdateTransformVisitor;

      template <typename IndexType> class Observer;

      DEFINE_PTR_TYPES( SwitchObserver );
      DEFINE_PTR_TYPES( TransformObserver );
      DEFINE_PTR_TYPES( ObjectObserver );
      DEFINE_PTR_TYPES( SceneObserver );
      DEFINE_PTR_TYPES( SceneTree );

      /*===========================================================================*/
      class SceneTree : public dp::util::Subject, public std::enable_shared_from_this<SceneTree>
      {
      protected:
        SceneTree(const dp::sg::core::SceneSharedPtr & scene );

      public:
        class Event : public dp::util::Event
        {
        public:
          enum Type
          {
              Added
            , Removed
            , Changed
            , ActiveChanged
            , TraversalMaskChanged
          };

          Event(ObjectTreeIndex index, ObjectTreeNode const& node, Type subType)
            : m_type( subType )
            , m_index( index )
            , m_node( node )
          {
          }


          Type getType() const { return m_type; }
          ObjectTreeIndex getIndex() const { return m_index; }
          ObjectTreeNode const & getNode() const { return m_node; }

        protected:
          Type               m_type;
          ObjectTreeIndex       m_index;
          ObjectTreeNode const& m_node;

        };

      public:
        DP_SG_XBAR_API static SceneTreeSharedPtr create( dp::sg::core::SceneSharedPtr const & scene );
        virtual ~SceneTree();

        DP_SG_XBAR_API dp::sg::core::SceneSharedPtr const & getScene() const;

        DP_SG_XBAR_API void addSubTree(const dp::sg::core::NodeSharedPtr& root, ObjectTreeIndex parentIndex, ObjectTreeIndex leftSibling);

        /** \brief Replace a subtree by a another subtree.
            \param root node of the new subtree to put into the SceneTree
            \param nodeIndex index of the node which should be replaced by the new subtree
        **/
        DP_SG_XBAR_API void replaceSubTree( dp::sg::core::NodeSharedPtr const& root, ObjectTreeIndex nodeIndex );

        //! Add all renderer options required by this renderer to the given rendererOptions object.
        DP_SG_XBAR_API void addRendererOptions( const dp::sg::ui::RendererOptionsSharedPtr& rendererOptions );

        // TODO reimplement support? Not required due to two sided rendering...
        //bool isMirrorTransform( TransformIndex index ) const { return m_transformTree.operator[](index).m_worldBits & TransformTreeNode::ISMIRRORTRANSFORM; }
        bool isMirrorTransform(TransformIndex index) const { assert(!"not implemented");/*return m_transformTree.operator[](index).m_worldBits & TransformTreeNode::ISMIRRORTRANSFORM;*/ }

        DP_SG_XBAR_API void update(dp::sg::core::CameraSharedPtr const& camera, float lodScaleRange);

        //! Add a new object to the Tree
        DP_SG_XBAR_API ObjectTreeIndex addObject( const ObjectTreeNode & node, ObjectTreeIndex parentIndex, ObjectTreeIndex siblingIndex );

        // special functions to mark object tree indices as special nodes
        DP_SG_XBAR_API void addLOD( dp::sg::core::LODSharedPtr const& lod, ObjectTreeIndex index );
        DP_SG_XBAR_API void addSwitch(  const dp::sg::core::SwitchSharedPtr& s, ObjectTreeIndex index );
        DP_SG_XBAR_API void addGeoNode( ObjectTreeIndex index );
        DP_SG_XBAR_API void addLightSource( ObjectTreeIndex index );

        // remove an index and the tree below it from the object tree. removes all referenced DIs, detaches affected objects from
        // the ObjectObserver and removes all affected transforms
        // note: doesnt work for the root node, as this requires a SceneTree rebuild
        DP_SG_XBAR_API void removeObjectTreeIndex( ObjectTreeIndex index );

        DP_SG_XBAR_API ObjectTree& getObjectTree();
        DP_SG_XBAR_API ObjectTreeNode& getObjectTreeNode( ObjectTreeIndex index );

        const std::set< ObjectTreeIndex >& getLightSources() const { return m_lightSources; }
        TransformTree & getTransformTree() { return m_transformTree; }

      protected:
        // remove a transform from the transform array
        DP_SG_XBAR_API void removeTransform(TransformIndex index);
        DP_SG_XBAR_API void updateTransformTree(dp::sg::core::CameraSharedPtr const& camera);
        DP_SG_XBAR_API void updateObjectTree(dp::sg::core::CameraSharedPtr const& camera, float lodScaleRange);

        DP_SG_XBAR_API void onRootNodeChanged( );

      private:
        void init();

        friend class UpdateTransformVisitor;
        friend class UpdateObjectVisitor;
        friend class SceneObserver;
        friend class SceneGenerator;
        friend class GeneratorState;

        dp::sg::core::SceneSharedPtr m_scene;
        // keep a reference to the root node of the scene so that treplaceSubTree works if the root node of the scene gets exchanged.
        dp::sg::core::NodeSharedPtr  m_rootNode;

        bool m_dirty;

        //TODO check switchobserver for shared switches!
        ObjectObserverSharedPtr    m_objectObserver;
        SwitchObserverSharedPtr    m_switchObserver;
        SceneObserverSharedPtr     m_sceneObserver;


        ObjectTree                               m_objectTree;
        ObjectTreeIndex                          m_objectTreeSentinel;
        ObjectTreeIndex                          m_objectTreeRootNode;
        std::vector< ObjectTreeIndex >           m_objectIndexStack;    // temp variable

        std::set< ObjectTreeIndex >              m_lightSources;

        TransformTree m_transformTree;
      };

      /*===========================================================================*/
      class ClipPlaneInstance
      {
      protected:
        ClipPlaneInstance()
          : m_transformIndex(~0)
        {}

      public:
        static ClipPlaneInstanceSharedPtr create()
        {
          return( std::shared_ptr<ClipPlaneInstance>( new ClipPlaneInstance() ) );
        }

      public:
        bool operator<(const ClipPlaneInstance &rhs) const
        {
          return m_transformIndex < rhs.m_transformIndex || ( m_transformIndex == rhs.m_transformIndex && m_clipPlane < rhs.m_clipPlane);
        }

        bool operator==(const ClipPlaneInstance &rhs) const
        {
          return m_transformIndex == rhs.m_transformIndex && m_clipPlane == rhs.m_clipPlane;
        }

        TransformIndex m_transformIndex;
        dp::sg::core::ClipPlaneSharedPtr m_clipPlane;
      };

    } // namespace xbar
  } // namespace sg
} // namespace dp
