// Copyright NVIDIA Corporation 2010-2012
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

#include <dp/sg/xbar/xbar.h>
#include <dp/sg/xbar/TransformTree.h>
#include <dp/sg/xbar/ObjectTree.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/algorithm/TransformStack.h>
#include <deque>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {
      DEFINE_PTR_TYPES( SceneTree );

      class DrawableInstance;
      typedef DrawableInstance* DrawableInstanceWeakPtr;

      DEFINE_PTR_TYPES( GeneratorState );

      class GeneratorState
      {
      public:
        static GeneratorStateSharedPtr create( SceneTreeSharedPtr const& sceneTree )
        {
          return( std::shared_ptr<GeneratorState>( new GeneratorState( sceneTree ) ) );
        }

      public:
        // set current environment for transform and object tree
        DP_SG_XBAR_API void setCurrentTransformTreeData( TransformTreeIndex parentIndex, TransformTreeIndex siblingIndex );
        DP_SG_XBAR_API void setCurrentObjectTreeData( ObjectTreeIndex parentIndex, ObjectTreeIndex siblingIndex );  

        // build up transform tree
        DP_SG_XBAR_API void pushTransform( dp::sg::core::TransformSharedPtr const& t );
        DP_SG_XBAR_API void pushTransform( dp::sg::core::BillboardSharedPtr const& bb );
        DP_SG_XBAR_API void popTransform();
        DP_SG_XBAR_API TransformTreeIndex getParentTransformIndex() const;
        DP_SG_XBAR_API TransformTreeIndex getSiblingTransformIndex() const;

        DP_SG_XBAR_API void addDynamicTransformIndex( TransformTreeIndex index );

        // build up object tree
        DP_SG_XBAR_API void pushObject( dp::sg::core::GroupSharedPtr const& group );
        DP_SG_XBAR_API void pushObject( dp::sg::core::LODSharedPtr const& lod );
        DP_SG_XBAR_API void pushObject( dp::sg::core::SwitchSharedPtr const& sw );
        DP_SG_XBAR_API void popObject();    

        // add GeoNode as leaf under object tree, returns the index in the tree
        DP_SG_XBAR_API ObjectTreeIndex addGeoNode( dp::sg::core::GeoNodeSharedPtr const& geoNode );

        // add LightSource as leaf under object tree, returns the index in the tree
        DP_SG_XBAR_API ObjectTreeIndex addLightSource( dp::sg::core::LightSourceSharedPtr const& lightSource );

        // data access functions
        DP_SG_XBAR_API ObjectTreeIndex getParentObjectIndex() const;
        DP_SG_XBAR_API ObjectTreeIndex getSiblingObjectIndex() const;
        
        // clip plane handling
        DP_SG_XBAR_API void pushClipPlaneSet();
        DP_SG_XBAR_API void popClipPlaneSet();
        DP_SG_XBAR_API void addClipPlane( const ClipPlaneInstanceSharedPtr& clipPlane );
        DP_SG_XBAR_API size_t getNumberOfClipPlanes() const;
        DP_SG_XBAR_API const ClipPlaneInstanceSharedPtr& getClipPlane( unsigned int i ) const;
        DP_SG_XBAR_API const SmartClipPlaneGroup& getClipPlaneGroup();

      protected:
        GeneratorState( SceneTreeSharedPtr const& sceneTree );

      private:
        typedef std::stack< std::pair< TransformTreeIndex, TransformTreeIndex> > TransformParentSiblingStack;
        typedef std::stack< std::pair< ObjectTreeIndex,    ObjectTreeIndex > >   ObjectParentSiblingStack;

      private:
        // add a node in the tree, update sibling information but don't push the object onto the parent stack
        ObjectTreeIndex insertNode( dp::sg::core::ObjectSharedPtr const& g );

      private:
        SceneTreeSharedPtr            m_sceneTree;

        TransformParentSiblingStack   m_transformParentSiblingStack; // A stack of indices of the current parent and sibling nodes
        ObjectParentSiblingStack      m_objectParentSiblingStack;    // A stack of indices of the current parent and sibling nodes

        //TODO0 initialize these two before inserting
        std::vector< SmartClipPlaneGroup> m_clipPlaneGroups;         // A vector of groups of ClipPlanes
      };

    } // namespace xbar
  } // namespace sg
} // namespace dp
